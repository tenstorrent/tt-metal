# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DRAM weight prefetcher for GLM-4.7-Flash on Galaxy Wormhole (8x9 grid).

Ported from glm4_moe (REAP) prefetcher_setup.py. Uses SubDevice + GlobalCB
infrastructure to overlap DRAM weight reads with compute across decode layers.

STATUS: full module ported + device-verified (SubDevice manager + GlobalCB
construct/teardown cleanly on Flash's 8x9 grid). make_ring_config/make_ring_mem_cfg
are generic (per-MLA-weight num_cores still needs on-device tuning). The model-side
threading (register weights, consume global_cb in the decode matmuls, re-grid worker
ops) is NOT wired yet — the deep integration step. Gated by GLM4_MOE_LITE_PREFETCH;
when off this module is never imported on the hot path.

Grid (Flash == REAP == WH Galaxy 8x9, x=0..7 y=0..8):
  - Prefetcher SubDevice: 12 active sender cores (6 in col 6, 6 in col 7)
  - Worker SubDevice: columns 0-5 (54 cores, includes 24 receivers + origin)
  - Hop core (3,6) for NOC1 ring routing
"""

import torch
import ttnn
from loguru import logger


def get_glm_core_ranges(mesh_device, num_global_cb_receivers: int = 2):
    """Core ranges for the prefetcher on WH Galaxy (8x9). Verbatim from REAP —
    Flash shares the identical grid, so the layout transfers unchanged."""
    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    logger.info("Flash prefetcher: device grid {}x{}", grid_x, grid_y)

    dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(12)]

    all_sender_cores = [
        ttnn.CoreCoord(6, 0),
        ttnn.CoreCoord(6, 1),
        ttnn.CoreCoord(6, 3),
        ttnn.CoreCoord(6, 4),
        ttnn.CoreCoord(6, 6),
        ttnn.CoreCoord(6, 7),
        ttnn.CoreCoord(7, 0),
        ttnn.CoreCoord(7, 1),
        ttnn.CoreCoord(7, 3),
        ttnn.CoreCoord(7, 4),
        ttnn.CoreCoord(7, 6),
        ttnn.CoreCoord(7, 7),
    ]
    all_receiver_pairs = [
        (4, 0),
        (5, 0),
        (4, 1),
        (5, 1),
        (4, 3),
        (5, 3),
        (4, 4),
        (5, 4),
        (4, 6),
        (5, 6),
        (4, 7),
        (5, 7),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (2, 3),
        (3, 3),
        (2, 4),
        (3, 4),
        (2, 6),
        (3, 6),
        (2, 7),
        (3, 7),
    ]
    sender_receiver_mapping = []
    for i in range(12):
        r0 = all_receiver_pairs[i * 2]
        r1 = all_receiver_pairs[i * 2 + 1]
        recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*r0), ttnn.CoreCoord(*r1))])
        sender_receiver_mapping.append((all_sender_cores[i], recv_crs))

    dummy_sender_cores = [
        ttnn.CoreCoord(6, 2),
        ttnn.CoreCoord(6, 5),
        ttnn.CoreCoord(6, 8),
        ttnn.CoreCoord(7, 2),
        ttnn.CoreCoord(7, 5),
        ttnn.CoreCoord(7, 8),
    ]
    dummy_receiver_mapping = [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 2), ttnn.CoreCoord(5, 2))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 5), ttnn.CoreCoord(5, 5))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 8), ttnn.CoreCoord(5, 8))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 2))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(3, 5))]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(3, 8))]),
    ]
    for ds, dr in zip(dummy_sender_cores, dummy_receiver_mapping):
        sender_receiver_mapping.append((ds, dr))

    all_senders_with_dummies = list(all_sender_cores) + dummy_sender_cores
    sender_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_senders_with_dummies])
    worker_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, grid_y - 1))])
    return (
        all_sender_cores,
        dram_cores,
        sender_core_range_set,
        all_receiver_pairs,
        worker_core_range_set,
        sender_receiver_mapping,
    )


def make_ring_config(M, K, N, num_cores, ring_cores):
    """Generic ring matmul program config (gather_in0). num_cores MUST divide N/32.

    For Flash MLA weights each shape needs its own num_cores chosen so that
    N_tiles % num_cores == 0 AND len(ring_cores) == num_cores. This is the part
    that requires on-device validation per weight (wrong config hangs)."""
    tile = 32
    N_tiles = N // tile
    K_tiles = K // tile
    assert N_tiles % num_cores == 0, f"N_tiles={N_tiles} not divisible by num_cores={num_cores}"
    in0_block_w = K_tiles // num_cores
    out_block_w = N_tiles // num_cores
    out_block_h = M // tile
    sbw = min(8, out_block_w)
    while sbw > 0 and out_block_w % sbw != 0:
        sbw -= 1
    max_x = max(c[0] for c in ring_cores)
    max_y = max(c[1] for c in ring_cores)
    hop_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 6), ttnn.CoreCoord(3, 6))])
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(max_x + 1, max_y + 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sbw,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        num_global_cb_receivers=2,
    )


class Glm4MoeLitePrefetcherSetup:
    """Foundation: SubDevice split + GlobalCB. Model-side matmul threading TBD."""

    def __init__(self, mesh_device, n_tensors_per_layer: int, n_layers: int, global_cb_tiles: int = 900):
        self.mesh_device = mesh_device
        self.n_tensors = n_tensors_per_layer
        self.n_layers = n_layers
        (
            self.sender_cores,
            self.dram_cores,
            self.sender_core_range_set,
            self.receiver_cores,
            self.worker_core_range_set,
            self.sender_receiver_mapping,
        ) = get_glm_core_ranges(mesh_device)
        # Sized for the largest Flash prefetched weight bank shard (o_proj 5120x2048
        # bf8 ~ 853 tiles/bank); 900 tiles conservative and known to fit WH L1.
        self.global_cb_size = global_cb_tiles * 1088
        self.global_circular_buffer = None
        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False
        self.tensors = []
        self.tensor_addrs = []

    def create_global_cb(self):
        if self.global_circular_buffer is None:
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
            )
            logger.info("Flash Global CB created, size={}", self.global_cb_size)

    def insert_tensor(self, tensor: ttnn.Tensor):
        self.tensors.append(tensor)
        self.tensor_addrs.append(tensor.buffer_address())

    def ensure_ready(self):
        """One-time: load SubDevice manager + create GlobalCB. Call before trace capture."""
        if not self._sub_device_loaded:
            if self.mesh_sub_device_manager_id is None:
                prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
                worker_sub_device = ttnn.SubDevice([self.worker_core_range_set])
                self.mesh_sub_device_manager_id = self.mesh_device.create_sub_device_manager(
                    [prefetcher_sub_device, worker_sub_device], 0
                )
            self.mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id)
            self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device_id, self.worker_sub_device_id])
            self._sub_device_loaded = True
        self.create_global_cb()
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    @staticmethod
    def make_ring_mem_cfg(num_cores, M, shard_dim, ring_cores):
        """L1 WIDTH_SHARDED MemoryConfig for a ring matmul input/output, pinned to the
        exact receiver cores so the matmul CB is a subset of global_cb.all_cores()."""
        shard_w = shard_dim // num_cores
        assert len(ring_cores) == num_cores, f"Expected {num_cores} ring cores, got {len(ring_cores)}"
        core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in ring_cores])
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range, [M, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def get_input_tensors(self):
        """Build the uint32 address tensor (replicated onto the 12 sender cores) and
        return [weights..., addr_tensor] for dram_prefetcher."""
        assert (
            len(self.tensor_addrs) == self.n_tensors * self.n_layers
        ), f"Expected {self.n_tensors * self.n_layers} addresses, got {len(self.tensor_addrs)}"
        tensor_addrs = torch.tensor(self.tensor_addrs)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        addr_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_core_range_set,
                [tensor_addrs.shape[0] // len(self.dram_cores), tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=addr_mem_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return self.tensors[: self.n_tensors] + [tt_addrs]

    def compile_prefetch(self):
        """Pre-compile dram_prefetcher OUTSIDE trace capture (so the traced issue hits
        the program cache and writes no runtime args). Do NOT synchronize — the
        prefetcher stalls without a consumer; just free the garbage output."""
        if not hasattr(self, "_tt_tensors") or self._tt_tensors is None:
            self._tt_tensors = self.get_input_tensors()
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device_id, self.worker_sub_device_id])
        garbage = ttnn.dram_prefetcher(
            self._tt_tensors, num_layers=self.n_layers, global_cb=self.global_circular_buffer
        )
        ttnn.deallocate(garbage)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def start_prefetch(self):
        """Launch async prefetch INSIDE trace capture (issues the cached op only)."""
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device_id, self.worker_sub_device_id])
        garbage = ttnn.dram_prefetcher(
            self._tt_tensors, num_layers=self.n_layers, global_cb=self.global_circular_buffer
        )
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        return garbage

    def stop_prefetch(self, garbage):
        ttnn.deallocate(garbage)

    def teardown(self):
        """Reset the SubDevice manager (restore full-grid dispatch)."""
        try:
            self.mesh_device.reset_sub_device_stall_group()
        except Exception:
            pass
        if self.mesh_sub_device_manager_id is not None:
            try:
                self.mesh_device.remove_sub_device_manager(self.mesh_sub_device_manager_id)
            except Exception:
                pass
        self._sub_device_loaded = False
