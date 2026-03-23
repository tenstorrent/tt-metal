# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRAM weight prefetcher for GLM-4.7 on Galaxy Wormhole (8x9 grid).

Adapted from Llama3 70B Galaxy prefetcher (prefetcher_common.py).
Uses SubDevice + GlobalCB infrastructure to overlap DRAM weight reads
with compute across layers.

Architecture (8x9 grid, x=0..7, y=0..8):
  - Prefetcher SubDevice: 12 sender cores (6 in col 6, 6 in col 7)
  - Worker SubDevice: columns 0-5 (54 cores, includes 24 receivers)
  - Unassigned: 6 cores in cols 6,7 (rows 2,5,8) — OK per Llama pattern
  - Receivers: 2 per sender in cols 4-5 (for col-6), cols 2-3 (for col-7)
  - Prefetcher runs on dedicated SubDevice, overlapping with worker compute
  - Worker includes origin (0,0) so matmul grids starting from (0,0) stay
    within worker SubDevice — no sub_device_id needed for matmul/linear ops.

Usage in model_tt.py decode:
  1. prefetcher.create_global_cb()
  2. garbage = ttnn.dram_prefetcher(tt_tensors, num_layers=N, global_cb=cb)
  3. mesh_device.set_sub_device_stall_group([worker_sub_device_id])
  4. for layer in layers: layer.forward(..., global_cb=cb, sub_device_id=worker_id)
  5. ttnn.deallocate(garbage)
"""
import os

import torch
import ttnn
from loguru import logger


def get_glm_core_ranges(mesh_device, num_global_cb_receivers: int = 2):
    """Core ranges for GLM-4.7 prefetcher on WH Galaxy (8x9 compute grid).

    Uses columns 6 and 7 as sender columns, giving workers a contiguous
    block of columns 0-5 (6 cols x 9 rows = 54 cores) that includes
    origin (0,0). This allows matmul grids starting from (0,0) to stay
    within the worker SubDevice without needing sub_device_id.

    Layout (8x9 grid, x=0..7, y=0..8):
      - Prefetcher SubDevice: 12 active sender cores (6 in col 6, 6 in col 7)
      - Worker SubDevice: columns 0-5 (54 cores, includes 24 receiver cores)
      - Unassigned: 3 cores in col 6 + 3 in col 7 (not in any SubDevice, OK per Llama pattern)
      - 24 receiver cores: 2 per sender, in cols 4-5 (for col-6 senders)
        and cols 2-3 (for col-7 senders)
    """
    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    logger.info("Prefetcher: device grid {}x{}", grid_x, grid_y)

    # 12 DRAM banks on WH — address tensor is replicated across these cores
    dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(12)]

    # 12 active sender cores: 6 in column 6, 6 in column 7.
    # Spread across rows 0-8 with gaps (rows 2,5,8 unassigned in each column).
    # The sender->DRAM bank mapping is handled by the address tensor sharding,
    # not by physical core position.
    all_sender_cores = [
        ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 1),
        ttnn.CoreCoord(6, 3), ttnn.CoreCoord(6, 4),
        ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 7),
        ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, 1),
        ttnn.CoreCoord(7, 3), ttnn.CoreCoord(7, 4),
        ttnn.CoreCoord(7, 6), ttnn.CoreCoord(7, 7),
    ]

    # Receiver cores: 2 per sender, adjacent in the worker SubDevice.
    # Col-6 senders -> receivers in cols 4,5 (same row, nearest worker cols).
    # Col-7 senders -> receivers in cols 2,3 (same row, non-overlapping).
    all_receiver_pairs = [
        (4, 0), (5, 0),   # for sender (6,0)
        (4, 1), (5, 1),   # for sender (6,1)
        (4, 3), (5, 3),   # for sender (6,3)
        (4, 4), (5, 4),   # for sender (6,4)
        (4, 6), (5, 6),   # for sender (6,6)
        (4, 7), (5, 7),   # for sender (6,7)
        (2, 0), (3, 0),   # for sender (7,0)
        (2, 1), (3, 1),   # for sender (7,1)
        (2, 3), (3, 3),   # for sender (7,3)
        (2, 4), (3, 4),   # for sender (7,4)
        (2, 6), (3, 6),   # for sender (7,6)
        (2, 7), (3, 7),   # for sender (7,7)
    ]

    # Build sender->receiver mapping: each sender -> CoreRangeSet with 1 CoreRange of 2 cores.
    # API: Sequence[tuple[CoreCoord, CoreRangeSet]]
    sender_receiver_mapping = []
    for i in range(12):
        sender = all_sender_cores[i]
        r0 = all_receiver_pairs[i * 2]
        r1 = all_receiver_pairs[i * 2 + 1]
        recv_crs = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(*r0), ttnn.CoreCoord(*r1)),
        ])
        sender_receiver_mapping.append((sender, recv_crs))

    # Dummy senders: fill remaining rows in sender cols (rows 2,5,8) so that
    # the global CB covers ALL cores in the matmul bounding box.
    # Following Llama's pattern (dummy_sender_cores + dummy_receiver_cores).
    #
    # Real pattern: col-6 sender at (6,y) → receivers at (4,y),(5,y)
    #               col-7 sender at (7,y) → receivers at (2,y),(3,y)
    # Dummies follow the same pattern for rows 2,5,8:
    dummy_sender_cores = [
        ttnn.CoreCoord(6, 2),
        ttnn.CoreCoord(6, 5),
        ttnn.CoreCoord(6, 8),
        ttnn.CoreCoord(7, 2),
        ttnn.CoreCoord(7, 5),
        ttnn.CoreCoord(7, 8),
    ]
    dummy_receiver_mapping = [
        # (6,2) → (4,2),(5,2)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 2), ttnn.CoreCoord(5, 2))]),
        # (6,5) → (4,5),(5,5)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 5), ttnn.CoreCoord(5, 5))]),
        # (6,8) → (4,8),(5,8)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 8), ttnn.CoreCoord(5, 8))]),
        # (7,2) → (2,2),(3,2)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 2))]),
        # (7,5) → (2,5),(3,5)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(3, 5))]),
        # (7,8) → (2,8),(3,8)
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(3, 8))]),
    ]
    for ds, dr in zip(dummy_sender_cores, dummy_receiver_mapping):
        sender_receiver_mapping.append((ds, dr))

    # Hop core (3,6) is used for NOC1 ring routing. In this layout, (3,6) is
    # already a receiver core (for sender (7,6)), so it's automatically in the
    # global CB — no special dummy mapping needed.

    # Sender core range set: active + dummy senders.
    all_senders_with_dummies = list(all_sender_cores) + dummy_sender_cores
    sender_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(c, c) for c in all_senders_with_dummies]
    )

    # Worker core range set: columns 0-5, rows 0-(grid_y-1).
    # Contiguous block including origin (0,0) — matmul grids from (0,0) stay
    # within worker SubDevice. Includes receiver cores.
    worker_core_range_set = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, grid_y - 1)),
    ])

    logger.info(
        "GLM prefetcher core layout: {} senders, {} receiver pairs, "
        "worker cols 0-5 rows 0-{}, sender cols 6,7",
        len(all_sender_cores), len(sender_receiver_mapping), grid_y - 1,
    )

    return (
        all_sender_cores,
        dram_cores,
        sender_core_range_set,
        all_receiver_pairs,
        worker_core_range_set,
        sender_receiver_mapping,
    )


class Glm4MoePrefetcherSetup:
    """DRAM weight prefetcher for GLM-4.7 decode on Galaxy Wormhole.

    Registers attention and/or MoE weights for async prefetch during decode.
    """

    def __init__(
        self,
        mesh_device,
        n_tensors_per_layer: int,
        n_layers: int,
    ):
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

        # Global CB — sized for the largest DRAM bank shard (single buffer).
        # WH Galaxy L1 per core is ~1.36 MB free, so double-buffering (1600 tiles)
        # doesn't fit. Use single-buffer (800 tiles) — prefetcher still overlaps
        # DRAM reads with compute, just can't pipeline two consecutive reads.
        # QKV bank shard = 800 tiles, O-proj bank shard = 672 tiles.
        self.global_cb_size = 800 * 1088
        self.global_circular_buffer = None

        # SubDevice setup — DEFERRED to ensure_ready() so that
        # create_sub_device_manager doesn't interfere with prefill CCL ops.
        # Prefill runs before decode and needs the full grid (no SubDevice restriction).
        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False

        self.tensors = []
        self.tensor_addrs = []

        # Ring matmul configs for prefetcher-aware decode.
        # Uses hop_cores=(3,6) for NOC1 ring routing (within worker cols 0-5).
        # Ring size must divide both K_tiles and N_tiles:
        # QKV: K=5120(160t), N=1792(56t), gcd=8 → num_cores=8
        # O-proj: K=1536(48t), N=5120(160t), gcd=16 → num_cores=16
        #
        # Ring cores are selected from the 24 receiver cores to ensure ALL
        # cores in the matmul's CB are within global_cb.all_cores().
        # Receiver layout: cols 4-5 (for col-6 senders), cols 2-3 (for col-7).
        qkv_ring_cores = list(self.receiver_cores[:8])  # first 4 pairs = 8 cores
        oproj_ring_cores = list(self.receiver_cores[:16])  # first 8 pairs = 16 cores
        self.qkv_ring_cores = qkv_ring_cores
        self.oproj_ring_cores = oproj_ring_cores

        self.qkv_program_config = self._make_ring_config(
            B=1, M=32, K=5120, N=1792, num_cores=8,
            ring_cores=qkv_ring_cores,
        )
        self.oproj_program_config = self._make_ring_config(
            B=1, M=32, K=1536, N=5120, num_cores=16,
            ring_cores=oproj_ring_cores,
        )

        # Input/output memory configs for ring matmul.
        # gather_in0=True requires:
        #   - in0 WIDTH_SHARDED on ring cores with shard=[M, K/num_cores]
        #   - output WIDTH_SHARDED with shard=[M, N/num_cores]
        # Cores are the exact receiver cores from the global CB mapping.
        self.qkv_input_mem_cfg = self._make_ring_mem_cfg(
            num_cores=8, M=32, shard_dim=5120, ring_cores=qkv_ring_cores,
        )
        self.qkv_output_mem_cfg = self._make_ring_mem_cfg(
            num_cores=8, M=32, shard_dim=1792, ring_cores=qkv_ring_cores,
        )
        self.oproj_input_mem_cfg = self._make_ring_mem_cfg(
            num_cores=16, M=32, shard_dim=1536, ring_cores=oproj_ring_cores,
        )
        self.oproj_output_mem_cfg = self._make_ring_mem_cfg(
            num_cores=16, M=32, shard_dim=5120, ring_cores=oproj_ring_cores,
        )

        logger.info(
            "Glm4MoePrefetcherSetup: n_tensors={}, n_layers={}, global_cb_size={}",
            n_tensors_per_layer, n_layers, self.global_cb_size,
        )

    @staticmethod
    def _make_ring_mem_cfg(num_cores, M, shard_dim, ring_cores):
        """Create L1 WIDTH_SHARDED MemoryConfig for ring matmul input or output.

        Uses the exact receiver cores from the global CB mapping to ensure
        the matmul's CB core ranges are a subset of global_cb.all_cores().
        """
        shard_w = shard_dim // num_cores
        assert len(ring_cores) == num_cores, (
            f"Expected {num_cores} ring cores, got {len(ring_cores)}"
        )
        core_range = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))
            for c in ring_cores
        ])
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range, [M, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )

    @staticmethod
    def _make_ring_config(B, M, K, N, num_cores, ring_cores):
        """Create MatmulMultiCoreReuseMultiCast1DProgramConfig for prefetch.

        Uses Llama Galaxy's ring matmul pattern with gather_in0=True.
        The num_cores must divide N_tiles = N/32.
        ring_cores: list of (x,y) tuples for the actual receiver cores.
        """
        tile = 32
        M *= B  # fuse_batch=True
        N_tiles = N // tile
        K_tiles = K // tile

        assert N_tiles % num_cores == 0, (
            f"N_tiles={N_tiles} not divisible by num_cores={num_cores}"
        )

        in0_block_w = K_tiles // num_cores
        out_block_w = N_tiles // num_cores
        out_block_h = M // tile

        sbw = min(8, out_block_w)
        while sbw > 0 and out_block_w % sbw != 0:
            sbw -= 1

        # Compute bounding box grid from the actual ring core positions.
        # The 1D program config uses compute_with_storage_grid_size as a
        # bounding box — it must contain all ring cores.
        max_x = max(c[0] for c in ring_cores)
        max_y = max(c[1] for c in ring_cores)
        gx = max_x + 1  # 0-indexed → size
        gy = max_y + 1

        # Hop cores for NOC1 ring routing (same as Llama Galaxy)
        hop_core_range_set = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(3, 6), ttnn.CoreCoord(3, 6)),
        ])

        logger.info("Prefetch ring config: K={}×N={}→M={} cores={} grid=({},{})",
                     K, N, M, num_cores, gx, gy)

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
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

    def create_global_cb(self):
        """Lazy-create the Global Circular Buffer (must be called before decode)."""
        if self.global_circular_buffer is None:
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )
            logger.info("Global CB created, size={}", self.global_cb_size)

    def insert_tensor(self, tensor: ttnn.Tensor):
        """Register a weight tensor for prefetching."""
        self.tensors.append(tensor)
        self.tensor_addrs.append(tensor.buffer_address())

    def get_input_tensors(self):
        """Build the address tensor and return [weights..., addr_tensor]."""
        assert len(self.tensor_addrs) == self.n_tensors * self.n_layers, (
            f"Expected {self.n_tensors * self.n_layers} addresses, "
            f"got {len(self.tensor_addrs)}"
        )

        tensor_addrs = torch.tensor(self.tensor_addrs)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_core_range_set,
                [tensor_addrs.shape[0] // len(self.dram_cores), tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_tensor_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=tensor_addrs_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return self.tensors[: self.n_tensors] + [tt_tensor_addrs]

    def ensure_ready(self):
        """One-time setup: load SubDevice manager + create GlobalCB.

        Call BEFORE trace capture. These are non-traceable setup ops that
        must happen once before the first decode.
        """
        if not self._sub_device_loaded:
            # Create SubDevice objects + manager on first call (deferred from __init__
            # to avoid interfering with prefill CCL ops on TG mesh).
            if self.mesh_sub_device_manager_id is None:
                prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
                worker_sub_device = ttnn.SubDevice([self.worker_core_range_set])
                self.mesh_sub_device_manager_id = self.mesh_device.create_sub_device_manager(
                    [prefetcher_sub_device, worker_sub_device], 0
                )
            self.mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id)
            # Use both SubDevices for setup ops (address tensor shards onto sender
            # cores, GlobalCB touches both SubDevices).
            self.mesh_device.set_sub_device_stall_group(
                [self.prefetcher_sub_device_id, self.worker_sub_device_id]
            )
            self._sub_device_loaded = True
            logger.info("Prefetcher: SubDevice manager created+loaded (deferred from init)")
        self.create_global_cb()
        # Build the address tensor once (reused in every start_prefetch call)
        if not hasattr(self, '_tt_tensors') or self._tt_tensors is None:
            self._tt_tensors = self.get_input_tensors()
        # After all setup, narrow stall group to worker-only so regular
        # decode ops dispatch to worker cores without hitting the
        # "Programs must be executed on a single sub-device" assertion.
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def compile_prefetch(self):
        """Pre-compile dram_prefetcher program OUTSIDE trace capture.

        Must be called after ensure_ready() and before begin_trace_capture().
        Runs dram_prefetcher once so the program is in the cache.  When the
        same op is issued inside trace capture it hits the cache and avoids
        any device-buffer writes (SetRuntimeArgs) that would violate trace constraints.

        NOTE: We do NOT synchronize after dram_prefetcher — the prefetcher
        kernel reads DRAM banks and fills the global CB, but with no consumer
        it stalls after filling the triple buffer.  synchronize_device would
        hang waiting for it.  Instead we just deallocate the garbage tensor
        (which frees the output buffer) and let the kernel drain naturally
        when the SubDevice manager is reset or the next operation runs.
        """
        # dram_prefetcher dispatches to BOTH SubDevices.
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_sub_device_id, self.worker_sub_device_id]
        )
        garbage = ttnn.dram_prefetcher(
            self._tt_tensors,
            num_layers=self.n_layers,
            global_cb=self.global_circular_buffer,
        )
        # Do NOT synchronize — prefetcher stalls without consumers.
        # Just deallocate the garbage tensor to free the output buffer.
        ttnn.deallocate(garbage)
        # Restore worker-only stall group for subsequent setup ops.
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        logger.info("Prefetcher: compile_prefetch() done — program cached")

    def start_prefetch(self):
        """Launch async DRAM prefetch. Call INSIDE trace capture.

        Assumes ensure_ready() and compile_prefetch() were already called
        before trace capture.  Only issues the cached dram_prefetcher op
        and adjusts the stall group — no device-buffer writes.
        """
        # dram_prefetcher dispatches to BOTH SubDevices (sender + receiver cores).
        # Temporarily expand stall group, then narrow back to worker-only.
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_sub_device_id, self.worker_sub_device_id]
        )
        garbage = ttnn.dram_prefetcher(
            self._tt_tensors,
            num_layers=self.n_layers,
            global_cb=self.global_circular_buffer,
        )
        # Only stall on worker cores (prefetcher runs independently)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        return garbage

    def stop_prefetch(self, garbage):
        """Stop prefetch and clean up.

        Only deallocates the garbage tensor.  Stall group remains [worker_only]
        (set by start_prefetch) so subsequent compute (norm, LM head, argmax)
        executes on worker cores.  Matches Llama Galaxy pattern.
        """
        ttnn.deallocate(garbage)
