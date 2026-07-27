# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DRAM weight prefetcher for GLM-4.7-Flash on Galaxy Wormhole (8x9 grid).

Overlaps decode weight reads with compute, attacking the largest remaining decode
bucket (matmul, ~39.5% of step time at ~20% M=1 bandwidth efficiency). Ported from
the working glm4_moe (REAP) implementation.

STATUS: the SubDevice + GlobalCB apparatus and the ring-config math are in place and
unit-tested on the host (see tests/test_prefetcher_config.py). The model-side
threading -- registering DRAM-sharded weights, consuming the GlobalCB in the decode
matmuls, and re-gridding every decode op onto the worker SubDevice -- is NOT wired
yet. Gated by GLM4_MOE_LITE_PREFETCH; when off this module is never imported on the
hot path.

Grid (8x9, x=0..7 y=0..8), matching REAP exactly:
  - Prefetcher SubDevice: 8 sender cores, column 6, rows 0-7
  - 16 receiver cores: columns 4-5, rows 0-7 (a contiguous 2x8 block)
  - Worker SubDevice: columns 0-5 (54 cores; contains the receivers and origin (0,0))
  - Unassigned: column 7 and (6,8) -- not in any SubDevice, which is fine

WHY 8 BANKS / 16 RECEIVERS AND NOT 12/24: a 12-bank x 2-receiver contract builds a
24-core ring. gather_in0 requires num_cores to divide BOTH K_tiles and N_tiles, and
24 divides neither of Flash's o_proj dims (160 x 64 tiles), so the ring deadlocks
waiting for pages the producer cannot send. REAP hit this exact deadlock and moved to
8 banks x 2 receivers = a 16-core ring. An earlier revision of this file carried
REAP's pre-fix 12/24 layout while claiming to be "verbatim from REAP"; it was not,
and it could not have worked. Do not reintroduce it.
"""

from __future__ import annotations

import math

import torch
import ttnn
from loguru import logger

TILE = 32
# Receiver cores per sender in the GlobalCB contract. Fixed by the sender->receiver
# mapping built below and passed to the matmul as num_global_cb_receivers.
NUM_GLOBAL_CB_RECEIVERS = 2
# Bytes per bfloat8_b tile (1024 elements + per-tile exponent metadata).
BF8_TILE_BYTES = 1088


def get_glm_core_ranges(mesh_device, num_global_cb_receivers: int = NUM_GLOBAL_CB_RECEIVERS):
    """Core ranges for the prefetcher on WH Galaxy (8x9).

    Column 6 holds the senders, leaving workers a contiguous columns 0-5 block that
    includes origin (0,0) -- so matmul grids anchored at (0,0) stay inside the worker
    SubDevice without needing an explicit sub_device_id.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    logger.info("Flash prefetcher: device grid {}x{}", grid_x, grid_y)

    # Eight DRAM banks x two receivers = a 16-core ring. See the module docstring for
    # why this is not 12.
    dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(8)]

    # Senders sit outside the rectangular worker SubDevice (columns 0-5).
    all_sender_cores = [ttnn.CoreCoord(6, y) for y in range(8)]

    # Receiver pairs are bank-major then row-major, matching gather_in0's ring walk.
    # A contiguous 2x8 block means the matmul's remote-CB core set exactly equals the
    # GlobalCB receiver set, so no dedicated hop core is needed.
    all_receiver_pairs = [(x, y) for y in range(8) for x in (4, 5)]

    sender_receiver_mapping = []
    for i, sender in enumerate(all_sender_cores):
        r0 = all_receiver_pairs[i * num_global_cb_receivers]
        r1 = all_receiver_pairs[i * num_global_cb_receivers + 1]
        recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*r0), ttnn.CoreCoord(*r1))])
        sender_receiver_mapping.append((sender, recv_crs))

    sender_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_sender_cores])
    worker_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, grid_y - 1))])

    logger.info(
        "Flash prefetcher layout: {} senders (col 6), {} receivers (cols 4-5), worker cols 0-5 rows 0-{}",
        len(all_sender_cores),
        len(all_receiver_pairs),
        grid_y - 1,
    )
    return (
        all_sender_cores,
        dram_cores,
        sender_core_range_set,
        all_receiver_pairs,
        worker_core_range_set,
        sender_receiver_mapping,
    )


def ring_feasibility(K: int, N: int, max_cores: int = 16) -> list[int]:
    """Return the ring sizes that are valid for a (K, N) weight, largest first.

    gather_in0 shards the activation across the ring along K and the output along N,
    so num_cores must divide BOTH K_tiles and N_tiles exactly -- a truncating divide
    produces a ring that deadlocks rather than an error. The program config's grid is
    also built as (min(8, n), n // min(8, n)), so n must factor that way.

    Returns [] when a weight cannot be ring-prefetched at all. Flash's MLA weights:

        w_o        K=5120 N=2048  (160 x  64 tiles)  -> 16, 8, ...  USE 16
        w_q_b      K= 768 N=5120  ( 24 x 160 tiles)  ->  8          (later increment)
        w_q_kv_a   K=2048 N=1344  ( 64 x  42 tiles)  ->  []         infeasible (gcd 2)
        w_q_a      K=2048 N= 768  ( 64 x  24 tiles)  ->  8          only if unfused
        w_kv_a     K=2048 N= 576  ( 64 x  18 tiles)  ->  []         infeasible (gcd 2)

    w_kv_b1/w_kv_b2 are per-head 3D weights, not 2D matmuls, so the ring does not
    apply to them at all.
    """
    if K % TILE or N % TILE:
        return []
    k_tiles, n_tiles = K // TILE, N // TILE
    out = []
    for n in range(min(max_cores, math.gcd(k_tiles, n_tiles)), 0, -1):
        if k_tiles % n or n_tiles % n:
            continue
        gx = min(8, n)
        if gx == 0 or n % gx:
            continue
        out.append(n)
    return out


def global_cb_tiles_for(K: int, N: int, num_cores: int) -> int:
    """Per-receiver GlobalCB payload, in tiles, for one prefetched weight.

    Each receiver holds the full K extent for its slice of N:
        K_tiles * (N_tiles / num_cores)
    Sizing off DRAM bank count instead of receiver count (as an earlier revision of
    this file did) over-allocates and needlessly eats L1.
    """
    return (K // TILE) * ((N // TILE) // num_cores)


class Glm4MoeLitePrefetcherSetup:
    """SubDevice split + GlobalCB + ring configs. Model-side threading still TBD."""

    # First prototype scope: o_proj only. It is the largest 2D decode weight
    # (5120x2048 = 11.1 MB/layer in bf8, 2.4x the next candidate) and its 160x64
    # tile shape is dimensionally identical to REAP's QKV, which runs at
    # num_cores=16 with a 640-tile CB -- so the proven config transfers unchanged.
    # Keeping every prefetched weight on the same ring size avoids mixing ring
    # widths over one GlobalCB contract; w_q_b (8 cores) is a later increment.
    OPROJ_K = 5120
    OPROJ_N = 2048
    RING_CORES = 16

    def __init__(self, mesh_device, n_tensors_per_layer: int, n_layers: int, global_cb_tiles: int | None = None):
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

        assert (
            len(self.receiver_cores) == self.RING_CORES
        ), f"ring size {self.RING_CORES} must equal receiver count {len(self.receiver_cores)}"

        feasible = ring_feasibility(self.OPROJ_K, self.OPROJ_N, max_cores=self.RING_CORES)
        assert self.RING_CORES in feasible, (
            f"o_proj K={self.OPROJ_K} N={self.OPROJ_N} cannot use a {self.RING_CORES}-core ring "
            f"(feasible: {feasible}). A non-dividing ring deadlocks on device."
        )

        tiles = global_cb_tiles or global_cb_tiles_for(self.OPROJ_K, self.OPROJ_N, self.RING_CORES)
        self.global_cb_size = tiles * BF8_TILE_BYTES
        self.global_circular_buffer = None

        self.oproj_ring_cores = list(self.receiver_cores)
        self.oproj_program_config = self.make_ring_config(
            B=1, M=TILE, K=self.OPROJ_K, N=self.OPROJ_N, num_cores=self.RING_CORES
        )
        self.oproj_input_mem_cfg = self.make_ring_mem_cfg(
            num_cores=self.RING_CORES, M=TILE, shard_dim=self.OPROJ_K, ring_cores=self.oproj_ring_cores
        )
        self.oproj_output_mem_cfg = self.make_ring_mem_cfg(
            num_cores=self.RING_CORES, M=TILE, shard_dim=self.OPROJ_N, ring_cores=self.oproj_ring_cores
        )

        # Worker grids for re-gridding decode ops once the SubDevice is active.
        self.worker_scg = self.worker_core_range_set
        # hidden=2048 = 64 tiles; the committed sharded norm uses 8 cores.
        self.norm_core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])

        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False
        self._tt_tensors = None
        self.tensors = []
        self.tensor_addrs = []

        logger.info(
            "Glm4MoeLitePrefetcherSetup: n_tensors={} n_layers={} ring={} global_cb={} tiles ({} B)",
            n_tensors_per_layer,
            n_layers,
            self.RING_CORES,
            tiles,
            self.global_cb_size,
        )

    @staticmethod
    def make_ring_config(B: int, M: int, K: int, N: int, num_cores: int):
        """gather_in0 ring matmul program config.

        num_cores must divide BOTH K_tiles and N_tiles -- see ring_feasibility. The
        grid describes the number of output blocks, not physical ring placement
        (that comes from the input/output shard specs), so it is built to contain
        exactly num_cores.
        """
        M *= B  # fuse_batch=True
        n_tiles, k_tiles = N // TILE, K // TILE

        assert k_tiles % num_cores == 0, f"K_tiles={k_tiles} not divisible by num_cores={num_cores}"
        assert n_tiles % num_cores == 0, f"N_tiles={n_tiles} not divisible by num_cores={num_cores}"

        in0_block_w = k_tiles // num_cores
        out_block_w = n_tiles // num_cores
        out_block_h = M // TILE

        sbw = min(8, out_block_w)
        while sbw > 0 and out_block_w % sbw != 0:
            sbw -= 1

        gx = min(8, num_cores)
        gy = num_cores // gx
        assert gx * gy == num_cores, f"num_cores={num_cores} does not factor into an 8-wide grid"

        # Contiguous 2x8 receiver ring needs no hop core.
        hop_core_range_set = ttnn.CoreRangeSet([])
        logger.info(
            "Flash ring config: K={} N={} M={} cores={} grid=({},{}) in0_block_w={} per_core_N={}",
            K,
            N,
            M,
            num_cores,
            gx,
            gy,
            in0_block_w,
            out_block_w,
        )
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
            num_global_cb_receivers=NUM_GLOBAL_CB_RECEIVERS,
        )

    @staticmethod
    def make_ring_mem_cfg(num_cores: int, M: int, shard_dim: int, ring_cores):
        """L1 WIDTH_SHARDED config pinned to the exact receiver cores, so the matmul
        CB core set is a subset of global_cb.all_cores()."""
        assert len(ring_cores) == num_cores, f"Expected {num_cores} ring cores, got {len(ring_cores)}"
        assert shard_dim % num_cores == 0, f"shard_dim={shard_dim} not divisible by num_cores={num_cores}"
        core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in ring_cores])
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range, [M, shard_dim // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def create_global_cb(self):
        if self.global_circular_buffer is None:
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
            )
            logger.info("Flash GlobalCB created, size={}", self.global_cb_size)

    def insert_tensor(self, tensor: ttnn.Tensor):
        """Register one DRAM-sharded weight, in per-layer order."""
        self.tensors.append(tensor)
        self.tensor_addrs.append(tensor.buffer_address())

    def get_input_tensors(self):
        """Build the uint32 address tensor (sharded onto the sender cores) and return
        [first-layer weights..., addr_tensor] for dram_prefetcher."""
        expected = self.n_tensors * self.n_layers
        assert len(self.tensor_addrs) == expected, f"Expected {expected} addresses, got {len(self.tensor_addrs)}"

        tensor_addrs = torch.tensor(self.tensor_addrs).repeat(len(self.dram_cores), 1)
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

    def ensure_ready(self):
        """One-time setup: load the SubDevice manager, create the GlobalCB, build the
        address tensor. Call BEFORE trace capture.

        Ordering matters: the address tensor shards onto SENDER cores, so it must be
        built while the stall group still spans both SubDevices. Building it after
        narrowing to worker-only trips the "Programs must be executed on a single
        sub-device" assertion.
        """
        if not self._sub_device_loaded:
            if self.mesh_sub_device_manager_id is None:
                prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
                worker_sub_device = ttnn.SubDevice([self.worker_core_range_set])
                self.mesh_sub_device_manager_id = self.mesh_device.create_sub_device_manager(
                    [prefetcher_sub_device, worker_sub_device], 0
                )
            self.mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id)
            self.mesh_device.set_sub_device_stall_group(
                [self.prefetcher_sub_device_id, self.worker_sub_device_id],
            )
            self._sub_device_loaded = True
            logger.info("Flash prefetcher: SubDevice manager created+loaded")

        self.create_global_cb()
        if self._tt_tensors is None:
            self._tt_tensors = self.get_input_tensors()
        # Narrow to worker-only so ordinary decode ops dispatch without hitting the
        # single-sub-device assertion.
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def compile_prefetch(self):
        """Pre-compile dram_prefetcher OUTSIDE trace capture so the traced issue hits
        the program cache and writes no runtime args.

        Do NOT synchronize: the prefetcher fills the GlobalCB and then stalls with no
        consumer, so synchronize_device would hang. Just free the garbage output.
        """
        assert self._tt_tensors is not None, "call ensure_ready() before compile_prefetch()"
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device_id, self.worker_sub_device_id])
        garbage = ttnn.dram_prefetcher(
            self._tt_tensors, num_layers=self.n_layers, global_cb=self.global_circular_buffer
        )
        ttnn.deallocate(garbage)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        logger.info("Flash prefetcher: compile_prefetch() done — program cached")

    def start_prefetch(self):
        """Issue the cached dram_prefetcher INSIDE trace capture."""
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
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False
        self._tt_tensors = None
