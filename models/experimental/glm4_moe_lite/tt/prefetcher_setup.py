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
import os

import torch
import ttnn
from loguru import logger

from models.experimental.glm4_moe_lite.tt.linear_helpers import worker_grid_x


def _sharded_norm_cores() -> int:
    """Decode sharded-RMSNorm core count. Read here rather than imported from
    decoder_layer_tt to keep this module free of a dependency on the decode path."""
    return int(os.environ.get("GLM4_MOE_LITE_SHARDED_NORM_CORES", "8").strip() or "8")


TILE = 32
# Receiver cores per sender in the GlobalCB contract, passed to the matmul as
# num_global_cb_receivers. 2 is the WH Galaxy value; Blackhole's wider grid can host 4,
# which halves the per-receiver GlobalCB payload -- see ring_cores_for/global_cb_tiles_for.
NUM_GLOBAL_CB_RECEIVERS = 2
# Senders in the ring, one per DRAM bank. 8 on both arches: it is Blackhole's actual bank
# count, and on WH (12 views) the ring was deliberately built on 8 because a 24-core ring
# divides neither of w_o's dimensions -- see the module docstring.
NUM_DRAM_BANKS = 8
# Bytes per bfloat8_b tile (1024 elements + per-tile exponent metadata).
BF8_TILE_BYTES = 1088


def get_glm_core_ranges(mesh_device, num_global_cb_receivers: int = NUM_GLOBAL_CB_RECEIVERS):
    """Core ranges for the prefetcher, derived from the device grid.

    The layout rule, which is what actually matters and is arch-independent:

      - one sender per DRAM bank, in a single column *outside* the worker rectangle;
      - `num_global_cb_receivers` receivers per sender, in contiguous rows immediately to
        the left of the senders, so the matmul's remote-CB core set exactly equals the
        GlobalCB receiver set and no dedicated hop core is needed;
      - the worker SubDevice is the remaining leftmost columns, a solid rectangle anchored
        at origin (0,0) -- so matmul grids, which are origin-anchored, stay inside it
        without needing an explicit sub_device_id.

    On WH Galaxy (8x9 grid, 8 DRAM banks, 2 receivers) this reproduces the original
    layout exactly: senders column 6, receivers columns 4-5, workers columns 0-5.
    On Blackhole (12x10, 8 banks) it gives senders column 10, workers columns 0-9, and
    room for 4 receivers per sender -- which is what makes a 32-core ring possible.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    logger.info("Flash prefetcher: device grid {}x{}", grid_x, grid_y)

    # One sender per DRAM bank. WH Galaxy exposes 12 DRAM views but the ring is built on 8
    # (see the module docstring for why 8x2=16 and not 12x24); Blackhole has exactly 8, so
    # taking the device's bank count keeps both correct.
    num_banks = min(NUM_DRAM_BANKS, int(mesh_device.dram_grid_size().x))
    dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(num_banks)]

    n_workers_x = worker_grid_x(grid_x)
    sender_x = n_workers_x
    if sender_x >= grid_x:
        raise ValueError(f"device grid {grid_x}x{grid_y} is too narrow for a sender column")

    # Senders occupy one row per bank, so the grid must be at least that tall.
    if num_banks > grid_y:
        raise ValueError(f"{num_banks} DRAM banks need {num_banks} sender rows, grid has {grid_y}")
    all_sender_cores = [ttnn.CoreCoord(sender_x, y) for y in range(num_banks)]

    # Receivers sit in the columns immediately left of the senders, inside the worker
    # rectangle. Bank-major then row-major, matching gather_in0's ring walk, with each
    # sender's receivers contiguous along x so they form a single CoreRange.
    recv_x0 = sender_x - num_global_cb_receivers
    if recv_x0 < 0:
        raise ValueError(
            f"{num_global_cb_receivers} receivers per sender do not fit left of column {sender_x} "
            f"on a {grid_x}-wide grid"
        )
    all_receiver_pairs = [(x, y) for y in range(num_banks) for x in range(recv_x0, sender_x)]

    sender_receiver_mapping = []
    for i, sender in enumerate(all_sender_cores):
        group = all_receiver_pairs[i * num_global_cb_receivers : (i + 1) * num_global_cb_receivers]
        recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*group[0]), ttnn.CoreCoord(*group[-1]))])
        sender_receiver_mapping.append((sender, recv_crs))

    sender_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_sender_cores])
    worker_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_workers_x - 1, grid_y - 1))]
    )

    logger.info(
        "Flash prefetcher layout: {} senders (col {}), {} receivers (cols {}-{}), " "worker cols 0-{} rows 0-{}",
        len(all_sender_cores),
        sender_x,
        len(all_receiver_pairs),
        recv_x0,
        sender_x - 1,
        n_workers_x - 1,
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


def ring_cores_for(mesh_device, K: int, N: int) -> tuple[int, int]:
    """Largest legal (ring_cores, receivers_per_sender) for a (K, N) weight on this device.

    The ring width is the lever that decides whether the prefetcher fits in L1 at all,
    because the per-receiver GlobalCB payload is `K_tiles * (N_tiles / ring_cores)` -- so
    doubling the ring halves it. For Flash's w_o (5120x2048 = 160x64 tiles) with L1 at
    1,572,864 B on Blackhole and SDPA's circular buffers needing 1,033,568 B:

        ring=16 -> 640 tiles = 696,320 B  ->  1,729,888 B total, over on both arches
        ring=32 -> 320 tiles = 348,160 B  ->  1,381,728 B total, fits on BH with ~191 KB spare

    WH Galaxy cannot reach ring=32: it has 8 sender columns' worth of grid but only room
    for 2 receivers per sender beside them. Blackhole's wider grid fits 4, giving 8x4=32.
    gcd(160, 64) = 32 so that is also the maximum legal ring for w_o -- a ring that does
    not divide both dimensions deadlocks on device rather than raising.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    num_banks = min(NUM_DRAM_BANKS, int(mesh_device.dram_grid_size().x))
    # Receivers must fit in the worker columns to the left of the sender column.
    max_receivers = max(1, worker_grid_x(int(grid.x)) - 1)
    best = (0, 0)
    for receivers in range(max_receivers, 0, -1):
        ring = num_banks * receivers
        if ring in ring_feasibility(K, N, max_cores=ring):
            best = (ring, receivers)
            break
    if best[0] == 0:
        raise ValueError(f"no legal ring for K={K} N={N} on a {grid.x}x{grid.y} grid with {num_banks} banks")
    return best


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
    # widths over one GlobalCB contract; w_q_b is a later increment.
    OPROJ_K = 5120
    OPROJ_N = 2048
    # WH Galaxy ring width, kept for reference. The live value is `self.ring_cores`, sized
    # from the device by ring_cores_for() -- 16 on WH, 32 on Blackhole, and that doubling
    # is what brings the GlobalCB under the L1 budget.
    RING_CORES = 16

    def __init__(self, mesh_device, n_tensors_per_layer: int, n_layers: int, global_cb_tiles: int | None = None):
        self.mesh_device = mesh_device
        self.n_tensors = n_tensors_per_layer
        self.n_layers = n_layers

        self.ring_cores, self.receivers_per_sender = ring_cores_for(mesh_device, self.OPROJ_K, self.OPROJ_N)
        (
            self.sender_cores,
            self.dram_cores,
            self.sender_core_range_set,
            self.receiver_cores,
            self.worker_core_range_set,
            self.sender_receiver_mapping,
        ) = get_glm_core_ranges(mesh_device, num_global_cb_receivers=self.receivers_per_sender)

        assert (
            len(self.receiver_cores) == self.ring_cores
        ), f"ring size {self.ring_cores} must equal receiver count {len(self.receiver_cores)}"

        feasible = ring_feasibility(self.OPROJ_K, self.OPROJ_N, max_cores=self.ring_cores)
        assert self.ring_cores in feasible, (
            f"o_proj K={self.OPROJ_K} N={self.OPROJ_N} cannot use a {self.ring_cores}-core ring "
            f"(feasible: {feasible}). A non-dividing ring deadlocks on device."
        )

        tiles = global_cb_tiles or global_cb_tiles_for(self.OPROJ_K, self.OPROJ_N, self.ring_cores)
        self.global_cb_size = tiles * BF8_TILE_BYTES
        self.global_circular_buffer = None

        self.oproj_ring_cores = list(self.receiver_cores)
        self.oproj_program_config = self.make_ring_config(
            B=1,
            M=TILE,
            K=self.OPROJ_K,
            N=self.OPROJ_N,
            num_cores=self.ring_cores,
            num_receivers=self.receivers_per_sender,
        )
        self.oproj_input_mem_cfg = self.make_ring_mem_cfg(
            num_cores=self.ring_cores, M=TILE, shard_dim=self.OPROJ_K, ring_cores=self.oproj_ring_cores
        )
        self.oproj_output_mem_cfg = self.make_ring_mem_cfg(
            num_cores=self.ring_cores, M=TILE, shard_dim=self.OPROJ_N, ring_cores=self.oproj_ring_cores
        )

        # Worker grids for re-gridding decode ops once the SubDevice is active.
        self.worker_scg = self.worker_core_range_set
        # Sharded RMSNorm grid, as (gx, gy). The norm's default layout is a 1 x num_cores ROW
        # at y=0, which on WH spans x=0..7 and lands on the sender columns; it must instead
        # be a rectangle that fits inside the worker columns. Pick the widest rectangle no
        # wider than the worker region, so the core count -- and the parallelism it buys --
        # is preserved on any grid.
        self.norm_core_grid = self._norm_rect(
            _sharded_norm_cores(), worker_grid_x(int(mesh_device.compute_with_storage_grid_size().x))
        )

        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False
        self._tt_tensors = None
        self.tensors = []
        self.tensor_addrs = []

        logger.info(
            "Glm4MoeLitePrefetcherSetup: n_tensors={} n_layers={} ring={} ({} recv/sender) "
            "global_cb={} tiles ({} B) norm_grid={}",
            n_tensors_per_layer,
            n_layers,
            self.ring_cores,
            self.receivers_per_sender,
            tiles,
            self.global_cb_size,
            self.norm_core_grid,
        )

    @staticmethod
    def _norm_rect(num_cores: int, max_x: int) -> tuple[int, int]:
        """Widest (gx, gy) rectangle with gx*gy == num_cores and gx <= max_x.

        Reproduces the WH choice of (4, 2) for 8 cores in a 6-wide worker region.
        """
        for gx in range(min(num_cores, max_x), 0, -1):
            if num_cores % gx == 0:
                return (gx, num_cores // gx)
        return (1, num_cores)

    @staticmethod
    def make_ring_config(B: int, M: int, K: int, N: int, num_cores: int, num_receivers: int | None = None):
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

        # A contiguous receiver block needs no hop core.
        hop_core_range_set = ttnn.CoreRangeSet([])
        receivers = NUM_GLOBAL_CB_RECEIVERS if num_receivers is None else int(num_receivers)
        logger.info(
            "Flash ring config: K={} N={} M={} cores={} recv/sender={} grid=({},{}) " "in0_block_w={} per_core_N={}",
            K,
            N,
            M,
            num_cores,
            receivers,
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
            num_global_cb_receivers=receivers,
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

        ORDERING TRAP -- this is only safe when the NEXT thing to execute contains
        consumers of the GlobalCB. dram_prefetcher stalls until something drains it, so

            compile_prefetch(); start_prefetch()      # <-- DEADLOCKS

        with no consuming matmul in between. Verified the hard way: 2935% CPU spin and
        a Galaxy reset. The traced model path is safe because start_prefetch() is issued
        inside trace capture, which records rather than executes, and the compile-time
        stall drains when the traced consumers replay. Any non-traced use (unit tests,
        bring-up scripts) must skip compile_prefetch entirely and issue exactly one
        start_prefetch immediately followed by its matmuls.
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
        """Restore full-grid dispatch by unloading and removing the SubDevice manager.

        Order matters: a loaded manager cannot be removed. Skipping
        clear_loaded_sub_device_manager() makes remove fail with
        "Cannot remove active sub device manager", which -- if swallowed -- silently
        leaves every subsequent op confined to worker columns 0-5. That is a
        particularly nasty failure for this model, because prefill needs the full grid.

        Exceptions are logged rather than dropped, for the same reason: this ran
        "successfully" for a while purely because the failure was being hidden.
        """
        try:
            self.mesh_device.reset_sub_device_stall_group()
        except Exception as e:  # pragma: no cover - device state dependent
            logger.warning("Flash prefetcher: reset_sub_device_stall_group failed: {}", e)
        if self.mesh_sub_device_manager_id is not None:
            try:
                self.mesh_device.clear_loaded_sub_device_manager()
            except Exception as e:  # pragma: no cover
                logger.warning("Flash prefetcher: clear_loaded_sub_device_manager failed: {}", e)
            try:
                self.mesh_device.remove_sub_device_manager(self.mesh_sub_device_manager_id)
            except Exception as e:  # pragma: no cover
                logger.warning("Flash prefetcher: remove_sub_device_manager failed: {}", e)
        self.mesh_sub_device_manager_id = None
        self._sub_device_loaded = False
        self._tt_tensors = None
