# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM-core prefetcher class for tt-transformers models on Blackhole.

Drop-in alternative to ``Prefetcher`` (``prefetcher.py``) that pushes weights from
programmable DRAM cores (DRISC kernels) into the receiver ring instead of from
worker cores. Public surface mirrors ``Prefetcher`` so MLP/attention/model code
swaps without changes.

Lifecycle differs internally: a single ``ttnn.experimental.start_dram_core_prefetcher``
runs for the model's life, with ``queue_dram_core_prefetcher_request`` called once
per ``run()`` (per decode ``forward()``). ``stop()`` is a no-op between forwards;
the real shutdown happens in ``teardown()`` or when the MeshDevice closes.
"""

import os
from typing import Callable, List, Optional, Union

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode

# Map ttnn weight dtypes to tile byte sizes for GCB sizing.
_TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int) -> ttnn.CoreRangeSet:
    """Receivers for bank ``bank_idx``: ring positions [b*r, (b+1)*r) on a ``ring_cols × *`` worker rectangle.

    Mirrors the helper in ``tests/.../test_prefetcher_BH_dram_core_large.py``.
    """
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def is_dram_core_prefetcher_supported(
    model_name: str, num_devices: int, num_dram_banks: int, recv_per_bank: int
) -> bool:
    """Strict divisibility + GCB-size check for the DRAM-core prefetcher path.

    The DRAM-core path has the same uniform-receivers requirement as the worker path,
    plus an actual ``n_per_device_tiles % ring_size == 0`` check that the worker-side
    ``is_prefetcher_supported`` skips. Also enforces ``num_blocks * in1_block_size <=
    L1_PER_BANK_BUDGET`` (the GCB allocation per receiver). Returns True iff every weight
    class (FF1/FF3, FF2, QKV, WO) divides cleanly across
    ``ring_size = num_dram_banks * recv_per_bank`` AND its per-receiver GCB footprint
    fits worker L1.
    """
    # Import here to avoid circular import (this module is imported from prefetcher.py).
    from models.tt_transformers.tt.prefetcher import VERIFIED_MODEL_CONFIGS

    verified = next((m for m in VERIFIED_MODEL_CONFIGS if m in model_name), None)
    if not is_blackhole() or verified is None:
        return False
    cfg = VERIFIED_MODEL_CONFIGS[verified]
    dim, hidden_dim, n_heads, n_kv_heads = cfg["dim"], cfg["hidden_dim"], cfg["n_heads"], cfg["n_kv_heads"]
    if n_kv_heads % num_devices != 0:
        return False
    ring_size = num_dram_banks * recv_per_bank
    TILE = 32
    # Per-device N for each weight class.
    n_hidden_per_dev = hidden_dim // num_devices  # FF1/FF3 out, FF2 in
    n_dim_per_dev = dim  # FF2 out, WO out
    head_dim = dim // n_heads
    qkv_size_per_dev = (n_heads + 2 * n_kv_heads) * head_dim // num_devices
    wo_in_per_dev = n_heads * head_dim // num_devices

    def _tiles_divide(n: int) -> bool:
        return (n % TILE == 0) and ((n // TILE) % ring_size == 0)

    # K of each op must divide ring too (gather_in0 K split).
    if not _tiles_divide(dim):  # FF1/FF3 K, attn input K
        return False
    if not _tiles_divide(hidden_dim // num_devices):  # FF2 K
        return False
    if not _tiles_divide(n_hidden_per_dev):  # FF1/FF3 N
        return False
    if not _tiles_divide(n_dim_per_dev):  # FF2 N, WO N
        return False
    if not _tiles_divide(qkv_size_per_dev):
        return False
    if not _tiles_divide(wo_in_per_dev):
        return False

    # Per-receiver GCB footprint must fit worker L1. The factory allocates
    # ``num_blocks * in1_block_size = ring_size * (K_per_shard_tiles * N_per_recv_tiles)
    # * tile_bytes`` per receiver, where K_per_shard_tiles = K_tiles / ring_size and
    # N_per_recv_tiles = N_per_device_tiles / ring_size. So the per-receiver footprint
    # simplifies to ``K_tiles * N_per_device_tiles * tile_bytes / ring_size`` — at small
    # rings the K_tiles factor dominates and pushes past L1.
    # The largest is bfloat8_b at 1088 B/tile; budget ~1.3 MB per bank to leave headroom
    # for other L1 allocations (matches the worker-path budget in is_prefetcher_supported).
    BYTES_PER_TILE_BFP8 = 1088
    L1_BUDGET = 1300000
    op_shapes = [
        (dim, n_hidden_per_dev),  # FF1/FF3: K=dim, N=hidden_dim/num_devices
        (hidden_dim // num_devices, n_dim_per_dev),  # FF2: K=hidden_dim/num_devices, N=dim
        (dim, qkv_size_per_dev),  # attn QKV: K=dim, N=qkv_size/num_devices
        (n_heads * head_dim // num_devices, n_dim_per_dev),  # attn WO: K=n_heads*head_dim/num_devices, N=dim
    ]
    for k_dim, n_dim in op_shapes:
        k_tiles = k_dim // TILE
        n_tiles_per_recv = (n_dim // TILE) // ring_size  # already int-divisible by check above
        # Same arithmetic as DramCorePrefetcher._build_global_cb:
        # num_blocks=ring_size, in1_block=(k_tiles/ring) * n_tiles_per_recv * tile_bytes.
        # Per-receiver footprint = num_blocks * in1_block.
        per_recv_bytes = ring_size * (k_tiles // ring_size) * n_tiles_per_recv * BYTES_PER_TILE_BFP8
        if per_recv_bytes > L1_BUDGET:
            return False
    return True


class DramCorePrefetcher(LightweightModule):
    """DRAM-core (DRISC) prefetcher with the same public surface as ``Prefetcher``.

    Receiver layout is a ``num_dram_banks × recv_per_bank`` rectangle anchored at ``(0,0)``
    on the worker grid. Bank ``b``'s receivers occupy ring positions ``[b*r, (b+1)*r)`` in
    row-major. DRAM sender cores live on a separate programmable core type so they don't
    occupy worker grid coordinates.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_tensors: int,
        num_layers: int,
        num_receiver_cores: Optional[int] = None,
    ):
        self.mesh_device: ttnn.MeshDevice = mesh_device
        self.num_tensors: int = num_tensors
        self.num_layers: int = num_layers
        self.num_senders: int = mesh_device.dram_grid_size().x
        self.model_name: str = os.getenv("HF_MODEL", "")
        assert self.model_name != "", "HF_MODEL is not set. DRAM Prefetcher must be run with a model."

        # Pick recv_per_bank. Auto-mode walks 1,2,4,8 and takes the largest supported.
        candidates: List[int] = [num_receiver_cores] if num_receiver_cores is not None else [8, 4, 2, 1]
        picked = None
        for rpb in candidates:
            if is_dram_core_prefetcher_supported(
                self.model_name, self.mesh_device.get_num_devices(), self.num_senders, rpb
            ):
                picked = rpb
                break
        assert picked is not None, (
            f"No supported recv_per_bank for {self.model_name} on {self.mesh_device.get_num_devices()} devices with "
            f"{self.num_senders} DRAM banks. Tried {candidates}."
        )
        self.num_receiver_cores: int = picked
        self.ring_size: int = self.num_senders * self.num_receiver_cores

        # Receiver rectangle: cols 0..ring_cols-1, rows 0..ring_rows-1.
        self.ring_cols: int = self.num_senders
        self.ring_rows: int = self.num_receiver_cores
        self._receiver_rect = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.ring_cols - 1, self.ring_rows - 1))}
        )

        # Consumer sub-device covers the entire worker grid. DRAM senders are on a
        # separate programmable core type and are not in this set. The matmul receivers
        # (and any other compute op) must be inside this sub-device, so we do NOT
        # subtract the receiver rectangle — only the worker-side Prefetcher does that to
        # carve out *worker-grid* sender columns (0/7), which we don't have here.
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.all_worker_cores_range_set: ttnn.CoreRangeSet = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )

        # Per-bank receivers as a list aligned with bank indices.
        self._bank_to_receivers: List = [
            (b, _bank_receivers_row_major(b, self.num_receiver_cores, self.ring_cols)) for b in range(self.num_senders)
        ]

        # State.
        self.global_cb: Optional[ttnn.GlobalCircularBuffer] = None
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.prefetcher_sub_device = None  # filled in init()
        self.callbacks: List[Callable[[], None]] = []
        self.prefetched_tensors: List[ttnn.Tensor] = []
        self.prefetched_program_configs: List = []
        self.mode: Mode = Mode.PREFILL
        self.init_decode_done: bool = False
        self.init_prefill_done: bool = False
        self.prefetch_done: bool = False
        self._started: bool = False  # Tracks lazy StartDramCorePrefetcher
        self._stopped: bool = False  # Set by teardown(); blocks re-entry

        logger.info(
            f"[DramCorePrefetcher] model={self.model_name} banks={self.num_senders} recv/bank={self.num_receiver_cores} ring={self.ring_size}"
        )

    # ---- Public surface compatible with Prefetcher ----

    def register_callback(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def insert_tensor(self, tensor: ttnn.Tensor, program_config=None) -> None:
        """Register a tensor to be prefetched. ``program_config`` is required for GCB sizing."""
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot insert tensors"
        assert program_config is not None, (
            "DramCorePrefetcher.insert_tensor requires program_config (the 1D mcast gather_in0 matmul "
            "program config that will consume this weight). Threaded through from MLP/Attention."
        )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for DRAM-core prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(f"Tensor volume ({tensor.volume()}) must be divisible by ring_size ({self.ring_size}).")
        self.prefetched_tensors.append(tensor)
        self.prefetched_program_configs.append(program_config)
        logger.info(
            f"[DramCorePrefetcher] Inserted tensor of shape {tensor.shape} ({len(self.prefetched_tensors)}/{self.num_tensors})"
        )

    def init(self, mode: Mode = Mode.DECODE) -> None:
        if mode == Mode.DECODE and self.init_decode_done or mode == Mode.PREFILL and self.init_prefill_done:
            return
        self.mode = mode
        # Lazy import to avoid hard dep order at module load.
        from models.tt_transformers.tt.prefetcher import PrefetcherSubDevice

        # Single consumer sub-device — DRAM senders live on a different programmable core
        # type, not the worker sub-device.
        self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
        self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
        self.prefetcher_sub_device.init_sub_device_manager()
        self.worker_sub_device_id = self.prefetcher_sub_device.sub_devices_id[-1]

        logger.info(
            f"[DramCorePrefetcher.init] mode={mode} ring={self.ring_size} "
            f"receivers={self._receiver_rect.bounding_box()} "
            f"workers={self.all_worker_cores_range_set.num_cores()} cores"
        )
        if mode == Mode.DECODE:
            self.init_decode_done = True
        else:
            self.init_prefill_done = True

    def prefetch(self) -> None:
        if self.mode != Mode.DECODE:
            return
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode."
        assert self.callbacks, "No callbacks registered for the prefetcher."
        if self.prefetch_done:
            return
        for cb in self.callbacks:
            cb()
        self.prefetch_done = True

    def run(self) -> None:
        """Start DRISC daemon (lazy, first call) and queue one request per call."""
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode."
        assert not self._stopped, "DramCorePrefetcher has been torn down; cannot run again."
        assert (
            len(self.prefetched_tensors) >= self.num_tensors
        ), f"Insufficient inserted tensors: {len(self.prefetched_tensors)} < {self.num_tensors}"

        if self.global_cb is None:
            self._build_global_cb()
        if not self._started:
            ttnn.experimental.start_dram_core_prefetcher(self.mesh_device, enable_performance_mode=True)
            self._started = True

        # One request per forward(): the matmuls in one decode pass consume
        # num_layers * num_tensors blocks; per-GCB state is preserved so successive
        # Queue calls resume where the previous one stopped.
        ttnn.experimental.queue_dram_core_prefetcher_request(
            self.mesh_device,
            self.prefetched_tensors[: self.num_tensors],
            num_layers=self.num_layers,
            global_cb=self.global_cb,
        )
        # Stall worker consumer ops on the worker sub-device; the DRAM sender lives outside.
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def stop(self) -> None:
        """Per-forward stop. NO-OP for the DRAM-core path — DRISC daemon stays up.

        Real shutdown happens in ``teardown()`` or when the MeshDevice closes.
        """
        return

    def teardown(self) -> None:
        if self._started and not self._stopped:
            ttnn.experimental.stop_dram_core_prefetcher(self.mesh_device)
            self._stopped = True

    def __del__(self):
        # Best-effort cleanup. MeshDevice close has its own graceful fallback if we miss it.
        try:
            self.teardown()
        except Exception:
            pass

    # ---- Helpers expected by model_config.py and the model code ----

    def to_core_range_set(
        self, cores: List, return_list: bool = False
    ) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
        """Convert cores (CoreCoord/CoreRange/CoreRangeSet) to CoreRangeSet(s).

        Matches ``Prefetcher.to_core_range_set``.
        """
        assert cores, "No cores provided"

        def to_ranges(c):
            if isinstance(c, ttnn.CoreRangeSet):
                return c.ranges()
            if isinstance(c, ttnn.CoreRange):
                return [c]
            if isinstance(c, ttnn.CoreCoord):
                return [ttnn.CoreRange(c, c)]
            raise ValueError(f"Unsupported core type: {type(c)}")

        if return_list:
            return [ttnn.CoreRangeSet(to_ranges(c)) for c in cores]
        return ttnn.CoreRangeSet([r for c in cores for r in to_ranges(c)])

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRangeSet]:
        """One CoreRangeSet per sender. With this layout, every sender is 'active'.

        ``sender_active``/``receiver_active`` are accepted for parity with the worker-core
        ``Prefetcher.receiver_cores`` signature but are ignored — the DRAM-core path has no
        inactive subset (every DRAM bank is a sender, every receiver in the rectangle is
        active).
        """
        del sender_active, receiver_active
        return [crs for (_b, crs) in self._bank_to_receivers]

    def dynamic_worker_core_grid(self, num_cores: int) -> ttnn.CoreRangeSet:
        """Allocate ``num_cores`` worker cores from rows below the receiver rectangle.

        Rectangle occupies rows 0..ring_rows-1; workers spill into rows ring_rows.. in
        row-major order, wrapping to a new row when the chip's column count is exhausted.
        """
        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x, grid_y = grid.x, grid.y
        rows_left = grid_y - self.ring_rows
        cores_per_row = grid_x
        assert num_cores <= rows_left * cores_per_row, (
            f"dynamic_worker_core_grid: requested {num_cores} cores but only "
            f"{rows_left * cores_per_row} cores are available below the receiver rectangle "
            f"(grid={grid_x}x{grid_y}, rect_rows={self.ring_rows})."
        )
        full_rows = num_cores // cores_per_row
        tail = num_cores % cores_per_row
        ranges = []
        row0 = self.ring_rows
        if full_rows > 0:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, row0), ttnn.CoreCoord(grid_x - 1, row0 + full_rows - 1)))
        if tail > 0:
            tail_row = row0 + full_rows
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, tail_row), ttnn.CoreCoord(tail - 1, tail_row)))
        return ttnn.CoreRangeSet(ranges)

    # ---- Private ----

    def _build_global_cb(self) -> None:
        """Construct the DRAM-sender GlobalCircularBuffer sized for all queued (weight, pc) pairs."""
        assert len(self.prefetched_tensors) == len(self.prefetched_program_configs)
        assert len(self.prefetched_tensors) >= self.num_tensors

        max_in1_block_size = 0
        for tensor, pc in zip(
            self.prefetched_tensors[: self.num_tensors], self.prefetched_program_configs[: self.num_tensors]
        ):
            tile_bytes = _TILE_BYTES[tensor.dtype]
            # gather_in0 matmul uses actual_in0_block_w = weight_K_tiles / ring_size, NOT pc.in0_block_w.
            weight_K = tensor.shape[-2]
            weight_K_tiles = weight_K // ttnn.TILE_SIZE
            assert weight_K_tiles % self.ring_size == 0, (
                f"Weight K_tiles {weight_K_tiles} must be divisible by ring_size {self.ring_size}; "
                "this should have been caught by is_dram_core_prefetcher_supported."
            )
            actual_in0_block_w = weight_K_tiles // self.ring_size
            in1_block_size = actual_in0_block_w * pc.per_core_N * tile_bytes
            max_in1_block_size = max(max_in1_block_size, in1_block_size)
            logger.info(
                f"[DramCorePrefetcher] tensor K={weight_K} N_per_core={pc.per_core_N} tile_bytes={tile_bytes} "
                f"in1_block_size={in1_block_size}"
            )

        # Minimum buffer = num_blocks * max_in1_block_size = ring_size * max_in1_block_size (one layer's pages).
        num_blocks = self.ring_size
        gcb_size = num_blocks * max_in1_block_size
        logger.info(
            f"[DramCorePrefetcher] Creating GCB: ring={self.ring_size} max_in1={max_in1_block_size} size={gcb_size}"
        )
        self.global_cb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            self.mesh_device,
            self.prefetched_program_configs[: self.num_tensors],
            self.prefetched_tensors[: self.num_tensors],
            bank_to_receivers=self._bank_to_receivers,
            size=gcb_size,
        )
