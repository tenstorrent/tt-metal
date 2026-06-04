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


def _ring_pos_coord(ring_pos: int, ring_cols: int) -> ttnn.CoreCoord:
    """Map a ring position to its receiver-rectangle ``(col, row)`` in row-major order."""
    return ttnn.CoreCoord(ring_pos % ring_cols, ring_pos // ring_cols)


def _bank_receivers_strided(
    bank_idx: int, recv_per_bank: int, num_dram_banks: int, ring_cols: int
) -> ttnn.CoreRangeSet:
    """Receiver-contiguous (strided) receivers for bank ``bank_idx``.

    Bank ``b`` feeds ring positions ``[b, b + num_dram_banks, b + 2*num_dram_banks, ...]``.
    Pairs with the round-robin ``NdShardSpec`` weight layout (shard ``s`` lands on bank
    ``s % num_dram_banks`` slab ``s // num_dram_banks``) so that ring position ``r`` receives
    shard ``r`` without any host permutation. Mirrors ``_bank_receivers_strided`` in
    ``tests/.../test_prefetcher_BH_dram_core_large.py``. With ``ring_cols == num_dram_banks``
    this is simply column ``b`` of the rectangle.
    """
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx + s * num_dram_banks
        cores.append(ttnn.CoreRange(_ring_pos_coord(ring_pos, ring_cols), _ring_pos_coord(ring_pos, ring_cols)))
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
    from models.tt_transformers.tt.prefetcher import BYTES_PER_TILE_BFP8, VERIFIED_MODEL_CONFIGS

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

    # Every prefetched weight is allocated with its real (unpadded) per-device dims — the model
    # does NOT pad qkv/hidden for the recv-contig path (mlp.py / attention.py pass the raw
    # hidden_dim//num_devices, qkv_size//num_devices, etc.). So each weight's K and N must
    # tile-divide the ring on its own; otherwise the per-receiver shard width isn't tile-aligned
    # and the 1D-ring matmul's num_blocks != num_cores. This is what forces a smaller ring for
    # larger models (e.g. 8B's qkv_size//device=1536 -> 48 tiles divides ring=16 but not ring=32,
    # so 8B picks recv_per_bank=2 / ring=16, matching the worker prefetcher's choice).
    #
    # Covers: dim (FF1/FF3 K, FF2/WO/QKV N or K), n_hidden_per_dev (FF1/FF3 N, FF2 K),
    # qkv_size_per_dev (QKV N), wo_in_per_dev (WO N for a single-row mesh; == dim//num_devices).
    for n in (dim, n_hidden_per_dev, qkv_size_per_dev, wo_in_per_dev):
        if not _tiles_divide(n):
            return False

    # Per-receiver GCB footprint must fit worker L1. The factory allocates
    # ``num_blocks * in1_block_size = ring_size * (K_per_shard_tiles * N_per_recv_tiles)
    # * tile_bytes`` per receiver, where K_per_shard_tiles = K_tiles / ring_size and
    # N_per_recv_tiles = N_per_device_tiles / ring_size. So the per-receiver footprint
    # simplifies to ``K_tiles * N_per_device_tiles * tile_bytes / ring_size`` — at small
    # rings the K_tiles factor dominates and pushes past L1.
    # The largest is bfloat8_b at 1088 B/tile (shared BYTES_PER_TILE_BFP8); budget ~1.3 MB per
    # bank to leave headroom for other L1 allocations (matches is_prefetcher_supported).
    L1_BUDGET = 1300000
    # NOTE: this gate assumes a single-row mesh (cluster_shape == (1, num_devices)), the only
    # topology this path is verified on. There, attention.py allocates WO as
    # K=dim//cluster_shape[0]=dim, N=dim//cluster_shape[1]=dim//num_devices — i.e. (dim,
    # wo_in_per_dev). The op_shapes WO entry below uses (wo_in_per_dev, dim); divisibility holds
    # because BOTH dim and wo_in_per_dev are checked above, and the L1 footprint is symmetric in
    # K and N (k_tiles * n_total_tiles / ring), so the order is immaterial for this check. A
    # multi-row mesh (cluster_shape[0] > 1) is not validated here; recv_contig_weight_mem_config's
    # `n % ring_size == 0` assert and _build_global_cb's K-divisibility assert catch a mismatch
    # cleanly at allocation time rather than corrupting silently.
    op_shapes = [
        (dim, n_hidden_per_dev),  # FF1/FF3: K=dim, N=hidden_dim/num_devices
        (n_hidden_per_dev, n_dim_per_dev),  # FF2: K=hidden_dim/num_devices, N=dim
        (dim, qkv_size_per_dev),  # attn QKV: K=dim, N=qkv_size/num_devices
        (wo_in_per_dev, n_dim_per_dev),  # attn WO: see NOTE above (K/N symmetric for this check)
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

        # Pick recv_per_bank. Auto-mode walks 8,4,2,1 (descending) and takes the largest supported.
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

        # Receiver rectangle is num_senders (cols) x num_receiver_cores (rows): ring position r
        # maps to (col=r%num_senders, row=r//num_senders), and bank b owns rectangle column b
        # (its strided ring positions b, b+num_senders, ...).
        #
        # Receiver cores in ring-position order. The gather_in0 matmul walks its core_grid in
        # this order, so ring core r computes output N-cols [r*per_core_N, (r+1)*per_core_N).
        # receiver_cores() returns this list (flattened) as the matmul grid.
        self._ring_cores: List[ttnn.CoreCoord] = [_ring_pos_coord(r, self.num_senders) for r in range(self.ring_size)]

        # Consumer sub-device covers the entire worker grid. DRAM senders are on a
        # separate programmable core type and are not in this set. The matmul receivers
        # (and any other compute op) must be inside this sub-device, so we do NOT
        # subtract the receiver rectangle — only the worker-side Prefetcher does that to
        # carve out *worker-grid* sender columns (0/7), which we don't have here.
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.all_worker_cores_range_set: ttnn.CoreRangeSet = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )

        # Per-bank receivers as a list aligned with bank indices. Strided (receiver-contiguous):
        # bank b feeds ring positions [b, b+num_senders, ...], matching the round-robin NdShardSpec
        # weight layout so shard r reaches ring position r. Used only to build the DRAM-sender GCB;
        # the matmul grid comes from receiver_cores() (ring order), which is intentionally decoupled.
        self._bank_to_receivers: List = [
            (b, _bank_receivers_strided(b, self.num_receiver_cores, self.num_senders, self.num_senders))
            for b in range(self.num_senders)
        ]

        # State.
        self.global_cb: Optional[ttnn.GlobalCircularBuffer] = None
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.prefetcher_sub_device = None  # filled in init()
        self.callbacks: List[Callable[[], None]] = []
        self.prefetched_tensors: List[ttnn.Tensor] = []
        self.prefetched_program_configs: List = []
        self._queue_payload: Optional[List] = None  # cached (weight, block_count) list, built once
        self.mode: Mode = Mode.PREFILL  # set by init(); read by lm_head.py
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
        assert hasattr(program_config, "per_core_N"), (
            "DramCorePrefetcher.insert_tensor needs a 1D-mcast matmul program config exposing per_core_N "
            f"(used to size the GCB); got {type(program_config).__name__}."
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
        # Called for BOTH modes via model.switch_mode(): prefill then decode. Actual prefetching is
        # decode-only (prefetch()/run() are gated to decode), but the consumer sub-device must exist
        # in both modes so worker ops have a valid sub-device to run on. Mirrors Prefetcher.init.
        if mode == Mode.DECODE and self.init_decode_done or mode == Mode.PREFILL and self.init_prefill_done:
            return
        self.mode = mode
        # Lazy import to avoid hard dep order at module load.
        from models.tt_transformers.tt.prefetcher import PrefetcherSubDevice

        # Single consumer sub-device covering the whole worker grid, in both modes — DRAM senders
        # live on a separate programmable core type, so there are no worker-grid sender columns to
        # carve out (unlike the worker-core Prefetcher).
        self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
        self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
        self.prefetcher_sub_device.init_sub_device_manager()
        self.worker_sub_device_id = self.prefetcher_sub_device.sub_devices_id[-1]

        logger.info(
            f"[DramCorePrefetcher.init] mode={mode} ring={self.ring_size} "
            f"receivers={self.num_senders}x{self.num_receiver_cores} "
            f"workers={self.all_worker_cores_range_set.num_cores()} cores"
        )
        self.init_decode_done = mode == Mode.DECODE
        self.init_prefill_done = mode == Mode.PREFILL

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
        # The model registers num_tensors weights per decoder layer, so prefetch() collects
        # num_tensors * num_layers DISTINCT tensors (in construction order). This mirrors the
        # worker Prefetcher contract (create_address_tensor asserts the same count).
        assert len(self.prefetched_tensors) == self.num_tensors * self.num_layers, (
            f"Expected {self.num_tensors} * {self.num_layers} = {self.num_tensors * self.num_layers} inserted "
            f"tensors (num_tensors per layer), got {len(self.prefetched_tensors)}."
        )

        if self.global_cb is None:
            self._build_global_cb()
        if not self._started:
            ttnn.experimental.start_dram_core_prefetcher(self.mesh_device)
            self._started = True

        # One request per forward(): queue every inserted weight once, in construction order, so
        # each decoder layer gets its OWN weights (NOT layer-0's replayed). The matmuls consume
        # them in the same order across the decode pass; per-GCB state is preserved so successive
        # Queue calls resume where the previous stopped. The queue API takes a flattened list of
        # (weight, block_count) pairs — block_count = ring_size K-blocks per tensor. The payload
        # list is invariant after prefetch(), so it's cached in _queue_payload.
        if self._queue_payload is None:
            self._queue_payload = [(t, self.ring_size) for t in self.prefetched_tensors]
        ttnn.experimental.queue_dram_core_prefetcher_request(
            self.mesh_device,
            self._queue_payload,
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
        """Receiver cores in ring-position order, one CoreRangeSet per ring position.

        The model flattens this (via ``to_core_range_set``) into the matmul ``core_grid``; the
        gather_in0 matmul walks that grid in order, so ring core r computes N-cols
        [r*per_core_N, ...) and must receive shard r. This ring order is intentionally
        decoupled from ``_bank_to_receivers`` (strided), which only feeds the DRAM-sender GCB.

        ``sender_active``/``receiver_active`` are accepted for parity with the worker-core
        ``Prefetcher.receiver_cores`` signature but are ignored — the DRAM-core path has no
        inactive subset (every DRAM bank is a sender, every receiver in the rectangle is active).
        """
        del sender_active, receiver_active
        return [ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]) for c in self._ring_cores]

    def dram_banks(self) -> List[ttnn.CoreCoord]:
        """DRAM bank cores (the DRISC senders), one per bank: ``(0,0)..(num_senders-1, 0)``.

        Matches ``Prefetcher.dram_banks`` (a bound method on the worker class). Used by lm_head.py
        to lay out the (non-prefetched) LM-head weight across the DRAM grid.
        """
        return [ttnn.CoreCoord(b, 0) for b in range(self.num_senders)]

    def weight_mem_config(
        self, k: int, n: int, default: ttnn.MemoryConfig, is_galaxy: bool = False
    ) -> ttnn.MemoryConfig:
        """Memory config for a prefetched (K, N) weight (uniform with ``Prefetcher.weight_mem_config``).

        Returns the receiver-contiguous DRAM layout this backend requires, except on galaxy/TG
        meshes (the recv-contig path is not supported there) where it falls back to ``default``.
        Lets MLP/attention call ``prefetcher.weight_mem_config(...)`` without branching on backend.
        """
        if is_galaxy:
            return default
        return self.recv_contig_weight_mem_config(k, n)

    def recv_contig_weight_mem_config(self, k: int, n: int) -> ttnn.MemoryConfig:
        """Receiver-contiguous DRAM memory config for a prefetched (K, N) weight.

        Allocates the weight as an NdShardSpec with ``num_shards = ring_size`` (over-subscribed
        relative to the ``num_senders`` DRAM banks) distributed round-robin, each shard
        ``(K, N // ring_size)``. Paired with the strided GCB topology, shard r (columns
        [r*n_per_recv, (r+1)*n_per_recv)) is delivered to ring position r — exactly the weight
        slice the gather_in0 matmul's ring core r consumes. Callers allocate prefetched weights
        with this config when the DRAM-core backend is active (mlp.py / attention.py).
        """
        assert n % self.ring_size == 0, f"N={n} must divide ring_size={self.ring_size} for receiver-contiguous layout"
        n_per_recv = n // self.ring_size
        dram_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_senders - 1, 0))}
        )
        return ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                ttnn.Shape([k, n_per_recv]),
                dram_core_range_set,
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        )

    def dynamic_worker_core_grid(self, num_cores: int) -> ttnn.CoreRangeSet:
        """Allocate ``num_cores`` worker cores as a SOLID rectangle below the receiver rectangle.

        The receiver rectangle occupies rows ``0..num_receiver_cores-1``; the returned grid sits in
        rows ``num_receiver_cores..``. It must be a filled rectangle (``num_cores() == bounding-box``)
        because consumers like the residual RMSNorm assert ``shard_spec.grid.num_cores() ==
        bbox_num_cores`` — a ragged row-major spill (one full row + a partial tail) fails that check.

        We pick the tallest rectangle ``width x height`` with ``width * height == num_cores`` that
        fits in ``grid_x`` columns and the available rows below the receiver rectangle, anchored at
        ``(0, num_receiver_cores)``. (The worker-core Prefetcher likewise returns a tall rectangle.)
        """
        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x, grid_y = grid.x, grid.y
        rows_left = grid_y - self.num_receiver_cores
        row0 = self.num_receiver_cores
        # Tallest rectangle: largest height that divides num_cores, fits the available rows, and
        # leaves a width that fits the column count.
        height = next(
            (h for h in range(min(rows_left, num_cores), 0, -1) if num_cores % h == 0 and num_cores // h <= grid_x),
            0,
        )
        assert height > 0, (
            f"dynamic_worker_core_grid: cannot fit {num_cores} cores into a rectangle within "
            f"{rows_left} rows x {grid_x} cols below the receiver rectangle "
            f"(grid={grid_x}x{grid_y}, rect_rows={self.num_receiver_cores})."
        )
        width = num_cores // height
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, row0), ttnn.CoreCoord(width - 1, row0 + height - 1))]
        )

    # ---- Private ----

    def _build_global_cb(self) -> None:
        """Construct the DRAM-sender GlobalCircularBuffer sized for all queued (weight, pc) pairs."""
        # Lazy import to avoid hard dep order at module load (see is_dram_core_prefetcher_supported).
        from models.tt_transformers.tt.prefetcher import TILE_BYTES

        assert len(self.prefetched_tensors) == len(self.prefetched_program_configs)
        assert len(self.prefetched_tensors) == self.num_tensors * self.num_layers

        # Size over every inserted weight (all layers): the GCB is a ring sized for one layer's
        # worth of in-flight pages (num_blocks = ring_size), so we take the max block across all.
        max_in1_block_size = 0
        for tensor, pc in zip(self.prefetched_tensors, self.prefetched_program_configs):
            tile_bytes = TILE_BYTES[tensor.dtype]
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
        # Receiver-contiguous weights are NdShardSpec (round-robin), which the matmul-aware
        # _for_matmul_1d factory rejects (it asserts a K-row-major single-wide-shard-per-bank
        # layout). Build the DRAM-sender GCB directly from the strided bank->receivers topology;
        # the underlying GCB object is identical.
        self.global_cb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(
            self.mesh_device,
            self._bank_to_receivers,
            gcb_size,
        )
