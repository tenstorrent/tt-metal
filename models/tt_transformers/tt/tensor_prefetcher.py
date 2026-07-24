# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM-core tensor prefetcher for tt-transformers models on Blackhole.

Weights are registered while the model is constructed so one shared GCB can be
sized for every decode matmul. ``init(Mode.DECODE)`` builds that GCB and starts
the long-running DRISC daemon. Each consuming matmul then uses
``prefetch_and_linear`` to queue exactly its own weight immediately before the
linear op; trace capture records those requests alongside their consumers.
"""

import os
from typing import List, Optional, Union

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.recv_contig_layout import bank_receivers_contiguous as _bank_receivers_contiguous
from models.tt_transformers.tt.recv_contig_layout import recv_contig_mem_config
from models.tt_transformers.tt.recv_contig_layout import ring_pos_coord as _ring_pos_coord

STREAMING_GCB_MAX_SIZE = 788032


def gcb_block_bytes(k_tiles: int, per_core_n_tiles: int, ring_size: int, tile_bytes: int) -> int:
    """Per-receiver GCB block footprint (bytes) for one gather_in0 matmul weight.

    The matmul splits ``in0_block_w`` to ``k_tiles / ring_size``; each receiver holds one
    ``(k_tiles/ring_size, per_core_N)`` tile block. The full GCB ring holds ``ring_size`` such
    blocks. Single source for the arithmetic shared by the support gate
    (:func:`is_tensor_prefetcher_config_supported`) and the builder
    (:meth:`TensorPrefetcher._build_global_cb`), so the pre-check and the allocation can't drift.
    """
    return (k_tiles // ring_size) * per_core_n_tiles * tile_bytes


def is_tensor_prefetcher_config_supported(
    model_name: str, num_devices: int, num_dram_banks: int, recv_per_bank: int
) -> bool:
    """Strict divisibility + streaming-GCB-size check for the Tensor Prefetcher path.

    The DRAM-core path has the same uniform-receivers requirement as the worker path,
    plus an actual ``n_per_device_tiles % ring_size == 0`` check that the worker-side
    ``is_prefetcher_supported`` skips. Because the GCB always streams (``stream_in1``), it holds
    only a partial ring — as many whole receiver blocks as fit in ``STREAMING_GCB_MAX_SIZE``,
    and ``_build_global_cb`` requires at least two. So the size gate is that a single receiver
    block is small enough for two to fit in that budget, NOT that the whole ring fits L1.
    Returns True iff every weight class (FF1/FF3, FF2, QKV, WO) divides cleanly across
    ``ring_size = num_dram_banks * recv_per_bank`` AND clears that streaming budget.
    """
    # Import here to avoid circular import (this module is imported from prefetcher.py).
    from models.tt_transformers.tt.prefetcher import BYTES_PER_TILE_BFP8, resolve_tensor_prefetcher_model_cfg

    cfg = resolve_tensor_prefetcher_model_cfg(model_name)
    if cfg is None:
        return False
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

    # Streaming-GCB size gate. The factory streams weights (``stream_in1`` is always enabled), so
    # per receiver it keeps ``num_blocks * in1_block_size`` where num_blocks = min(ring_size,
    # STREAMING_GCB_MAX_SIZE // in1_block_size) and one block is
    # ``(K_tiles/ring_size) * N_per_recv_tiles * tile_bytes`` (see gcb_block_bytes). The window is
    # therefore capped by STREAMING_GCB_MAX_SIZE regardless of ring, and the only per-config limit
    # is _build_global_cb's "at least two blocks" requirement: 2 * one_block <=
    # STREAMING_GCB_MAX_SIZE. The largest tile is bfloat8_b at 1088 B/tile (shared
    # BYTES_PER_TILE_BFP8).
    # NOTE: this gate assumes a single-row mesh (cluster_shape == (1, num_devices)), the only
    # topology this path is verified on (make_prefetcher rejects multi-row meshes up front). There,
    # attention.py allocates WO as K=dim//cluster_shape[0]=dim, N=dim//cluster_shape[1]=dim//
    # num_devices — i.e. (dim, wo_in_per_dev). The op_shapes WO entry below uses (wo_in_per_dev,
    # dim); divisibility holds because BOTH dim and wo_in_per_dev are checked above, and one block
    # is symmetric in K and N (k_tiles * n_total_tiles / ring^2), so the order is immaterial here.
    op_shapes = [
        (dim, n_hidden_per_dev),  # FF1/FF3: K=dim, N=hidden_dim/num_devices
        (n_hidden_per_dev, n_dim_per_dev),  # FF2: K=hidden_dim/num_devices, N=dim
        (dim, qkv_size_per_dev),  # attn QKV: K=dim, N=qkv_size/num_devices
        (wo_in_per_dev, n_dim_per_dev),  # attn WO: see NOTE above (K/N symmetric for this check)
    ]
    for k_dim, n_dim in op_shapes:
        k_tiles = k_dim // TILE
        n_tiles_per_recv = (n_dim // TILE) // ring_size  # already int-divisible by check above
        block_bytes = gcb_block_bytes(k_tiles, n_tiles_per_recv, ring_size, BYTES_PER_TILE_BFP8)
        if 2 * block_bytes > STREAMING_GCB_MAX_SIZE:
            return False
    return True


def _tallest_rectangle_height(grid_x: int, rows_left: int, num_cores: int) -> int:
    """Largest height ``h`` s.t. ``num_cores`` is an ``h x (num_cores/h)`` rectangle fitting
    ``rows_left`` rows and ``grid_x`` cols, or ``0`` if none fits. Shared by
    :func:`norm_grid_fits` and :meth:`TensorPrefetcher.dynamic_worker_core_grid` so the
    feasibility pre-check and the actual allocation can't disagree."""
    if rows_left <= 0:
        return 0
    return next(
        (h for h in range(min(rows_left, num_cores), 0, -1) if num_cores % h == 0 and num_cores // h <= grid_x),
        0,
    )


def norm_grid_fits(mesh_device: "ttnn.MeshDevice", recv_per_bank: int, num_cores: int = 32) -> bool:
    """Whether a ``num_cores`` decode-norm rectangle fits below a ``recv_per_bank``-row receiver
    rectangle on this mesh's worker grid.

    ``get_norm_config`` places the decode RMSNorm on ``dynamic_worker_core_grid(32)``, which
    anchors a solid rectangle in the rows *below* the receiver rectangle. A large ring (e.g.
    recv_per_bank=8 -> an 8-row receiver rectangle on a 10-row grid) can leave too few rows for
    it, so ``dynamic_worker_core_grid`` would assert. Gating candidate selection on this lets
    ``make_prefetcher`` pick a smaller ring (or fall back to the worker prefetcher) instead.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return _tallest_rectangle_height(grid.x, grid.y - recv_per_bank, num_cores) > 0


class TensorPrefetcher(LightweightModule):
    """Tensor-prefetcher state shared by Tensor-Prefetcher-fed model matmuls.

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
        # Lazy import to avoid the import cycle (this module is imported from prefetcher.py).
        from models.tt_transformers.tt.prefetcher import full_grid_core_range_set

        self.mesh_device: ttnn.MeshDevice = mesh_device
        self.num_tensors: int = num_tensors
        self.num_layers: int = num_layers
        self.num_senders: int = mesh_device.dram_grid_size().x
        self.model_name: str = os.getenv("HF_MODEL", "")
        assert self.model_name != "", "HF_MODEL is not set. Tensor Prefetcher must be run with a model."
        # Dual senders per bank and in1 streaming are always on — they are the validated,
        # highest-throughput configuration and the GCB sizing below assumes streaming.
        self.dual_senders_per_bank: bool = True
        self.stream_in1: bool = True
        self.uses_tensor_prefetcher: bool = True

        # Pick recv_per_bank. Auto-mode walks 8,4,2,1 (descending) and takes the largest supported.
        # A candidate must both satisfy the model/GCB config check AND leave room below the
        # receiver rectangle for the decode-norm grid (norm_grid_fits) — otherwise get_norm_config's
        # dynamic_worker_core_grid(32) would assert at decode time.
        candidates: List[int] = [num_receiver_cores] if num_receiver_cores is not None else [8, 4, 2, 1]
        picked = None
        for rpb in candidates:
            if is_tensor_prefetcher_config_supported(
                self.model_name, self.mesh_device.get_num_devices(), self.num_senders, rpb
            ) and norm_grid_fits(self.mesh_device, rpb):
                picked = rpb
                break
        assert picked is not None, (
            f"No supported recv_per_bank for {self.model_name} on {self.mesh_device.get_num_devices()} devices with "
            f"{self.num_senders} DRAM banks. Tried {candidates}."
        )
        self.num_receiver_cores: int = picked
        self.ring_size: int = self.num_senders * self.num_receiver_cores

        # Receiver rectangle is num_senders (cols) x num_receiver_cores (rows): ring position r
        # maps to (col=r%num_senders, row=r//num_senders).
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
        self.all_worker_cores_range_set: ttnn.CoreRangeSet = full_grid_core_range_set(self.mesh_device)

        # Per-bank receivers as a list aligned with bank indices. Bank b feeds the contiguous ring
        # arc [b*num_receiver_cores, (b+1)*num_receiver_cores), matching the CONTIGUOUS_1D
        # NdShardSpec required by the gather_in0 matmul's bank-to-receiver validation.
        self._bank_to_receivers: List = [
            (b, _bank_receivers_contiguous(b, self.num_receiver_cores, self.num_senders))
            for b in range(self.num_senders)
        ]

        # State.
        self.global_cb: Optional[ttnn.GlobalCircularBuffer] = None
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.registered_weights: List[ttnn.Tensor] = []
        self.registered_program_configs: List = []
        self.mode: Mode = Mode.PREFILL
        self._started: bool = False
        self._stopped: bool = False
        # Surrounding (non-GCB-matmul) ops — SDPA, create/concat heads, lm_head, rope — are NOT
        # co-located on the prefetcher's cores: the DRISC senders are off the worker grid and the
        # receiver ring is a small rectangle, so those ops use the model's default placement (the
        # same the no-prefetcher path uses). The worker-core backend sets this True and reserves
        # its worker grid for them. Config functions branch on this instead of the backend type.
        # NOTE: this co-location split is expected to change in the future (surrounding ops may be
        # placed on the freed worker grid for the DRAM-core backend too).
        self.colocate_ops: bool = False

        logger.info(
            f"[TensorPrefetcher] model={self.model_name} banks={self.num_senders} "
            f"recv/bank={self.num_receiver_cores} ring={self.ring_size}"
        )

    def insert_tensor(self, tensor: ttnn.Tensor, program_config=None) -> None:
        """Register one weight and its consuming decode program config for GCB sizing."""
        assert not self._started, "Cannot register weights after the Tensor prefetcher has started."
        assert program_config is not None, (
            "TensorPrefetcher.insert_tensor requires program_config (the 1D mcast gather_in0 matmul "
            "program config that will consume this weight). Threaded through from MLP/Attention."
        )
        assert hasattr(program_config, "per_core_N"), (
            "TensorPrefetcher.insert_tensor needs a 1D-mcast matmul program config exposing per_core_N "
            f"(used to size the GCB); got {type(program_config).__name__}."
        )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for Tensor Prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(f"Tensor volume ({tensor.volume()}) must be divisible by ring_size ({self.ring_size}).")
        self.registered_weights.append(tensor)
        self.registered_program_configs.append(program_config)
        logger.info(
            f"[TensorPrefetcher] Registered tensor of shape {tensor.shape} "
            f"({len(self.registered_weights)}/{self.num_tensors * self.num_layers})"
        )

    def init(self, mode: Mode = Mode.DECODE) -> None:
        """Set model mode and start the Tensor prefetcher when decode is first entered."""
        self.mode = mode
        if mode != Mode.DECODE or self._started:
            return
        assert not self._stopped, "TensorPrefetcher has been torn down; cannot run again."
        assert len(self.registered_weights) == self.num_tensors * self.num_layers, (
            f"Expected {self.num_tensors} * {self.num_layers} = {self.num_tensors * self.num_layers} inserted "
            f"tensors (num_tensors per layer), got {len(self.registered_weights)}."
        )
        self._build_global_cb()
        ttnn.experimental.start_tensor_prefetcher(self.mesh_device, dual_senders_per_bank=self.dual_senders_per_bank)
        self._started = True
        logger.info(f"[TensorPrefetcher.init] started decode prefetcher with ring={self.ring_size}")

    def teardown(self) -> None:
        if self._started and not self._stopped:
            ttnn.experimental.stop_tensor_prefetcher(self.mesh_device)
            self._stopped = True

    def __del__(self):
        # Best-effort cleanup. MeshDevice close has its own graceful fallback if we miss it.
        try:
            self.teardown()
        except Exception:
            pass

    def register_callback(self, callback) -> None:
        """Run ``callback`` now. The Tensor Prefetcher backend has no deferred prefetch phase, so weight
        registration happens immediately (the worker backend defers this to prefetch-time)."""
        callback()

    # The Tensor Prefetcher backend has no separate prefetch/run/stop phase — each matmul queues its own
    # request via ``prefetch_and_linear``. These no-ops let the model driver call the prefetcher
    # lifecycle uniformly (matching ``Prefetcher.prefetch``/``run``/``stop``) without branching.
    def prefetch(self) -> None:
        pass

    def run(self) -> None:
        pass

    def stop(self) -> None:
        pass

    # ---- Helpers expected by model_config.py and the model code ----

    def to_core_range_set(
        self, cores: List, return_list: bool = False
    ) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
        """Convert cores (CoreCoord/CoreRange/CoreRangeSet) to CoreRangeSet(s).

        Matches ``Prefetcher.to_core_range_set``.
        """
        from models.tt_transformers.tt.prefetcher import to_core_range_set

        return to_core_range_set(cores, return_list=return_list)

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRangeSet]:
        """Receiver cores in ring-position order, one CoreRangeSet per ring position.

        The model flattens this (via ``to_core_range_set``) into the matmul ``core_grid``; the
        gather_in0 matmul walks that grid in order, so ring core r computes N-cols
        [r*per_core_N, ...) and must receive shard r. This ring-position order is the same order
        ``_bank_to_receivers`` (contiguous per bank) feeds the DRAM-sender GCB: bank b's arc is
        ring positions [b*recv_per_bank, (b+1)*recv_per_bank).

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

    def weight_cache_suffix(self) -> str:
        """Cache-key discriminator for the on-disk weight layout this backend produces.

        Receiver-contiguous (ND_SHARDED) weights are not interchangeable with the worker-core
        backend's width-sharded weights, but the weight cache is keyed only by name/dtype/layout.
        Model code appends this so the two backends never reuse each other's cache files.
        """
        return "_recv_contig"

    def weight_mem_config(
        self, k: int, n: int, default: ttnn.MemoryConfig, is_galaxy: bool = False
    ) -> ttnn.MemoryConfig:
        """Memory config for a prefetched (K, N) weight (uniform with ``Prefetcher.weight_mem_config``).

        Always returns the receiver-contiguous DRAM layout this backend requires. ``is_galaxy`` is
        accepted only for signature parity with ``Prefetcher.weight_mem_config`` and is ignored:
        this backend is selected only on single-row Blackhole meshes (make_prefetcher rejects
        galaxy/TG and other multi-row meshes), so there is no galaxy case to fall back for — and a
        fallback here would be a half-fallback anyway, since the GCB (_build_global_cb) is
        recv-contig-only. Lets MLP/attention call ``prefetcher.weight_mem_config(...)`` without
        branching on backend.

        The recv-contig layout allocates the weight as an NdShardSpec with ``num_shards =
        ring_size`` (over-subscribed relative to the ``num_senders`` DRAM banks) distributed
        CONTIGUOUS_1D, each shard ``(K, N // ring_size)``. Paired with the contiguous per-bank GCB
        topology, shard r (columns [r*n_per_recv, (r+1)*n_per_recv)) is delivered to ring position
        r — exactly the weight slice the gather_in0 matmul's ring core r consumes.
        """
        del is_galaxy
        return recv_contig_mem_config(
            k,
            n,
            self.ring_size,
            self.num_senders,
            ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
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
        # leaves a width that fits the column count (shared with norm_grid_fits' pre-check).
        height = _tallest_rectangle_height(grid_x, rows_left, num_cores)
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
        # Lazy import to avoid hard dep order at module load (see is_tensor_prefetcher_config_supported).
        from models.tt_transformers.tt.prefetcher import TILE_BYTES

        assert len(self.registered_weights) == len(self.registered_program_configs)
        assert len(self.registered_weights) == self.num_tensors * self.num_layers

        # Size over every inserted weight (all layers), taking the largest block across all.
        max_in1_block_size = 0
        for tensor, pc in zip(self.registered_weights, self.registered_program_configs):
            tile_bytes = TILE_BYTES[tensor.dtype]
            # gather_in0 matmul splits in0_block_w to weight_K_tiles / ring_size, NOT pc.in0_block_w
            # (see gcb_block_bytes).
            weight_K = tensor.shape[-2]
            weight_K_tiles = weight_K // ttnn.TILE_SIZE
            assert weight_K_tiles % self.ring_size == 0, (
                f"Weight K_tiles {weight_K_tiles} must be divisible by ring_size {self.ring_size}; "
                "this should have been caught by is_tensor_prefetcher_config_supported."
            )
            in1_block_size = gcb_block_bytes(weight_K_tiles, pc.per_core_N, self.ring_size, tile_bytes)
            max_in1_block_size = max(max_in1_block_size, in1_block_size)
            logger.info(
                f"[TensorPrefetcher] tensor K={weight_K} N_per_core={pc.per_core_N} tile_bytes={tile_bytes} "
                f"in1_block_size={in1_block_size}"
            )

        # Batched delivery needs a full ring. Streaming can use a partial ring; retain as many blocks
        # as fit below the largest full-grid program's static-CB boundary. On Blackhole the usable
        # top-down L1 region ends at 1,519,552 and the LM-head static CBs end at 731,520, leaving
        # 788,032 bytes. For Llama-8B's 30,464-byte largest block this keeps 25/32 blocks, the deepest
        # whole-block window that fits.
        num_blocks = self.ring_size
        if self.stream_in1:
            num_blocks = min(self.ring_size, STREAMING_GCB_MAX_SIZE // max_in1_block_size)
            assert num_blocks >= 2, (
                f"Streaming GCB needs at least two blocks, but only {num_blocks} blocks of "
                f"{max_in1_block_size} bytes fit in the {STREAMING_GCB_MAX_SIZE}-byte budget."
            )
        gcb_size = num_blocks * max_in1_block_size
        logger.info(
            f"[TensorPrefetcher] Creating GCB: ring={self.ring_size} window={num_blocks} "
            f"max_in1={max_in1_block_size} size={gcb_size}"
        )
        self.global_cb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d_recv_contig(
            self.mesh_device,
            self.registered_program_configs,
            self.registered_weights,
            self._bank_to_receivers,
            gcb_size,
            dual_senders_per_bank=self.dual_senders_per_bank,
        )
