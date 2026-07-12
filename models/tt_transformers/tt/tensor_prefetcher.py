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
from models.tt_transformers.tt.prefetcher_config import (
    ARCH_CONFIG,
    TensorPrefetcherReceiverLayout,
    allocate_tensor_prefetcher_receiver_layout,
)
from models.tt_transformers.tt.recv_contig_layout import recv_contig_mem_config

STREAMING_GCB_MAX_SIZE = 788032


def get_tensor_prefetcher_receiver_layout(
    mesh_device: ttnn.MeshDevice, receivers_per_bank: int
) -> Optional[TensorPrefetcherReceiverLayout]:
    """Allocate one logical receiver ring shared by every device in ``mesh_device``."""
    try:
        grid = mesh_device.compute_with_storage_grid_size()
        num_dram_banks = mesh_device.dram_grid_size().x
        if hasattr(mesh_device, "get_optimal_dram_bank_to_logical_worker_assignments"):
            device_anchors = mesh_device.get_optimal_dram_bank_to_logical_worker_assignments(ttnn.NOC.NOC_0)
        elif hasattr(mesh_device, "get_devices"):
            device_anchors = [
                device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
                for device in mesh_device.get_devices()
            ]
        else:
            device_anchors = [mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]

        anchor_sets = tuple(tuple((anchor.x, anchor.y) for anchor in anchors) for anchors in device_anchors)
        if not anchor_sets or any(len(anchors) != num_dram_banks for anchors in anchor_sets):
            return None
        right_anchor_sets = tuple(anchors[num_dram_banks // 2 :] for anchors in anchor_sets)
        if any(not right_anchors for right_anchors in right_anchor_sets):
            return None

        layout = allocate_tensor_prefetcher_receiver_layout(
            (grid.x, grid.y),
            anchor_sets[0],
            num_dram_banks,
            receivers_per_bank,
            device_right_starts=tuple(
                min(anchor[0] for anchor in right_anchors) for right_anchors in right_anchor_sets
            ),
        )
        if layout is None:
            return None

        # MeshDevice validates that each logical worker has the same translated address on
        # every device. Differing physical harvesting is handled by each device's translation.
        for x, y in layout.receiver_coords:
            mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    except (AttributeError, RuntimeError, ValueError):
        return None

    return layout


def select_tensor_prefetcher_receiver_layout(
    mesh_device: ttnn.MeshDevice,
    model_name: str,
    num_devices: int,
    receiver_candidates: List[int],
) -> Optional[TensorPrefetcherReceiverLayout]:
    """Select the largest model-valid receiver profile that the device can allocate."""
    num_dram_banks = mesh_device.dram_grid_size().x
    for receivers_per_bank in receiver_candidates:
        if not is_tensor_prefetcher_config_supported(model_name, num_devices, num_dram_banks, receivers_per_bank):
            continue
        layout = get_tensor_prefetcher_receiver_layout(mesh_device, receivers_per_bank)
        if layout is not None:
            return layout
    return None


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
    """Strict divisibility + GCB-size check for the Tensor Prefetcher path.

    The DRAM-core path has the same uniform-receivers requirement as the worker path,
    plus an actual ``n_per_device_tiles % ring_size == 0`` check that the worker-side
    ``is_prefetcher_supported`` skips. Also enforces ``num_blocks * in1_block_size <=
    L1_PER_BANK_BUDGET`` (the GCB allocation per receiver). Returns True iff every weight
    class (FF1/FF3, FF2, QKV, WO) divides cleanly across
    ``ring_size = num_dram_banks * recv_per_bank`` AND its per-receiver GCB footprint
    fits worker L1.
    """
    # Import here to avoid circular import (this module is imported from prefetcher.py).
    from models.tt_transformers.tt.prefetcher import BYTES_PER_TILE_BFP8, resolve_verified_model_cfg

    cfg = resolve_verified_model_cfg(model_name)
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
    # multi-row mesh (cluster_shape[0] > 1) is not validated here; weight_mem_config's
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
        # Per-receiver footprint = num_blocks (=ring_size) * one block, bfloat8_b being the largest.
        per_recv_bytes = ring_size * gcb_block_bytes(k_tiles, n_tiles_per_recv, ring_size, BYTES_PER_TILE_BFP8)
        if per_recv_bytes > L1_BUDGET:
            return False
    return True


class TensorPrefetcher(LightweightModule):
    """Tensor-prefetcher state shared by Tensor-Prefetcher-fed model matmuls.

    Bank ``b``'s receivers occupy contiguous ring positions ``[b*r, (b+1)*r)``. Receiver
    profiles are loaded from ``prefetcher_config.yaml`` and materialized on the runtime
    logical worker grid, including deterministic fallback when worker columns are harvested.
    DRAM sender cores live on a separate programmable core type and don't occupy worker grid
    coordinates.
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
        self.dual_senders_per_bank: bool = os.getenv("TT_METAL_TENSOR_PREFETCHER_DUAL_SENDERS", "0") == "1"
        self.stream_in1: bool = os.getenv("TT_METAL_TENSOR_PREFETCHER_STREAM_IN1", "0") == "1"
        self.colocate_attention_ops: bool = os.getenv("TT_METAL_TENSOR_PREFETCHER_COLOCATE_ATTENTION_OPS", "0") == "1"
        self.colocate_lm_head: bool = os.getenv("TT_METAL_TENSOR_PREFETCHER_COLOCATE_LM_HEAD", "0") == "1"
        self.colocation_start_core: ttnn.CoreCoord = ttnn.CoreCoord(0, 0)
        self.uses_tensor_prefetcher: bool = True

        # Pick recv_per_bank only when both model geometry and runtime core allocation support it.
        configured_candidates = ARCH_CONFIG["blackhole"]["tensor_prefetcher"]["receiver_candidates"]
        candidates: List[int] = [num_receiver_cores] if num_receiver_cores is not None else configured_candidates
        receiver_layout = select_tensor_prefetcher_receiver_layout(
            self.mesh_device,
            self.model_name,
            self.mesh_device.get_num_devices(),
            candidates,
        )
        assert receiver_layout is not None, (
            f"No supported recv_per_bank for {self.model_name} on {self.mesh_device.get_num_devices()} devices with "
            f"{self.num_senders} DRAM banks. Tried {candidates}."
        )
        self.num_receiver_cores: int = receiver_layout.ring_rows
        self.ring_size: int = self.num_senders * self.num_receiver_cores
        self.ring_cols: int = receiver_layout.ring_cols
        self.ring_rows: int = receiver_layout.ring_rows
        self.receiver_layout_name: str = receiver_layout.profile_name
        self.receiver_layout_used_fallback: bool = receiver_layout.used_fallback

        # Receiver cores in ring-position order. The gather_in0 matmul walks its core_grid in
        # this order, so ring core r computes output N-cols [r*per_core_N, (r+1)*per_core_N).
        # receiver_cores() returns this list (flattened) as the matmul grid.
        self._ring_cores: List[ttnn.CoreCoord] = [ttnn.CoreCoord(x, y) for x, y in receiver_layout.receiver_coords]

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
            (
                b,
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(core, core)
                        for core in self._ring_cores[b * self.num_receiver_cores : (b + 1) * self.num_receiver_cores]
                    ]
                ),
            )
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
        # Preserve the original aggregate flag for compatibility. Attention/RoPE and LM-head
        # colocation are independently opt-in because they have different placement and L1 risks.
        self.colocate_ops: bool = False

        logger.info(
            f"[TensorPrefetcher] model={self.model_name} banks={self.num_senders} "
            f"recv/bank={self.num_receiver_cores} ring={self.ring_size} "
            f"layout={self.receiver_layout_name} fallback={self.receiver_layout_used_fallback} "
            f"dual_senders={self.dual_senders_per_bank} stream_in1={self.stream_in1} "
            f"colocate_attention={self.colocate_attention_ops} colocate_lm_head={self.colocate_lm_head}"
        )
        if self.colocate_lm_head:
            logger.warning(
                "[TensorPrefetcher] LM-head colocation is experimental: isolated PCC passes, but full-model "
                "generation is not yet coherent."
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
        [r*per_core_N, ...) and must receive shard r. This ring order is intentionally
        grouped into contiguous per-bank arcs in ``_bank_to_receivers``.

        ``sender_active``/``receiver_active`` are accepted for parity with the worker-core
        ``Prefetcher.receiver_cores`` signature but are ignored — the DRAM-core path has no
        inactive subset (every DRAM bank is a sender and every configured ring core is active).
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

        Returns the receiver-contiguous DRAM layout this backend requires, except on galaxy/TG
        meshes (the recv-contig path is not supported there) where it falls back to ``default``.
        Lets MLP/attention call ``prefetcher.weight_mem_config(...)`` without branching on backend.

        The recv-contig layout allocates the weight as an NdShardSpec with ``num_shards =
        ring_size`` (over-subscribed relative to the ``num_senders`` DRAM banks) distributed
        contiguously by bank, each shard ``(K, N // ring_size)``. Paired with the contiguous GCB topology,
        shard r (columns [r*n_per_recv, (r+1)*n_per_recv)) is delivered to ring position r —
        exactly the weight slice the gather_in0 matmul's ring core r consumes.
        """
        if is_galaxy:
            return default
        return recv_contig_mem_config(
            k,
            n,
            self.ring_size,
            self.num_senders,
            ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
        )

    def dynamic_worker_core_grid(self, num_cores: int) -> ttnn.CoreRangeSet:
        """Allocate ``num_cores`` worker cores as a solid model-helper rectangle.

        The returned grid remains anchored below row ``num_receiver_cores`` to preserve the existing
        model configurations. With a sparse configured receiver placement it can intersect receiver cores;
        this is supported because the persistent GCB and model programs use disjoint L1 regions.
        It must be filled (``num_cores() == bounding-box``) because consumers like residual RMSNorm
        require a rectangular shard grid.

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
