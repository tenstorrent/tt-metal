import math
from typing import Callable, Optional

import ttnn

from .common import DeepSeekV4Module, _HIFI4, width_sharded_l1_config
from .system_config import active_system_config
from .weight_cache import _CachePath, _load_weight, _materialize
import torch

from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb


def get_width_shard_num_cores(width: int, device, num_cores: Optional[int] = None) -> int:
    if num_cores is None:
        num_cores = width // ttnn.TILE_SIZE
    device_grid_size = device.compute_with_storage_grid_size()
    device_cores = device_grid_size.x * device_grid_size.y
    while num_cores > device_cores:
        num_cores //= 2
    return num_cores


def regular_width_sharded_l1_config(
    height: int, width: int, device, num_cores: Optional[int] = None
) -> ttnn.MemoryConfig:
    assert width % ttnn.TILE_SIZE == 0, f"width {width} must be tile-aligned"
    num_cores = get_width_shard_num_cores(width, device, num_cores)
    shard_width = width // num_cores
    height_padded = ((height + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True),
        [height_padded, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def to_ttnn_device(
    tensor: torch.Tensor,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    cache_file_name: Optional[str] = None,
) -> ttnn.Tensor:
    return _load_weight(tensor, device, cache_file_name=cache_file_name, layout=layout)


def _receiver_ring_cols(num_cores: int, device, preferred_width: Optional[int] = None) -> int:
    """Width of a ``num_cores`` rectangle of receiver cores anchored at (0, 0).

    ``matmul_decode`` walks the GCB's receiver cores row-major and treats position ``p`` as
    weight slab ``p``, so the receivers must form a full rectangle: only then does a core's
    row-major index equal ``row * width + col``, which is what ``_bank_receivers_strided``
    assumes when it places ring position ``p`` at ``(p % width, p // width)``. A ragged set
    (what ``num_cores_to_corerangeset`` yields for a non-multiple of the grid width) would
    shift the mapping and silently pair receivers with the wrong slab.

    ``preferred_width`` is used when it yields a rectangle that fits, so the partial mode can
    keep its natural ``n_blocks``-wide by ``k_blocks``-tall arrangement; otherwise the widest
    divisor that fits the device grid is used.
    """
    grid = device.compute_with_storage_grid_size()

    def fits(w):
        return w is not None and 0 < w <= grid.x and num_cores % w == 0 and num_cores // w <= grid.y

    width = preferred_width if fits(preferred_width) else next((w for w in range(grid.x, 0, -1) if fits(w)), None)
    if width is None:
        raise ValueError(f"cannot form a rectangle of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return width


def _bank_receivers_strided(bank_idx: int, recv_per_bank: int, num_dram_banks: int, ring_cols: int):
    """Receivers fed by DRAM bank ``bank_idx``, matching ROUND_ROBIN_1D shard placement.

    Under ROUND_ROBIN_1D, weight shard ``s`` lands on bank ``s % num_dram_banks``. Giving bank
    ``b`` the ring positions ``b, b + num_dram_banks, ...`` therefore makes shard index equal
    ring position, so no permutation of the weight is needed.
    """
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx + s * num_dram_banks
        coord = ttnn.CoreCoord(ring_pos % ring_cols, ring_pos // ring_cols)
        cores.append(ttnn.CoreRange(coord, coord))
    return ttnn.CoreRangeSet(cores)


def decode_weight_layout(
    K: int,
    N: int,
    partial_width_sharded: bool = False,
    k_blocks: Optional[int] = None,
    n_blocks: Optional[int] = None,
    batch: Optional[int] = None,
    b_blocks: Optional[int] = None,
):
    """``(num_b_cores, slab_shape, preferred_width)`` for a :class:`LinearDecode` /
    :class:`BatchedLinearDecode` weight.

    The single source of truth for how a weight is cut across B cores, shared by the layer
    and by :func:`make_shared_decode_gcb` so a shared GCB cannot be sized against a layout
    that differs from the one the layer actually builds.

    ``batch`` selects :class:`BatchedLinearDecode`'s layout: the weight is folded along both
    batch and N into a ``[Bc*K, Nc]`` block per core (``Bc = batch/b_blocks``,
    ``Nc = N/n_blocks``, ``b_blocks`` defaulting to ``batch`` as the class does), spread over
    ``b_blocks * n_blocks`` cores. It is checked first since ``partial_width_sharded`` has no
    meaning for a batched weight.
    """
    if batch is not None:
        b_blocks = batch if b_blocks is None else b_blocks
        if n_blocks is None:
            raise ValueError("batch=... requires n_blocks")
        if batch % b_blocks or N % n_blocks:
            raise ValueError(
                f"b_blocks ({b_blocks}) must divide batch ({batch}) and n_blocks ({n_blocks}) must divide N ({N})"
            )
        bc = batch // b_blocks
        return b_blocks * n_blocks, (bc * K, N // n_blocks), n_blocks
    if partial_width_sharded:
        if k_blocks is None or n_blocks is None:
            raise ValueError("partial_width_sharded=True requires k_blocks and n_blocks")
        return k_blocks * n_blocks, (K // k_blocks, N // n_blocks), n_blocks
    num_b_cores = N // 64 if n_blocks is None else n_blocks
    return num_b_cores, (K, N // num_b_cores), None


def _slab_tiles(slab_shape):
    """One receiver's weight slab as ``(rows, cols)`` in whole 32x32 tiles."""
    height, width = slab_shape
    if height % ttnn.TILE_SIZE or width % ttnn.TILE_SIZE:
        raise ValueError(f"weight slab {list(slab_shape)} must be tile-aligned")
    return height // ttnn.TILE_SIZE, width // ttnn.TILE_SIZE


def _tile_bytes(dtype: ttnn.DataType) -> int:
    return ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE]).get_tile_size(dtype)


def _slab_bytes(dtype: ttnn.DataType, slab_shape) -> int:
    """Byte size of one receiver's weight slab, in whole 32x32 tiles."""
    rows, cols = _slab_tiles(slab_shape)
    return rows * cols * _tile_bytes(dtype)


def decode_gcb_page_bytes(specs, dtype: ttnn.DataType) -> int:
    """The GCB page size that lets every weight in ``specs`` stream through one buffer.

    A GCB page is a block of whole K-rows of a receiver's slab, and every weight sharing a
    buffer must agree on the page size -- that is what replaces the old requirement that they
    all have the *same* slab size. The largest such page is the greatest common divisor of the
    slab sizes, which is what this returns, so a group streams in as few pages as its shapes
    allow.

    Pass the result to :func:`make_shared_decode_gcb` and to each :class:`LinearDecode` as
    ``global_cb_page_bytes``; both derive everything else from it, so neither can be sized
    against a page the other is not using.
    """
    if not specs:
        raise ValueError("decode_gcb_page_bytes needs at least one weight spec")
    tile_bytes = _tile_bytes(dtype)
    slabs = [_slab_tiles(decode_weight_layout(**spec)[1]) for spec in specs]
    page_tiles = 0
    for rows, cols in slabs:
        page_tiles = math.gcd(page_tiles, rows * cols)
    for rows, cols in slabs:
        # A page is whole rows of the slab, so dividing the slab evenly by bytes is not
        # enough: the page has to be a whole number of rows, and that many rows has to
        # divide the slab's row count. Shapes that fail this cannot share a buffer, and the
        # symptom if it were not caught here is a matmul that rejects the k_blocks it is
        # given -- or, if it did not, a ring whose pages straddle rows.
        if page_tiles % cols or rows % (page_tiles // cols):
            raise ValueError(
                f"these weights have no common GCB page: a {page_tiles}-tile page is not a whole number of "
                f"rows of a {rows}x{cols}-tile slab. Split them across separate GCBs."
            )
    return page_tiles * tile_bytes


def decode_gcb_k_blocks(slab_shape, dtype: ttnn.DataType, page_bytes: int) -> int:
    """How many GCB pages of ``page_bytes`` one ``slab_shape`` slab is cut into."""
    slab_bytes = _slab_bytes(dtype, slab_shape)
    rows, cols = _slab_tiles(slab_shape)
    if page_bytes <= 0 or slab_bytes % page_bytes:
        raise ValueError(f"a {page_bytes} B page does not divide this weight's {slab_bytes} B slab evenly")
    k_blocks = slab_bytes // page_bytes
    if rows % k_blocks:
        raise ValueError(
            f"a {page_bytes} B page cuts this weight's slab into {k_blocks} pages, but its {rows} tile-rows "
            f"do not divide into that many blocks of whole rows"
        )
    return k_blocks


def _receiver_cores_in_order(core_range_set: ttnn.CoreRangeSet):
    """The GCB's receiver cores in the order ``matmul_decode`` assigns weight slabs.

    Mirrors ``corerange_to_cores(..., row_wise=true)``: each ``CoreRange`` in stored order,
    expanded row-wise. Deliberately read back from the ``CoreRangeSet`` rather than
    recomputed from the ring geometry, so the layer's notion of "slab p goes to core p"
    is the device's, whether the GCB was built here or handed in.
    """
    cores = []
    for core_range in core_range_set.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def make_shared_decode_gcb(device, specs, dtype: ttnn.DataType, num_pages: int = 2):
    """One GCB that several :class:`LinearDecode` weights can be prefetched through.

    ``specs`` is a list of ``decode_weight_layout`` keyword dicts (``K``, ``N``,
    ``partial_width_sharded``, ``k_blocks``, ``n_blocks``) **in the order the matmuls will
    consume them**. Two things must match across them, and both are checked here because
    getting either wrong hangs the device rather than raising.

    Same number of B cores, since a GCB's receiver set is fixed at construction.

    Same *page* size -- not the same slab size. A weight whose slab is several pages is
    streamed: the prefetcher delivers it a page at a time and the matmul accumulates across
    them, so the ring's page size stays fixed however much the slabs differ. That uniformity
    is load-bearing: measured against this path, a ring whose page size changes between
    transfers hangs (three weights of 128 KB, 256 KB and 256 KB through one GCB hang, while
    the same three at a uniform size pass). Both ends do re-derive the page geometry per
    transfer and credit any skipped ring tail to the other side over NOC -- the DRISC sender
    via ``resize_remote_sender_cb_interface`` per request, the receiver via the
    ``setup_remote_cb_interfaces`` that BRISC firmware runs at every program launch -- but the
    failing case is the one that leaves the read pointer mid-ring and then has to realign it
    up to a larger page. Streaming sidesteps it by never changing the page.

    ``num_pages`` is the ring depth in pages. It sets how far ahead the prefetcher may run --
    a deep ring lets it work through several weights while the workers are still on the first,
    which is the whole reason to share one buffer -- and 2 is the floor, since the matmul holds
    one page un-acked while the next is delivered.

    What is *not* checked anywhere is order. One GCB is one FIFO, so requests must be queued
    in the same order the matmuls consume them; a consumer that runs out of turn pops a
    page belonging to another weight, which is wrong results rather than an error. Keep
    ``specs``, the queueing order, and the forward order in agreement.
    """
    if not specs:
        raise ValueError("make_shared_decode_gcb needs at least one weight spec")
    if num_pages < 2:
        raise ValueError(f"a shared GCB needs at least 2 pages of depth, got {num_pages}")
    layouts = [decode_weight_layout(**spec) for spec in specs]
    core_counts = {num_cores for num_cores, _, _ in layouts}
    if len(core_counts) != 1:
        raise ValueError(
            f"weights sharing a GCB must use the same number of B cores, but the specs want {sorted(core_counts)}"
        )
    num_b_cores = core_counts.pop()
    # The per-spec preferred widths can disagree, so fall back to the common rectangle rather
    # than letting whichever weight is built first pick the receiver set for the others.
    ring_cols = _receiver_ring_cols(num_b_cores, device, preferred_width=None)
    bank_to_receivers = _bank_to_receivers(num_b_cores, device, ring_cols)
    size = num_pages * decode_gcb_page_bytes(specs, dtype)
    return ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, size)


def _dram_banks_for(num_b_cores: int, device) -> int:
    """The DRAM bank count, having checked the weight's slabs spread evenly over it."""
    num_dram_banks = device.dram_grid_size().x
    if num_b_cores % num_dram_banks != 0:
        raise ValueError(
            f"the prefetcher needs the {num_b_cores} weight slabs to divide evenly across {num_dram_banks} DRAM banks"
        )
    return num_dram_banks


def _bank_to_receivers(num_b_cores: int, device, ring_cols: int):
    """``(dram_bank, receivers)`` pairs placing slab ``p`` on the ``p``-th receiver."""
    num_dram_banks = _dram_banks_for(num_b_cores, device)
    recv_per_bank = num_b_cores // num_dram_banks
    return [
        (bank, _bank_receivers_strided(bank, recv_per_bank, num_dram_banks, ring_cols))
        for bank in range(num_dram_banks)
    ]


def _coalesced_core_range_set(cores) -> ttnn.CoreRangeSet:
    """A ``CoreRangeSet`` over ``cores`` in the same coalesced form the device builds.

    ``CoreRangeSet`` equality compares the range decomposition, not the member cores, so a
    set built as one ``CoreRange`` per core is *not* equal to the same cores expressed as a
    rectangle -- and ``matmul_decode`` asserts the output grid equals the GCB's (coalesced)
    receiver grid. Merging one core at a time reproduces that decomposition.
    """
    core_range_set = ttnn.CoreRangeSet([])
    for core in cores:
        core_range_set = core_range_set.merge(ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}))
    return core_range_set


def _prefetch_cache_file(cache_file_name: Optional[str]) -> Optional[str]:
    """A distinct cache path for the prefetcher weight layout.

    The DRAM ND-sharded weight is a different tensor from the DRAM-interleaved one the
    L1-copy path caches (and in partial mode a different element order too, since the
    prefetcher layout is not K-block-folded), so the two must not share a cache file.
    """
    if cache_file_name is None:
        return None
    return _CachePath(f"{cache_file_name}_prefetch", getattr(cache_file_name, "require_cache", False))


class Linear(DeepSeekV4Module):
    """``nn.Linear`` (bias-free) as ``x @ Wᵀ`` for ttnn.

    ttnn ``linear`` computes ``a @ b`` with ``b`` shaped ``[in, out]``, so we
    store the torch ``[out, in]`` weight transposed.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        mesh_mapper=None,
    ):
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
            mesh_mapper=mesh_mapper,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class LinearDecode(DeepSeekV4Module):
    """Bias-free ``x @ Wᵀ`` backed by ``ttnn.experimental.matmul_decode``.

    Both operands stay L1 width-sharded and resident on the core grid, which is
    the layout the decode-optimized matmul kernel expects. The (static) weight is
    prepared and loaded once in the constructor; ``forward`` only reshards the
    incoming activation into the matching width-sharded L1 config before the op.

    Two weight layouts are supported, selected by ``partial_width_sharded``:

    - ``False`` (fully width-sharded): the torch ``[out, in]`` weight is stored as
      ``[K, N]`` and width(N)-sharded across ``N // 64`` cores (shard ``[K, N/cores]``),
      matching ``test_matmul_decode``. The full activation is gathered onto every core.
    - ``True`` (partial width-sharded): ``[K, N]`` is reshaped/permuted so a 2D
      ``(K_blocks x N_blocks)`` grid of ``[Kc, Nc]`` blocks maps across
      ``K_blocks * N_blocks`` cores (``Kc = K/K_blocks``, ``Nc = N/N_blocks``), and the
      K-partials are reduced across cores. Requires ``k_blocks`` and ``n_blocks``.

    ``use_prefetcher=True`` switches how the weight reaches the compute cores. Instead of
    holding it DRAM-interleaved and copying it into L1 width-sharded form before every call,
    it is stored DRAM ND-sharded -- one contiguous slab per B core -- and the DRISC tensor
    prefetcher pushes each slab into the matmul's in1 circular buffer through a
    GlobalCircularBuffer, off the command queue. The slab shape is the same per-core weight
    block as the L1 path (``[K, N/cores]`` or ``[Kc, Nc]``), except that the partial layout is
    *not* K-block-folded: the DRAM ND shard already enumerates the ``(K_blocks x N_blocks)``
    grid row-major, which is the receiver order the op consumes slabs in.

    ``keep_weights_in_l1=True`` copies the weight into its width-sharded L1 layout once, in
    the constructor, and leaves it there: ``forward`` then neither copies it in nor frees it
    afterwards, so a decode step pays no DRAM->L1 transfer for this weight at all. The
    DRAM-interleaved copy is released, since nothing reads it again.

    That is a permanent L1 allocation of the whole weight (``K * N`` bytes at the weight's
    dtype, spread over the B cores), so it only fits a few small projections: L1 also has to
    hold every activation, every op's circular buffers and any GCB on the device. A weight
    left resident that does not fit shows up as a later op failing to build, not as an error
    here. Mutually exclusive with ``use_prefetcher``, whose whole point is that the weight
    never lands in L1 as a tensor.

    ``fetch_weights`` keeps its meaning across both paths -- stage this layer's weights ahead
    of the call that needs them -- but here it queues the prefetch request rather than
    copying into L1, so the transfer overlaps whatever the workers are still doing. Calling
    it is optional; ``forward`` queues the request itself if nobody did. With
    ``keep_weights_in_l1`` there is nothing left to stage and it is a no-op.

    The caller owns the prefetcher session: wrap the forward passes in
    ``ttnn.experimental.start_tensor_prefetcher`` / ``stop_tensor_prefetcher`` (plus a
    ``wait_for_cq_on_tensor_prefetcher``), since one session should span a whole model step
    rather than a single layer. ``ttnn.experimental.is_tensor_prefetcher_supported(device)``
    reports whether the device has the programmable DRAM cores this needs.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        K: int = -1,
        N: int = -1,
        partial_width_sharded: bool = False,
        num_inputA_cores: int = 32,
        k_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
        use_prefetcher: bool = False,
        num_prefetch_slabs: Optional[int] = None,
        global_cb=None,
        global_cb_page_bytes: Optional[int] = None,
        keep_weights_in_l1: bool = False,
        mesh_mapper=None,
        packed_weight_tensor: Optional[ttnn.Tensor] = None,
        packed_weight_spec=None,
    ):
        self.partial_width_sharded = partial_width_sharded
        self.num_inputA_cores = num_inputA_cores
        self.dtype = dtype
        self.device = device
        self.l1_weights = None
        self.use_prefetcher = use_prefetcher
        self.keep_weights_in_l1 = keep_weights_in_l1
        self.global_cb = None
        self.mesh_mapper = mesh_mapper
        self.gcb_k_blocks = 1
        self.prefetch_queued = False
        self.packed_weight_tensor = packed_weight_tensor
        self.packed_weight_spec = packed_weight_spec

        if keep_weights_in_l1 and use_prefetcher:
            raise ValueError(
                "keep_weights_in_l1 and use_prefetcher are mutually exclusive: the prefetched weight "
                "is streamed into the matmul's in1 buffer and never held as an L1 tensor"
            )

        assert K != -1 and N != -1, "K and N must be set"
        self.K = K
        self.N = N
        if partial_width_sharded:
            self.n_blocks = n_blocks
            self.k_blocks = k_blocks

        num_inputB_cores, shard_shape, preferred_width = decode_weight_layout(
            K, N, partial_width_sharded, k_blocks, n_blocks
        )
        self.num_inputB_cores = num_inputB_cores
        if packed_weight_tensor is not None:
            if use_prefetcher or keep_weights_in_l1 or packed_weight_spec is None:
                raise ValueError("packed weights require a spec and are mutually exclusive with other weight paths")
            # The packed placement may deliberately use a different legal cut than
            # DECODE_LAYOUTS (for example kv_proj is full-width on Z2 rather than
            # partial-K). The packed spec is the source of truth for this path.
            self.partial_width_sharded = packed_weight_spec.k_blocks > 1
            self.k_blocks = packed_weight_spec.k_blocks
            self.n_blocks = packed_weight_spec.n_blocks
            return

        if use_prefetcher:
            self._init_prefetched_weight(
                weight,
                cache_file_name,
                dtype,
                num_inputB_cores,
                shard_shape,
                preferred_width=preferred_width,
                num_slabs=(
                    num_prefetch_slabs
                    if num_prefetch_slabs is not None
                    else active_system_config().prefetcher.num_prefetch_slabs
                ),
                shared_cb=global_cb,
                shared_cb_page_bytes=global_cb_page_bytes,
            )
            return

        b_core_range_set = ttnn.num_cores_to_corerangeset(
            num_inputB_cores, self.device.compute_with_storage_grid_size(), row_wise=True
        )
        self.weights_memory_config = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # The decode op wants the weight as [K, N]; torch nn.Linear stores [out=N, in=K].
        w = _materialize(weight, cache_file_name, dtype)
        if w is None:
            # Cache hit: the tilized, width-sharded weight is already on disk and its
            # serialized spec carries the real (width-sharded) layout, so none of the
            # K/N-derived shard-config or torch reshape work below is needed. ``as_tensor``
            # requires a ``memory_config`` when a device is given but ignores it on a
            # cache-hit load, so pass a throwaway config just to satisfy that guard.
            self.weight = ttnn.as_tensor(
                None,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
                mesh_mapper=mesh_mapper,
            )
            self._make_weights_resident()
            return

        w = w.t().contiguous()
        if partial_width_sharded:
            # Fold the K-blocks into the width so a width-sharded [Kc, Nc] block lands on
            # core c = kb * n_blocks + nb (row-major), matching the op's expected geometry.
            kc = shard_shape[0]
            w = w.reshape(k_blocks, kc, self.N).permute(1, 0, 2).reshape(kc, self.N * k_blocks)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
            mesh_mapper=mesh_mapper,
        )
        self._make_weights_resident()

    def _make_weights_resident(self):
        """Move the weight into L1 for good, under ``keep_weights_in_l1``.

        The DRAM-interleaved copy is freed: with the weight resident nothing reads it again,
        and holding both would pay for the layout twice.
        """
        if not self.keep_weights_in_l1:
            return
        self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        self.weight.deallocate()
        self.weight = None

    def _init_prefetched_weight(
        self,
        weight,
        cache_file_name,
        dtype,
        num_inputB_cores,
        slab_shape,
        preferred_width,
        num_slabs,
        shared_cb=None,
        shared_cb_page_bytes=None,
    ):
        """Store the weight DRAM ND-sharded and point the layer at the GCB it prefetches through.

        Slab ``p`` of the weight must reach the B core whose row-major index in the receiver
        rectangle is ``p``. That holds here because the weight is distributed ROUND_ROBIN_1D
        (slab ``p`` -> DRAM bank ``p % banks``) and the receivers are laid out with the matching
        stride, so slab index, ring position and receiver row-major index all coincide. Nothing
        on the device checks this pairing -- a mismatch is wrong results, not an error.

        ``shared_cb`` adopts a GCB built by :func:`make_shared_decode_gcb` instead of building a
        private one, which is how several projections avoid each paying for their own buffer.
        That builder laid its receivers out with the same bank stride used here, so slab ``p``
        still lands on receiver ``p`` and only the weight's DRAM side is set up in this case.
        ``shared_cb_page_bytes`` is that GCB's page size, from :func:`decode_gcb_page_bytes`;
        it decides how many pages this weight's slab is streamed as, and every weight on the
        buffer must be streamed at the same page size for the ring to stay in step.
        """
        num_dram_banks = _dram_banks_for(num_inputB_cores, self.device)
        if shared_cb is not None:
            receivers = shared_cb.receiver_cores()
            if receivers.num_cores() != num_inputB_cores:
                raise ValueError(
                    f"this weight needs {num_inputB_cores} B cores but the shared GCB has "
                    f"{receivers.num_cores()} receivers"
                )
            # Fall back to one page per slab so a caller who shares a buffer between
            # equally-shaped weights need not think about pages at all.
            page_bytes = shared_cb_page_bytes or _slab_bytes(dtype, slab_shape)
            self.gcb_k_blocks = decode_gcb_k_blocks(slab_shape, dtype, page_bytes)
            # A whole number of pages, not merely enough for one: a leftover partial page is
            # what leaves the ring pointer mid-page and forces the realign that hangs (see
            # make_shared_decode_gcb). Two are needed to stream, since the matmul holds one
            # page un-acked while the next is delivered.
            min_pages = 2 if self.gcb_k_blocks > 1 else 1
            if shared_cb.size() < min_pages * page_bytes or shared_cb.size() % page_bytes != 0:
                raise ValueError(
                    f"the shared GCB holds {shared_cb.size()} B per receiver, which is not at least {min_pages} "
                    f"whole {page_bytes} B page(s) -- it was almost certainly sized for a different page size"
                )

        dram_memory_config = ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                ttnn.Shape(list(slab_shape)),
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}),
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        )

        cache_file_name = _prefetch_cache_file(cache_file_name)
        w = _materialize(weight, cache_file_name, dtype)
        if w is not None:
            # torch nn.Linear stores [out=N, in=K]; the op wants [K, N]. Unlike the L1 partial
            # path there is no K-block fold -- the ND shard supplies that enumeration.
            # Keep global N here when a mesh mapper will cut it into per-rank
            # ``self.N`` slices after this host transform.
            w = w.t().contiguous().reshape(1, 1, self.K, -1)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=dram_memory_config,
            cache_file_name=cache_file_name,
            mesh_mapper=self.mesh_mapper,
        )
        if shared_cb is not None:
            self.global_cb = shared_cb
        else:
            ring_cols = _receiver_ring_cols(num_inputB_cores, self.device, preferred_width)
            self.global_cb = make_matmul_decode_gcb(
                self.device,
                self.weight,
                _bank_to_receivers(num_inputB_cores, self.device, ring_cols),
                slab_shape=slab_shape,
                num_pages=num_slabs,
            )
        # The op requires the output to live on the receiver cores (full mode asserts the two
        # grids are equal; partial mode reduces onto the first n_blocks of them in row-major
        # order), so keep them to build the output config from. Read back from the GCB so this
        # is the device's own slab-to-core order rather than a second guess at it.
        self.receiver_cores = _receiver_cores_in_order(self.global_cb.receiver_cores())

    def _prefetch_output_memory_config(self, m_padded: int) -> ttnn.MemoryConfig:
        """Width-sharded L1 output over the GCB's receiver cores.

        ``compute_output_specs`` derives exactly this for a rank-2 activation, but takes an
        earlier branch for a rank-4 one and falls back to DRAM-interleaved -- which the
        callers here (a sharded RMSNorm, another decode matmul) reject. Building it
        explicitly keeps both ranks on the same sharded layout.

        Full mode leaves the results spread over every receiver; partial mode reduces the
        K-partials onto the first ``n_blocks`` receivers in row-major order, so only those
        carry output.
        """
        if self.partial_width_sharded:
            num_output_cores = self.n_blocks
            grid = _coalesced_core_range_set(self.receiver_cores[:num_output_cores])
        else:
            num_output_cores = len(self.receiver_cores)
            # Taken from the GCB rather than rebuilt, so the equality the full width-sharded
            # factory asserts between the B grid and the output grid cannot fail on a
            # decomposition mismatch.
            grid = self.global_cb.receiver_cores()
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [m_padded, self.N // num_output_cores], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _queue_prefetch(self):
        """Ask the DRISC senders to push this weight's slabs into the GCB.

        Split out from the matmul so a caller can hoist it: the transfer then runs on the
        DRAM-core path, off the command queue, while earlier ops still occupy the workers.
        ``block_count`` is how many GCB pages a receiver's slab is cut into: 1 for a private
        buffer sized to a whole slab, and however many pages the shared buffer's page size
        makes of this slab otherwise. It must be the same number the matmul below is given.

        ``capture_into_trace`` is what makes this work under traced decode. A request is a
        host-side write to the DRISC senders, not a command-queue op, so a trace would not
        record it: capture would push weights that the captured (non-executing) matmuls never
        drain, and every replay would then wait on credits nobody posts. With the flag set, a
        request issued while the current queue is mid-capture is recorded against that trace
        and re-sent on each ``execute_trace`` instead. Outside capture it is sent immediately
        as before, so this is unconditional.
        """
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(self.weight, self.gcb_k_blocks)], global_cb=self.global_cb, capture_into_trace=True
        )
        self.prefetch_queued = True

    def fetch_weights(self):
        if self.packed_weight_tensor is not None:
            return
        if self.use_prefetcher:
            self._queue_prefetch()
            return
        if self.keep_weights_in_l1:
            return
        self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        # self.weight.deallocate()

    def get_input_memory_config(self, m: int, k: int, tile_height: int = ttnn.TILE_SIZE) -> ttnn.MemoryConfig:
        a_core_range_set = ttnn.num_cores_to_corerangeset(
            self.num_inputA_cores, self.device.compute_with_storage_grid_size(), row_wise=True
        )
        a_memory_config = ttnn.create_sharded_memory_config(
            (((m + tile_height - 1) // tile_height) * tile_height, k // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return a_memory_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.packed_weight_tensor is not None:
            # Packed placement is the source of truth: a preceding packed projection may
            # have left this activation on a different zone/core count.
            tile_height = x.get_tile().tile_shape[0]
            input_memory_config = self.get_input_memory_config(x.shape[-2], x.shape[-1], tile_height)
            same_core_grid = x.is_sharded() and (
                tile_height < ttnn.TILE_SIZE
                or _receiver_cores_in_order(x.memory_config().shard_spec.grid)
                == _receiver_cores_in_order(input_memory_config.shard_spec.grid)
            )
            if not same_core_grid:
                x = ttnn.to_memory_config(x, input_memory_config)
            m_padded = ((x.shape[-2] + tile_height - 1) // tile_height) * tile_height
            if self.partial_width_sharded:
                receiver_cores = _receiver_cores_in_order(self.packed_weight_spec.cores)
                output_cores = _coalesced_core_range_set(receiver_cores[: self.n_blocks])
                output_num_cores = self.n_blocks
            else:
                output_cores = self.packed_weight_spec.cores
                output_num_cores = self.packed_weight_spec.num_cores
            output_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    output_cores,
                    [m_padded, self.N // output_num_cores],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            return ttnn.experimental.matmul_decode(
                x,
                self.packed_weight_tensor,
                partial_width_sharded=self.partial_width_sharded,
                output_mem_config=output_memory_config,
                packed_weight=self.packed_weight_spec,
            )
        if self.use_prefetcher:
            if not x.is_sharded():
                x = ttnn.to_memory_config(x, self.get_input_memory_config(x.shape[-2], x.shape[-1]))
            # Exactly one queued request per matmul: the matmul waits for one page per
            # receiver, so a missing request hangs it and a doubled one desynchronises the
            # GCB pointers. ``fetch_weights`` may already have issued this call's request.
            if not self.prefetch_queued:
                self._queue_prefetch()
            self.prefetch_queued = False
            m_padded = ((x.shape[-2] + 31) // 32) * 32
            try:
                return ttnn.experimental.matmul_decode(
                    x,
                    self.weight,
                    partial_width_sharded=self.partial_width_sharded,
                    global_cb=self.global_cb,
                    global_cb_k_blocks=self.gcb_k_blocks,
                    output_mem_config=self._prefetch_output_memory_config(m_padded),
                )
            except Exception:
                # The request is already with the DRISC senders, and matmul_decode does most
                # of its validation while building the program -- so a rejected call leaves
                # slabs queued that nothing will ever drain. A clean stop cannot retire that:
                # its sentinel queues behind the orphaned request, so the kernel blocks on a
                # full GCB and never reaches it. Force-stopping abandons the kernels, which
                # keeps the failure an exception instead of a hang; stop is a no-op if no
                # prefetcher is running, so the caller's own stop still behaves.
                ttnn.experimental.stop_tensor_prefetcher(self.device, force=True)
                raise
        if self.l1_weights is None or not self.l1_weights.is_allocated():
            # A resident weight has no DRAM copy left to re-shard from, so losing it is a
            # bug in whoever freed it rather than something to silently rebuild.
            assert not self.keep_weights_in_l1, "the resident L1 weight was deallocated by someone else"
            self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        m = x.shape[-2]
        m_padded = ((m + 31) // 32) * 32
        if self.partial_width_sharded:
            # The partial layout reduces the K-partials onto n_blocks output cores, so shard the
            # output WIDTH_SHARDED across n_blocks cores (shard
            # [padded_m, N / n_blocks]).
            output_core_range_set = ttnn.num_cores_to_corerangeset(
                self.n_blocks, self.device.compute_with_storage_grid_size(), row_wise=True
            )
            output_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    output_core_range_set,
                    [m_padded, self.N // self.n_blocks],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        else:
            # Full-width matmul_decode requires B and output to use the exact
            # same core range. Deriving the output grid independently from N
            # diverges for smaller widths (for example TP o_b: N=1024 gives
            # 16 B cores but the generic activation helper chooses 32).
            output_memory_config = regular_width_sharded_l1_config(
                m_padded, self.N, self.device, num_cores=self.num_inputB_cores
            )
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(x.shape[-2], x.shape[-1]))
        result = ttnn.experimental.matmul_decode(
            x, self.l1_weights, partial_width_sharded=self.partial_width_sharded, output_mem_config=output_memory_config
        )
        if not self.keep_weights_in_l1:
            self.l1_weights.deallocate()
            self.l1_weights = None
        return result


class BatchedLinearDecode(DeepSeekV4Module):
    """Batched (block-diagonal) ``x[b] @ W[b]`` via ``ttnn.experimental.matmul_decode``.

    A rank-4 activation ``[d0, d1, M, K]`` (batch = ``d0*d1``) is matmul'd against a
    per-batch weight ``[batch, K, N]`` that is folded along BOTH batch and N into a
    width-sharded ``[1, 1, Bc*K, b_blocks*N]`` tensor (``Bc = batch / b_blocks``,
    ``Nc = N / n_blocks``) laid across a ``b_blocks x n_blocks`` core grid -- the layout
    the batched matmul_decode factory expects. The op infers ``b_blocks`` / ``n_blocks``
    from the operand shapes and emits a DRAM-interleaved ``[d0, d1, M, N]`` result.

    As in :class:`LinearDecode`, the (static) weight is prepared once here and only the
    activation is resharded per call. ``preprocess`` (optional) is applied to the raw
    torch weight on a cache MISS, *before* the batch/N fold, to normalize it to
    ``[batch, K, N]`` (e.g. the o_a reshape from ``[batch*N, K]``); it is skipped on a
    cache hit (the folded, tilized weight is already on disk).

    ``use_prefetcher=True`` switches the weight to the same DRISC-prefetched path
    :class:`LinearDecode` uses: the identically-folded ``[1, 1, Bc*K, b_blocks*N]`` tensor is
    stored DRAM ND-sharded (one ``[Bc*K, Nc]`` slab per B core) instead of L1 width-sharded,
    and the tensor prefetcher pushes each slab into the matmul's in1 buffer through a
    ``GlobalCircularBuffer``. Only the destination changes -- the fold, and so the B-core
    geometry a shared GCB must be sized against (``decode_weight_layout(..., batch=batch,
    ...)``), is the same either way. ``matmul_decode``'s rank-4 activation path always emits a
    DRAM-interleaved output regardless of the weight's source, so this needs no output-side
    sharding. See :class:`LinearDecode`'s docstring for the ``global_cb`` / session details,
    which carry over unchanged.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        batch: int,
        K: int,
        N: int,
        b_blocks: Optional[int] = None,
        n_blocks: Optional[int] = None,
        num_inputA_cores: int = 32,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_prefetcher: bool = False,
        num_prefetch_slabs: Optional[int] = None,
        global_cb=None,
        global_cb_page_bytes: Optional[int] = None,
        mesh_mapper=None,
        global_batch: Optional[int] = None,
        packed_weight_tensor: Optional[ttnn.Tensor] = None,
        packed_weight_spec=None,
    ):
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.K = K
        self.N = N
        self.num_inputA_cores = num_inputA_cores
        self.use_prefetcher = use_prefetcher
        self.global_cb = None
        self.mesh_mapper = mesh_mapper
        self.global_batch = global_batch if global_batch is not None else batch
        self.gcb_k_blocks = 1
        self.prefetch_queued = False
        self.packed_weight_tensor = packed_weight_tensor
        self.packed_weight_spec = packed_weight_spec

        # One batch per core row (Bc = 1) by default; widen N across as many cores as the grid
        # allows while keeping each N-shard tile-aligned.
        self.b_blocks = b_blocks if b_blocks is not None else batch
        if n_blocks is None:
            device_grid = device.compute_with_storage_grid_size()
            max_cores = device_grid.x * device_grid.y
            n_blocks = max(1, max_cores // self.b_blocks)
            while n_blocks > 1 and (N % n_blocks != 0 or (N // n_blocks) % ttnn.TILE_SIZE != 0):
                n_blocks -= 1
        self.n_blocks = n_blocks

        assert batch % self.b_blocks == 0, "b_blocks must divide batch"
        assert N % self.n_blocks == 0, "n_blocks must divide N"
        self.bc = batch // self.b_blocks
        self.nc = N // self.n_blocks
        if self.global_batch % self.batch:
            raise ValueError(f"global_batch {self.global_batch} must be divisible by local batch {self.batch}")
        if self.global_batch != self.batch and mesh_mapper is None:
            raise ValueError("global_batch may differ from batch only when mesh_mapper shards the folded weight")
        if packed_weight_tensor is not None:
            if use_prefetcher or mesh_mapper is not None or packed_weight_spec is None:
                raise ValueError(
                    "packed weights require a spec and are mutually exclusive with the prefetcher and mesh sharding"
                )
            return

        def fold(w):
            # The ordinary path folds local ``batch``. A mesh-sharded weight instead
            # folds ``global_batch`` into one width, then ShardTensorToMesh(dim=3)
            # gives each TP rank its contiguous local batch groups.
            fold_factor = self.global_batch // self.batch
            fold_b_blocks = self.b_blocks * fold_factor
            fold_bc = self.global_batch // fold_b_blocks
            return (
                w.reshape(fold_b_blocks, fold_bc, K, N)
                .permute(1, 2, 0, 3)
                .reshape(1, 1, fold_bc * K, fold_b_blocks * N)
                .contiguous()
            )

        if use_prefetcher:
            self._init_prefetched_weight(
                weight,
                cache_file_name,
                dtype,
                preprocess,
                fold,
                slab_shape=(self.bc * K, self.nc),
                num_slabs=(
                    num_prefetch_slabs
                    if num_prefetch_slabs is not None
                    else active_system_config().prefetcher.num_prefetch_slabs
                ),
                shared_cb=global_cb,
                shared_cb_page_bytes=global_cb_page_bytes,
            )
            return

        b_core_range_set = ttnn.num_cores_to_corerangeset(
            self.b_blocks * self.n_blocks, device.compute_with_storage_grid_size(), row_wise=True
        )
        self.weights_memory_config = ttnn.create_sharded_memory_config(
            (self.bc * K, self.nc),
            core_grid=b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        w = _materialize(weight, cache_file_name, dtype)
        if w is not None:
            if preprocess is not None:
                w = preprocess(w)
            w = fold(w)
        self.weight = _load_weight(
            w, device, cache_file_name=cache_file_name, dtype=dtype, mesh_mapper=self.mesh_mapper
        )

    def _init_prefetched_weight(
        self,
        weight,
        cache_file_name,
        dtype,
        preprocess,
        fold,
        slab_shape,
        num_slabs,
        shared_cb=None,
        shared_cb_page_bytes=None,
    ):
        """Store the weight DRAM ND-sharded and point the layer at the GCB it prefetches through.

        Mirrors :meth:`LinearDecode._init_prefetched_weight` -- see its docstring for the
        slab-to-receiver pairing and the ``shared_cb`` / ``shared_cb_page_bytes`` contract, both
        unchanged here. The only difference is the weight tensor itself: it is folded along
        batch and N exactly as the L1 path folds it (``fold``), so the per-receiver slab the ND
        shard cuts from the folded ``[1, 1, Bc*K, b_blocks*N]`` tensor is the same ``[Bc*K,
        Nc]`` block the L1 path width-shards -- only the destination (DRAM ND-shard vs. L1
        width-shard) differs.
        """
        num_b_cores = self.b_blocks * self.n_blocks
        num_dram_banks = _dram_banks_for(num_b_cores, self.device)
        if shared_cb is not None:
            receivers = shared_cb.receiver_cores()
            if receivers.num_cores() != num_b_cores:
                raise ValueError(
                    f"this weight needs {num_b_cores} B cores but the shared GCB has "
                    f"{receivers.num_cores()} receivers"
                )
            page_bytes = shared_cb_page_bytes or _slab_bytes(dtype, slab_shape)
            self.gcb_k_blocks = decode_gcb_k_blocks(slab_shape, dtype, page_bytes)
            min_pages = 2 if self.gcb_k_blocks > 1 else 1
            if shared_cb.size() < min_pages * page_bytes or shared_cb.size() % page_bytes != 0:
                raise ValueError(
                    f"the shared GCB holds {shared_cb.size()} B per receiver, which is not at least {min_pages} "
                    f"whole {page_bytes} B page(s) -- it was almost certainly sized for a different page size"
                )

        dram_memory_config = ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                ttnn.Shape(list(slab_shape)),
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}),
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        )

        cache_file_name = _prefetch_cache_file(cache_file_name)
        w = _materialize(weight, cache_file_name, dtype)
        if w is not None:
            if preprocess is not None:
                w = preprocess(w)
            w = fold(w)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=dram_memory_config,
            cache_file_name=cache_file_name,
            mesh_mapper=self.mesh_mapper,
        )
        if shared_cb is not None:
            self.global_cb = shared_cb
        else:
            ring_cols = _receiver_ring_cols(num_b_cores, self.device, preferred_width=self.n_blocks)
            self.global_cb = make_matmul_decode_gcb(
                self.device,
                self.weight,
                _bank_to_receivers(num_b_cores, self.device, ring_cols),
                slab_shape=slab_shape,
                num_pages=num_slabs,
            )
        self.receiver_cores = _receiver_cores_in_order(self.global_cb.receiver_cores())

    def _queue_prefetch(self):
        """Ask the DRISC senders to push this weight's slabs into the GCB. See
        :meth:`LinearDecode._queue_prefetch`."""
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(self.weight, self.gcb_k_blocks)], global_cb=self.global_cb, capture_into_trace=True
        )
        self.prefetch_queued = True

    def fetch_weights(self):
        if self.packed_weight_tensor is not None:
            return
        if self.use_prefetcher:
            self._queue_prefetch()

    def deallocate(self):
        pass

    def get_input_memory_config(self, m: int, tile_height: int = ttnn.TILE_SIZE) -> ttnn.MemoryConfig:
        # Activation A is width(K)-sharded: shard [batch * m_padded, K / num_inputA_cores].
        m_padded = ((m + tile_height - 1) // tile_height) * tile_height
        a_core_range_set = ttnn.num_cores_to_corerangeset(
            self.num_inputA_cores, self.device.compute_with_storage_grid_size(), row_wise=True
        )
        return ttnn.create_sharded_memory_config(
            (self.batch * m_padded, self.K // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: rank-4 [d0, d1, M, K] with d0*d1 == batch. Reshard to the width(K)-sharded L1 layout,
        # then run the batched matmul_decode (b_blocks / n_blocks are inferred from the shapes).
        m = x.shape[-2]
        if self.packed_weight_tensor is not None:
            input_memory_config = self.get_input_memory_config(m, x.get_tile().tile_shape[0])
            same_core_grid = x.is_sharded() and (
                x.get_tile().tile_shape[0] < ttnn.TILE_SIZE
                or _receiver_cores_in_order(x.memory_config().shard_spec.grid)
                == _receiver_cores_in_order(input_memory_config.shard_spec.grid)
            )
            if not same_core_grid:
                x = ttnn.to_memory_config(x, input_memory_config)
            return ttnn.experimental.matmul_decode(x, self.packed_weight_tensor, packed_weight=self.packed_weight_spec)
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(m))
        if self.use_prefetcher:
            # Exactly one queued request per matmul, as in LinearDecode.forward: a missing
            # request hangs it and a doubled one desynchronises the GCB pointers.
            if not self.prefetch_queued:
                self._queue_prefetch()
            self.prefetch_queued = False
            try:
                return ttnn.experimental.matmul_decode(
                    x, self.weight, global_cb=self.global_cb, global_cb_k_blocks=self.gcb_k_blocks
                )  # DRAM-interleaved [d0, d1, M, N]
            except Exception:
                # See LinearDecode.forward: the request is already with the DRISC senders, so a
                # rejected call must force-stop rather than leave slabs nothing will ever drain.
                ttnn.experimental.stop_tensor_prefetcher(self.device, force=True)
                raise
        l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        y = ttnn.experimental.matmul_decode(x, l1_weights)  # DRAM-interleaved [d0, d1, M, N]
        l1_weights.deallocate()
        return y


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(
        self, weight, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None, sharded: bool = False
    ):
        w = _materialize(weight, cache_file_name, ttnn.bfloat16)
        self.weight = _load_weight(
            w.reshape(1, 1, 1, -1) if w is not None else None, device, cache_file_name=cache_file_name
        )
        self.eps = eps
        self.sharded = sharded

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        caller_shape = None
        if self.sharded:
            b, s, t, d = x.shape
            rows = b * s * t
            # The width-sharded L1 layout gives every core the *full* height (one
            # tile-width each, so only ``d // TILE_SIZE`` cores), which means its L1
            # footprint grows with the row count. That is a win for the single-token
            # decode activations it was added for, but the compressor's pooled entries
            # are ``max_seq // compress_rate`` rows tall: by max_seq 32k the shards plus
            # rms_norm's own CBs reach ~2.8 MB against a 1.5 MB budget, and even below
            # that the interleaved path is measurably faster past one tile-row. So only
            # shard while the whole tensor is a single tile-row.
            if rows <= ttnn.TILE_SIZE:
                # A batched decode arrives one tile-row per user (``[B,1,1,D]``), so those
                # ``rows`` sit in ``B`` separate tile-rows and a shard tall enough for the
                # padding would be ``B`` times the one that holds the data. Pack them onto
                # a single tile-row first -- the layout the projections already use, so a
                # B-user norm costs what a one-user norm does -- and hand the caller its
                # own shape back afterwards.
                if x.shape[-2] != rows:
                    caller_shape = list(x.shape)
                    x = ttnn.reshape(x, [1, 1, rows, d])
                x = ttnn.to_memory_config(x, width_sharded_l1_config(rows, d, x.device()))
        # Keep the output in the same sharded layout the norm just
        # consumed, so the next op (LinearDecode / _apply_rope) doesn't round-trip
        # through DRAM-interleaved. ``ttnn.rms_norm`` requires a sharded output to
        # match the input's memory layout, so passing the input's own config is the
        # documented contract; for DRAM-interleaved inputs this is the default anyway.
        assert x.is_sharded(), "input must be sharded"
        out = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps, memory_config=x.memory_config())
        return out if caller_shape is None else ttnn.reshape(out, caller_shape)


def _rms_norm_unweighted(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim (matches ``DeepseekV4UnweightedRMSNorm``)."""
    # if not x.is_sharded():
    #     b, s, t, d = x.shape
    #     x_mem_config = width_sharded_l1_config(b * s * t, d, x.device())
    #     x = ttnn.to_memory_config(x, x_mem_config)
    # See ``DeepSeekV4RMSNorm.forward``: keep the output in the input's (sharded)
    # layout instead of dropping to DRAM-interleaved, so the next op avoids a
    # sharded->DRAM->sharded round-trip.
    assert x.is_sharded(), "input must be sharded"
    return ttnn.rms_norm(x, epsilon=eps, memory_config=x.memory_config())
