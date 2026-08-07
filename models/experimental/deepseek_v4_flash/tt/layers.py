from typing import Callable, Optional

import ttnn

from .common import DeepSeekV4Module, _HIFI4, width_sharded_l1_config
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
    K: int, N: int, partial_width_sharded: bool = False, k_blocks: Optional[int] = None, n_blocks: Optional[int] = None
):
    """``(num_b_cores, slab_shape, preferred_width)`` for a :class:`LinearDecode` weight.

    The single source of truth for how a weight is cut across B cores, shared by the layer
    and by :func:`make_shared_decode_gcb` so a shared GCB cannot be sized against a layout
    that differs from the one the layer actually builds.
    """
    if partial_width_sharded:
        if k_blocks is None or n_blocks is None:
            raise ValueError("partial_width_sharded=True requires k_blocks and n_blocks")
        return k_blocks * n_blocks, (K // k_blocks, N // n_blocks), n_blocks
    num_b_cores = N // 64 if n_blocks is None else n_blocks
    return num_b_cores, (K, N // num_b_cores), None


def _slab_bytes(dtype: ttnn.DataType, slab_shape) -> int:
    """Byte size of one receiver's weight slab, in whole 32x32 tiles."""
    height, width = slab_shape
    if height % ttnn.TILE_SIZE or width % ttnn.TILE_SIZE:
        raise ValueError(f"weight slab {list(slab_shape)} must be tile-aligned")
    tiles = (height // ttnn.TILE_SIZE) * (width // ttnn.TILE_SIZE)
    return tiles * ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE]).get_tile_size(dtype)


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


def make_shared_decode_gcb(device, specs, dtype: ttnn.DataType, num_slabs: int = 1):
    """One GCB that several :class:`LinearDecode` weights can be prefetched through.

    ``specs`` is a list of ``decode_weight_layout`` keyword dicts (``K``, ``N``,
    ``partial_width_sharded``, ``k_blocks``, ``n_blocks``) **in the order the matmuls will
    consume them**. Two things must match across them, and both are checked here because
    getting either wrong hangs the device rather than raising.

    Same number of B cores, since a GCB's receiver set is fixed at construction.

    Same slab size, which is the harder constraint. A GCB page is one slab, so weights of
    different sizes would change the ring's page size from one transfer to the next. Both
    ends do re-derive the page geometry per transfer and are written to credit any skipped
    ring tail to the other side over NOC -- the DRISC sender via
    ``resize_remote_sender_cb_interface`` per request, the receiver via the
    ``setup_remote_cb_interfaces`` that BRISC firmware runs at every program launch -- but
    measured against this path that does not hold: three weights of 128 KB, 256 KB and
    256 KB through one GCB hang, while the same three at a uniform size pass, as does a
    single weight repeatedly wrapping a two-slab ring. The failing case is the one that
    leaves the read pointer mid-ring and then has to realign it up to a larger page. Until
    that is understood, group weights by slab size and give each group its own GCB.

    What is *not* checked anywhere is order. One GCB is one FIFO, so requests must be queued
    in the same order the matmuls consume them; a consumer that runs out of turn pops a
    page belonging to another weight, which is wrong results rather than an error. Keep
    ``specs``, the queueing order, and the forward order in agreement.
    """
    if not specs:
        raise ValueError("make_shared_decode_gcb needs at least one weight spec")
    layouts = [decode_weight_layout(**spec) for spec in specs]
    core_counts = {num_cores for num_cores, _, _ in layouts}
    if len(core_counts) != 1:
        raise ValueError(
            f"weights sharing a GCB must use the same number of B cores, but the specs want {sorted(core_counts)}"
        )
    num_b_cores = core_counts.pop()
    slab_sizes = {_slab_bytes(dtype, slab_shape) for _, slab_shape, _ in layouts}
    if len(slab_sizes) != 1:
        raise ValueError(
            f"weights sharing a GCB must have the same slab size, but the specs give {sorted(slab_sizes)} B; "
            "a GCB whose page size changes between transfers hangs, so group these by size and build one GCB "
            "per group"
        )
    # The per-spec preferred widths can disagree, so fall back to the common rectangle rather
    # than letting whichever weight is built first pick the receiver set for the others.
    ring_cols = _receiver_ring_cols(num_b_cores, device, preferred_width=None)
    bank_to_receivers = _bank_to_receivers(num_b_cores, device, ring_cols)
    size = num_slabs * slab_sizes.pop()
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
    ):
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
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

    ``fetch_weights`` keeps its meaning across both paths -- stage this layer's weights ahead
    of the call that needs them -- but here it queues the prefetch request rather than
    copying into L1, so the transfer overlaps whatever the workers are still doing. Calling
    it is optional; ``forward`` queues the request itself if nobody did.

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
        num_prefetch_slabs: int = 2,
        global_cb=None,
    ):
        self.partial_width_sharded = partial_width_sharded
        self.num_inputA_cores = num_inputA_cores
        self.dtype = dtype
        self.device = device
        self.l1_weights = None
        self.use_prefetcher = use_prefetcher
        self.global_cb = None
        self.prefetch_queued = False

        assert K != -1 and N != -1, "K and N must be set"
        self.K = K
        self.N = N
        if partial_width_sharded:
            self.n_blocks = n_blocks
            self.k_blocks = k_blocks

        num_inputB_cores, shard_shape, preferred_width = decode_weight_layout(
            K, N, partial_width_sharded, k_blocks, n_blocks
        )

        if use_prefetcher:
            self._init_prefetched_weight(
                weight,
                cache_file_name,
                dtype,
                num_inputB_cores,
                shard_shape,
                preferred_width=preferred_width,
                num_slabs=num_prefetch_slabs,
                shared_cb=global_cb,
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
            )
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
        )

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
        """
        num_dram_banks = _dram_banks_for(num_inputB_cores, self.device)
        if shared_cb is not None:
            receivers = shared_cb.receiver_cores()
            if receivers.num_cores() != num_inputB_cores:
                raise ValueError(
                    f"this weight needs {num_inputB_cores} B cores but the shared GCB has "
                    f"{receivers.num_cores()} receivers"
                )
            # A whole number of this weight's slabs, not merely enough for one: a leftover
            # partial slab is what leaves the ring pointer mid-page and forces the realign
            # that hangs (see make_shared_decode_gcb).
            slab_bytes = _slab_bytes(dtype, slab_shape)
            if shared_cb.size() < slab_bytes or shared_cb.size() % slab_bytes != 0:
                raise ValueError(
                    f"the shared GCB holds {shared_cb.size()} B per receiver, which is not a whole number of "
                    f"this weight's {slab_bytes} B slabs -- it was almost certainly sized for a differently "
                    f"shaped weight"
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
            w = w.t().contiguous().reshape(1, 1, self.K, self.N)
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=dram_memory_config,
            cache_file_name=cache_file_name,
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
                num_slabs=num_slabs,
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
        ``block_count`` is 1 in every mode -- a receiver's whole slab is one GCB page,
        because the compute kernel indexes in1 tiles by absolute position within it.

        ``capture_into_trace`` is what makes this work under traced decode. A request is a
        host-side write to the DRISC senders, not a command-queue op, so a trace would not
        record it: capture would push weights that the captured (non-executing) matmuls never
        drain, and every replay would then wait on credits nobody posts. With the flag set, a
        request issued while the current queue is mid-capture is recorded against that trace
        and re-sent on each ``execute_trace`` instead. Outside capture it is sent immediately
        as before, so this is unconditional.
        """
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(self.weight, 1)], global_cb=self.global_cb, capture_into_trace=True
        )
        self.prefetch_queued = True

    def fetch_weights(self):
        if self.use_prefetcher:
            self._queue_prefetch()
            return
        self.l1_weights = ttnn.to_memory_config(self.weight, self.weights_memory_config)
        # self.weight.deallocate()

    def get_input_memory_config(self, m: int, k: int) -> ttnn.MemoryConfig:
        a_core_range_set = ttnn.num_cores_to_corerangeset(
            self.num_inputA_cores, self.device.compute_with_storage_grid_size(), row_wise=True
        )
        a_memory_config = ttnn.create_sharded_memory_config(
            (32, k // self.num_inputA_cores),
            core_grid=a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return a_memory_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
            output_memory_config = regular_width_sharded_l1_config(m_padded, self.N, self.device)
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(x.shape[-2], x.shape[-1]))
        result = ttnn.experimental.matmul_decode(
            x, self.l1_weights, partial_width_sharded=self.partial_width_sharded, output_mem_config=output_memory_config
        )
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
    ):
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.K = K
        self.N = N
        self.num_inputA_cores = num_inputA_cores

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
            # w: [batch, K, N] -> [b_blocks, Bc, K, N] -> [Bc, K, b_blocks, N] -> [1, 1, Bc*K, b_blocks*N].
            w = (
                w.reshape(self.b_blocks, self.bc, K, N)
                .permute(1, 2, 0, 3)
                .reshape(1, 1, self.bc * K, self.b_blocks * N)
                .contiguous()
            )
        self.weight = _load_weight(w, device, cache_file_name=cache_file_name, dtype=dtype)

    def deallocate(self):
        pass

    def get_input_memory_config(self, m: int) -> ttnn.MemoryConfig:
        # Activation A is width(K)-sharded: shard [batch * m_padded, K / num_inputA_cores].
        m_padded = ((m + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
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
        if not x.is_sharded():
            x = ttnn.to_memory_config(x, self.get_input_memory_config(m))
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
                x = ttnn.to_memory_config(x, width_sharded_l1_config(rows, d, x.device()))
        # Keep the output in the same sharded layout the norm just
        # consumed, so the next op (LinearDecode / _apply_rope) doesn't round-trip
        # through DRAM-interleaved. ``ttnn.rms_norm`` requires a sharded output to
        # match the input's memory layout, so passing the input's own config is the
        # documented contract; for DRAM-interleaved inputs this is the default anyway.
        assert x.is_sharded(), "input must be sharded"
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps, memory_config=x.memory_config())


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
