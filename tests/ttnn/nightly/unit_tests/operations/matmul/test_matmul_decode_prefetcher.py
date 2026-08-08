# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for matmul_decode fed by the DRISC tensor prefetcher.

The weight lives in DRAM as an ND-sharded (receiver-contiguous) tensor -- one
slab per B core -- and the prefetcher pushes each slab into the matmul's in1
circular buffer through a DRAM-sender GlobalCircularBuffer.

All three program factories are covered. They differ only in the per-receiver
slab and the order the slabs must be delivered in:

    full     N_blocks receivers,          slab [K, N/N_blocks], idx -> n_idx
    partial  K_blocks*N_blocks receivers, slab [Kc, Nc],        idx -> (idx // N_blocks, idx % N_blocks)
    batched  b_blocks*n_blocks receivers, slab [Bc*K, Nc],      idx -> (idx // n_blocks, idx % n_blocks)

N is the fast-varying dimension in the two two-dimensional modes. A mismatched
order is silently wrong rather than an error, which is why each mode also has a
wrong-order guard asserting that the permuted layout does *not* match.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bank_receivers_strided,
    make_recv_contig_weight,
    tensor_prefetcher_session,
)
from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul_decode import get_tile_height


def _num_cores_to_rectangle_core_range_set(num_cores, device):
    """A single rectangular ``CoreRangeSet`` of exactly ``num_cores`` cores.

    Mirrors ``num_cores_to_rectangle_core_range_set`` in ``test_matmul_decode.py``:
    finds the widest ``x`` dividing ``num_cores`` that fits the device grid, giving
    an ``(x, num_cores // x)`` rectangle. A single-row ``CoreRangeSet`` (as wide as
    ``num_cores``) assumes the device grid is at least ``num_cores`` cores wide,
    which doesn't hold on every device, so we wrap into a rectangle instead.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


def test_matmul_decode_accepts_global_cb_kwarg(device):
    """The global_cb keyword exists and defaults to None (no behavior change)."""
    m, k, n = 32, 1024, 2048
    num_a_cores = 32
    num_b_cores = n // 64

    torch.manual_seed(0)
    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    a_grid = _num_cores_to_rectangle_core_range_set(num_a_cores, device)
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    b = ttnn.from_torch(
        pt_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (k, n // num_b_cores),
            core_grid=b_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    out = ttnn.experimental.matmul_decode(a, b, global_cb=None)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)


def _weight_tile_bytes(weight):
    """Bytes of one 32x32 tile of ``weight``, from its own tile and dtype.

    Block-float dtypes pack a tile smaller than the bfloat16 2048 (1088 for bfloat8_b,
    576 for bfloat4_b), so a hardcoded 2048 oversizes the GCB for them -- harmless for
    the passing tests but it makes the undersized-GCB test stop being undersized.
    """
    return weight.tile.get_tile_size(weight.dtype)


def _make_gcb_and_operands(
    device, m, k, n, num_a_cores, num_slabs=2, build_gcb=True, seed=0, gcb_k_blocks=1, num_pages=None
):
    """Build the activation, the DRAM receiver-contiguous weight, and the GCB.

    The B/receiver grid is the rectangle `_num_cores_to_rectangle_core_range_set`
    picks for `num_b_cores`, anchored at (0, 0). `bank_receivers_strided` maps ring
    position p to core `(p % ring_cols, p // ring_cols)`, so passing that rectangle's
    WIDTH as `ring_cols` makes ring position equal the core's row-major index -- which
    is the order matmul_decode assigns N-columns to B cores, and the order the weight's
    ND shards are laid out in. Passing `num_b_cores` instead is only correct when the
    rectangle happens to be a single row, and silently produces wrong results otherwise.

    When `build_gcb` is False, the GCB is not constructed here (skipping the inline
    size arithmetic entirely); the 5th element of the returned tuple is instead the
    `bank_to_receivers` list, for callers that want to build the GCB themselves (e.g.
    via `make_matmul_decode_gcb`) without duplicating this function's setup.

    `seed` seeds the torch RNG, so operands are reproducible per call. Callers that
    need two *different* operand sets in one test (e.g. to tell aliasing apart from
    correctness) must pass distinct seeds -- two calls with the same seed produce
    bit-identical tensors.

    `gcb_k_blocks` is how many pages a slab is cut into, and `num_pages` sizes the ring in
    those pages instead of in slabs -- together they build a GCB *smaller* than a slab,
    which is the point of streaming. The caller must pass the same `gcb_k_blocks` to the
    prefetch request's block_count and to matmul_decode's `global_cb_k_blocks`.
    """
    torch.manual_seed(seed)
    num_dram_banks = device.dram_grid_size().x
    num_b_cores = n // 64
    assert num_b_cores % num_dram_banks == 0, f"{num_b_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_b_cores // num_dram_banks

    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)

    a_grid = _num_cores_to_rectangle_core_range_set(num_a_cores, device)
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    # Rectangle width == the row-major stride, hence ring_cols.
    ring_cols = b_grid.bounding_box().grid_size().x
    # M < 32 needs a short activation tile, as in test_matmul_decode.py: the shard height
    # must be tile-aligned, and a 32-high tile would reject an m=1 or m=8 shard outright.
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((get_tile_height(m), 32)),
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    weight = make_recv_contig_weight(
        device,
        pt_b.reshape(1, 1, k, n),
        num_dram_banks=num_dram_banks,
        ring_size=num_b_cores,
        dtype=ttnn.bfloat4_b if k > 4096 else ttnn.bfloat16,
        distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )

    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
        for b in range(num_dram_banks)
    ]
    if not build_gcb:
        return pt_a, pt_b, a, weight, bank_to_receivers, num_b_cores

    # A GCB page is a `gcb_k_blocks`-th of a receiver's [K, N/num_b_cores] slab (the whole
    # slab at the default of 1).
    slab_bytes = (k // 32) * ((n // num_b_cores) // 32) * _weight_tile_bytes(weight)
    page_bytes = slab_bytes // gcb_k_blocks
    gcb_size = num_pages * page_bytes if num_pages is not None else num_slabs * slab_bytes
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, gcb_size)
    return pt_a, pt_b, a, weight, gcb, num_b_cores


def test_matmul_decode_gcb_output_spec(device):
    """With a GCB, the output grid/shard come from the GCB receivers, not from a
    legacy shard spec on the DRAM weight (which has none).

    The prefetcher must run even though this test only checks the output spec: once
    the in1 CB is GCB-backed the reader blocks until its page arrives, and a matmul
    launched without a prefetch request would wedge the device for every later test.
    """
    m, k, n = 32, 1024, 2048
    _, _, a, weight, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        ttnn.synchronize_device(device)

    assert tuple(out.shape) == (m, n)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out.memory_config().shard_spec.shape == [m, n // num_b_cores]


def test_matmul_decode_prefetched_weights_pcc(device):
    """Full end-to-end: prefetcher pushes each receiver's weight slab into the GCB,
    matmul_decode consumes it, result matches torch."""
    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


@pytest.mark.parametrize(
    "gcb_k_blocks, num_pages",
    [
        pytest.param(4, 2, id="k_blocks=4-gcb=half-a-slab"),
        pytest.param(32, 2, id="k_blocks=32-gcb=one-sixteenth-of-a-slab"),
        pytest.param(2, 6, id="k_blocks=2-gcb=three-slabs"),
    ],
)
def test_matmul_decode_prefetched_streamed_slab_pcc(device, gcb_k_blocks, num_pages):
    """Full width-sharded with the slab streamed in as several GCB pages.

    The GCB is deliberately smaller than a weight slab in the first two cases, which is the
    whole point of streaming: the ring is sized in pages, so the prefetcher can run ahead into
    the next weight instead of stalling at slab granularity. Each output tile is now finished
    across several pages with the packer accumulating into the output CB, so a dropped or
    double-counted partial sum shows up as a PCC failure here.

    The last case keeps the ring larger than a slab, so the sender is never the one throttled
    and the reader's credit return is exercised with pages already waiting.
    """
    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(
        device, m, k, n, num_a_cores=32, gcb_k_blocks=gcb_k_blocks, num_pages=num_pages
    )
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, gcb_k_blocks)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb, global_cb_k_blocks=gcb_k_blocks)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


def test_matmul_decode_streamed_repeated_invocations(device):
    """Back-to-back streamed prefetch+matmul pairs, alternating two distinct weights.

    A ring smaller than a slab wraps *within* one invocation, so the read and write pointers
    have to stay in step across invocations too. The two weights differ, so a receiver that
    left its pointer misaligned reads the wrong page and fails PCC instead of quietly
    re-reading identical data.
    """
    m, k, n = 32, 1024, 2048
    gcb_k_blocks = 4
    pt_a0, pt_b0, a0, w0, gcb, _ = _make_gcb_and_operands(
        device, m, k, n, num_a_cores=32, seed=0, gcb_k_blocks=gcb_k_blocks, num_pages=2
    )
    pt_a1, pt_b1, a1, w1, _, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, seed=1, build_gcb=False)
    assert not torch.equal(pt_b0, pt_b1), "the two weights must differ or a stale page read would go unnoticed"

    operands = [
        (a0, w0, pt_a0.to(torch.float32) @ pt_b0.to(torch.float32)),
        (a1, w1, pt_a1.to(torch.float32) @ pt_b1.to(torch.float32)),
    ]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            act, w, ref = operands[i % 2]
            ttnn.experimental.queue_tensor_prefetcher_request(device, [(w, gcb_k_blocks)], global_cb=gcb)
            out = ttnn.experimental.matmul_decode(act, w, global_cb=gcb, global_cb_k_blocks=gcb_k_blocks)
            assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)


def test_prefetch_and_matmul_decode_helper(device):
    """The paired helper issues the prefetch and the matmul against the same GCB."""
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


def test_make_matmul_decode_gcb_helper(device):
    """`make_matmul_decode_gcb` sizes a working GCB from the weight and receivers alone,
    with no inline size arithmetic at the call site."""
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb, prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, bank_to_receivers, _ = _make_gcb_and_operands(
        device, m, k, n, num_a_cores=32, build_gcb=False
    )
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    gcb = make_matmul_decode_gcb(device, weight, bank_to_receivers)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 1024, 2048),
        (8, 1024, 2048),
        (32, 1024, 2048),
        (32, 2048, 2048),
        (32, 1024, 4096),
        # Fewer receivers (16) than A cores (32), so the reader's core set is a strict
        # superset of the GCB receiver set and the non-receiver cores take the
        # `is_in1_receiver == 0` branch -- the one that must skip the remote-CB wait and
        # the sync handshake entirely. Every other shape here has receivers >= A cores,
        # which never exercises it; getting it wrong hangs those cores rather than
        # failing, so this case is the only guard against that.
        (32, 1024, 1024),
        (32, 4096, 512),
        (32, 4096, 1024),
        (32, 8192, 4096),
    ],
)
def test_matmul_decode_prefetched_shapes(device, m, k, n):
    """PCC across the decode M range and a couple of K/N shapes."""
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


def test_matmul_decode_prefetched_repeated_invocations(device):
    """Back-to-back prefetch+matmul pairs against one GCB, alternating two weights.

    Each pair consumes exactly one page per receiver, so the GCB read and write
    pointers must stay in lockstep across invocations. Drift shows up either as a
    hang or as iteration N returning iteration N-1's data.

    The two operand sets must differ. The default GCB is two slabs deep, so a
    receiver that failed to advance its read pointer would re-read the previous
    slab -- and with a single repeated weight that slab holds bit-identical data,
    so PCC would pass and the test would prove nothing. Alternating distinct
    weights and checking each result against its own reference is what makes a
    stale read observable.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a0, pt_b0, a0, w0, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, seed=0)
    pt_a1, pt_b1, a1, w1, _, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, seed=1, build_gcb=False)
    assert not torch.equal(pt_b0, pt_b1), "the two weights must differ or a stale slab read would go unnoticed"

    operands = [
        (a0, w0, pt_a0.to(torch.float32) @ pt_b0.to(torch.float32)),
        (a1, w1, pt_a1.to(torch.float32) @ pt_b1.to(torch.float32)),
    ]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            a, weight, ref = operands[i % 2]
            out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
            result = ttnn.to_torch(out).float()
            assert_with_pcc(ref, result, 0.99)


def _rectangle(width, height):
    """A ``width`` x ``height`` CoreRangeSet anchored at (0, 0)."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))})


def _dram_nd_sharded(device, pt, slab_shape, num_dram_banks):
    """Rank-4 `pt` in DRAM, ND-sharded into `slab_shape` slabs enumerated row-major.

    ROUND_ROBIN_1D puts shard s on bank s % num_dram_banks, which pairs with
    `bank_receivers_strided` to make shard index equal ring position.
    """
    return ttnn.as_tensor(
        pt,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                ttnn.Shape(list(slab_shape)),
                _rectangle(num_dram_banks, 1),
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        ),
    )


def _make_partial_gcb_and_operands(
    device,
    m,
    k,
    n,
    k_blocks,
    n_blocks,
    *,
    num_a_cores=32,
    build_gcb=True,
    seed=0,
    swap_slab_order=False,
    gcb_k_blocks=1,
    num_pages=2,
):
    """Operands, DRAM weight and GCB for the partial width-sharded mode.

    A [Kc, Nc] ND shard of the plain [K, N] weight enumerates slabs row-major over the
    (K_blocks x N_blocks) block grid, which is exactly the order the factory assigns blocks to
    receivers (idx -> k_idx = idx // N_blocks, n_idx = idx % N_blocks). Unlike the
    non-prefetcher partial path the weight is not K-block-folded. The receiver rectangle is
    n_blocks wide and k_blocks tall so a core's row-major index within it is that same
    k_idx * n_blocks + n_idx. No output memory config is needed: on the GCB path
    `compute_output_specs` shards the output over the first n_blocks receivers, which are
    exactly the base cores the K-partials reduce onto.

    `swap_slab_order` transposes the enumeration to K-fast-varying, which is the ordering
    mistake a caller is most likely to make; the wrong-order guard uses it.

    `seed` seeds the torch RNG, so operands are reproducible per call and callers needing two
    *different* operand sets in one test must pass distinct seeds.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb

    torch.manual_seed(seed)
    kc, nc = k // k_blocks, n // n_blocks
    num_b_cores = k_blocks * n_blocks
    num_dram_banks = device.dram_grid_size().x
    assert num_b_cores % num_dram_banks == 0, f"{num_b_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_b_cores // num_dram_banks

    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((get_tile_height(m), 32)),
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=_num_cores_to_rectangle_core_range_set(num_a_cores, device),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    pt_weight = pt_b
    if swap_slab_order:
        blocks = pt_b.reshape(k_blocks, kc, n_blocks, nc).permute(0, 2, 1, 3)
        pt_weight = blocks.permute(1, 0, 2, 3).reshape(k_blocks, n_blocks, kc, nc).permute(0, 2, 1, 3).reshape(k, n)
    weight = _dram_nd_sharded(device, pt_weight.reshape(1, 1, k, n), (kc, nc), num_dram_banks)

    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=n_blocks)) for b in range(num_dram_banks)
    ]

    if not build_gcb:
        return pt_a, pt_b, a, weight, bank_to_receivers
    gcb = make_matmul_decode_gcb(
        device, weight, bank_to_receivers, slab_shape=(kc, nc), k_blocks=gcb_k_blocks, num_pages=num_pages
    )
    return pt_a, pt_b, a, weight, gcb


@pytest.mark.parametrize(
    "m, k, n, k_blocks, n_blocks",
    [
        (32, 1024, 1024, 2, 8),
        (8, 1024, 1024, 2, 8),
    ],
)
def test_matmul_decode_partial_prefetched_weights_pcc(device, m, k, n, k_blocks, n_blocks):
    """Partial width-sharded end-to-end: the weight is cut into [Kc, Nc] slabs, one per GCB
    receiver, and the K-partials reduce onto the base cores.

    k_blocks and n_blocks are both > 1 so the receiver-order contract (N fast-varying) is
    genuinely exercised: with either block count equal to 1 the two possible orders coincide
    and a wrong ordering would still produce the right answer.

    There are also fewer receivers (16) than A cores (32), so the reader's core set is a
    strict superset of the GCB receiver set and the extra cores take the
    `is_in1_receiver == 0` branch -- the one that must skip the remote-CB wait and the sync
    handshake entirely. Getting that wrong hangs those cores rather than failing.
    """
    pt_a, pt_b, a, weight, gcb = _make_partial_gcb_and_operands(device, m, k, n, k_blocks, n_blocks)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, partial_width_sharded=True, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


@pytest.mark.parametrize(
    "gcb_k_blocks, num_pages",
    [
        pytest.param(4, 2, id="k_blocks=4-gcb=half-a-slab"),
        pytest.param(16, 3, id="k_blocks=16-gcb=three-sixteenths-of-a-slab"),
    ],
)
def test_matmul_decode_partial_prefetched_streamed_slab_pcc(device, gcb_k_blocks, num_pages):
    """Partial width-sharded with each [Kc, Nc] slab streamed in as several GCB pages.

    Here the streamed partial sums land in the partial CB that the cross-core K-reduction then
    consumes, so this covers the one mode where the packer accumulates into something other
    than the final output.
    """
    m, k, n, k_blocks, n_blocks = 32, 1024, 1024, 2, 8
    pt_a, pt_b, a, weight, gcb = _make_partial_gcb_and_operands(
        device, m, k, n, k_blocks, n_blocks, gcb_k_blocks=gcb_k_blocks, num_pages=num_pages
    )
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, gcb_k_blocks)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(
            a, weight, partial_width_sharded=True, global_cb=gcb, global_cb_k_blocks=gcb_k_blocks
        )
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


def test_matmul_decode_partial_prefetched_repeated_invocations(device):
    """Back-to-back prefetch+matmul pairs against one partial-mode GCB, alternating two weights.

    The two operand sets must differ. The GCB is two slabs deep, so a receiver that failed to
    advance its read pointer would re-read the previous slab -- and with a single repeated
    weight that slab holds bit-identical data, so PCC would pass and the test would prove
    nothing. Alternating distinct weights and checking each result against its own reference is
    what makes a stale read observable.
    """
    m, k, n, k_blocks, n_blocks = 32, 1024, 1024, 2, 8
    pt_a0, pt_b0, a0, w0, gcb = _make_partial_gcb_and_operands(device, m, k, n, k_blocks, n_blocks, seed=0)
    pt_a1, pt_b1, a1, w1, _ = _make_partial_gcb_and_operands(
        device, m, k, n, k_blocks, n_blocks, seed=1, build_gcb=False
    )
    assert not torch.equal(pt_b0, pt_b1), "the two weights must differ or a stale slab read would go unnoticed"

    operands = [
        (a0, w0, pt_a0.to(torch.float32) @ pt_b0.to(torch.float32)),
        (a1, w1, pt_a1.to(torch.float32) @ pt_b1.to(torch.float32)),
    ]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            a, weight, ref = operands[i % 2]
            ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
            out = ttnn.experimental.matmul_decode(a, weight, partial_width_sharded=True, global_cb=gcb)
            assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)


def test_matmul_decode_partial_prefetched_wrong_slab_order_is_wrong(device):
    """A K-fast-varying slab order must not accidentally produce the right answer.

    Nothing on the device checks the receiver order, so every other partial-mode test here
    would still pass if the factory happened to consume slabs in the transposed order. This is
    the only test that pins the contract down: it hands the op the same weight laid out
    K-fast instead of N-fast and requires the result to collapse.
    """
    m, k, n, k_blocks, n_blocks = 32, 1024, 1024, 2, 8
    pt_a, pt_b, a, weight, gcb = _make_partial_gcb_and_operands(
        device, m, k, n, k_blocks, n_blocks, swap_slab_order=True
    )
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, partial_width_sharded=True, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    passed, message = check_with_pcc_without_tensor_printout(ref, result, 0.99)
    assert not passed, f"a transposed slab order still matched the reference: {message}"


def _make_batched_gcb_and_operands(
    device,
    d0,
    d1,
    m,
    k,
    n,
    b_blocks,
    n_blocks,
    *,
    num_a_cores=16,
    build_gcb=True,
    seed=0,
    swap_slab_order=False,
    gcb_k_blocks=1,
    num_pages=2,
):
    """Operands, DRAM weight and GCB for the batched width-sharded mode.

    The weight fold is the one the non-prefetcher batched test uses:
    [batch, K, N] -> reshape(b_blocks, Bc, K, N) -> permute(1, 2, 0, 3) -> [1, 1, Bc*K, b_blocks*N].
    Its column blocks are already indexed b_idx * n_blocks + n_idx, so ND-sharding it into
    [Bc*K, Nc] slabs yields the required receiver order for free -- and the tile order the
    compute kernel assumes within a slab ([Bc, K_tiles, Nc_tiles]) falls out of the same fold.
    The receiver rectangle is n_blocks wide and b_blocks tall so a core's row-major index within
    it matches that slab index. The output is DRAM-interleaved, so no output_mem_config.

    `swap_slab_order` transposes the column-block enumeration to batch-fast-varying, for the
    wrong-order guard. `seed` seeds the torch RNG; distinct seeds give distinct operands.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb

    torch.manual_seed(seed)
    batch = d0 * d1
    bc, nc = batch // b_blocks, n // n_blocks
    num_b_cores = b_blocks * n_blocks
    num_dram_banks = device.dram_grid_size().x
    assert num_b_cores % num_dram_banks == 0, f"{num_b_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_b_cores // num_dram_banks

    pt_a = torch.randn((batch, m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((batch, k, n), dtype=torch.bfloat16)

    tile_height = get_tile_height(m)
    m_padded = ((m + tile_height - 1) // tile_height) * tile_height
    a = ttnn.from_torch(
        pt_a.reshape(d0, d1, m, k),
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((tile_height, 32)),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.create_sharded_memory_config(
            (batch * m_padded, k // num_a_cores),
            core_grid=_num_cores_to_rectangle_core_range_set(num_a_cores, device),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    folded = pt_b.reshape(b_blocks, bc, k, n).permute(1, 2, 0, 3).reshape(1, 1, bc * k, b_blocks * n)
    if swap_slab_order:
        folded = (
            folded.reshape(1, 1, bc * k, b_blocks, n_blocks, nc)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(1, 1, bc * k, b_blocks * n)
        )
    weight = _dram_nd_sharded(device, folded, (bc * k, nc), num_dram_banks)

    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=n_blocks)) for b in range(num_dram_banks)
    ]
    if not build_gcb:
        return pt_a, pt_b, a, weight, bank_to_receivers
    gcb = make_matmul_decode_gcb(
        device, weight, bank_to_receivers, slab_shape=(bc * k, nc), k_blocks=gcb_k_blocks, num_pages=num_pages
    )
    return pt_a, pt_b, a, weight, gcb


def _run_batched_prefetched(device, a, weight, gcb):
    ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
    out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
    return ttnn.to_torch(out).float()


@pytest.mark.parametrize(
    "d0, d1, m, k, n, b_blocks, n_blocks",
    [
        (1, 4, 32, 512, 1024, 2, 8),
    ],
)
def test_matmul_decode_batched_prefetched_weights_pcc(device, d0, d1, m, k, n, b_blocks, n_blocks):
    """Batched width-sharded end-to-end: each receiver owns one [Bc*K, Nc] slab and produces its
    own (batch-block, N-block) output block, with no cross-core reduction.

    b_blocks and n_blocks are both > 1 so the receiver-order contract (N fast-varying) is
    genuinely exercised.
    """
    batch = d0 * d1
    pt_a, pt_b, a, weight, gcb = _make_batched_gcb_and_operands(device, d0, d1, m, k, n, b_blocks, n_blocks)
    ref = torch.matmul(pt_a.to(torch.float32), pt_b.to(torch.float32))

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        result = _run_batched_prefetched(device, a, weight, gcb)

    assert_with_pcc(ref, result.reshape(batch, m, n), 0.99)


@pytest.mark.parametrize(
    "gcb_k_blocks, num_pages",
    [
        pytest.param(2, 2, id="k_blocks=2-page=one-whole-batch"),
        pytest.param(8, 2, id="k_blocks=8-page=quarter-of-a-batch"),
    ],
)
def test_matmul_decode_batched_prefetched_streamed_slab_pcc(device, gcb_k_blocks, num_pages):
    """Batched width-sharded with each [Bc*K, Nc] slab streamed in as several GCB pages.

    A batched slab is Bc batches stacked along its rows, so where a page boundary falls
    relative to a batch boundary decides whether the compute kernel accumulates or starts a
    fresh output tile. The two cases straddle that: at k_blocks=2 a page is exactly one batch
    (never accumulate), at k_blocks=8 a batch spans four pages (accumulate within a batch and
    reset at its end). Confusing the two mixes different batches' partial sums together.
    """
    d0, d1, m, k, n, b_blocks, n_blocks = 1, 4, 32, 512, 1024, 2, 8
    batch = d0 * d1
    pt_a, pt_b, a, weight, gcb = _make_batched_gcb_and_operands(
        device, d0, d1, m, k, n, b_blocks, n_blocks, gcb_k_blocks=gcb_k_blocks, num_pages=num_pages
    )
    ref = torch.matmul(pt_a.to(torch.float32), pt_b.to(torch.float32))

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, gcb_k_blocks)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb, global_cb_k_blocks=gcb_k_blocks)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result.reshape(batch, m, n), 0.99)


def test_matmul_decode_batched_prefetched_repeated_invocations(device):
    """Back-to-back prefetch+matmul pairs against one batched-mode GCB, alternating two weights.

    The two operand sets must differ. The GCB is two slabs deep, so a receiver that failed to
    advance its read pointer would re-read the previous slab -- and with a single repeated
    weight that slab holds bit-identical data, so PCC would pass and the test would prove
    nothing. Alternating distinct weights and checking each result against its own reference is
    what makes a stale read observable.
    """
    d0, d1, m, k, n, b_blocks, n_blocks = 1, 4, 32, 512, 1024, 2, 8
    batch = d0 * d1
    pt_a0, pt_b0, a0, w0, gcb = _make_batched_gcb_and_operands(device, d0, d1, m, k, n, b_blocks, n_blocks, seed=0)
    pt_a1, pt_b1, a1, w1 = _make_batched_gcb_and_operands(
        device, d0, d1, m, k, n, b_blocks, n_blocks, seed=1, build_gcb=False
    )[:4]
    assert not torch.equal(pt_b0, pt_b1), "the two weights must differ or a stale slab read would go unnoticed"

    operands = [
        (a0, w0, torch.matmul(pt_a0.to(torch.float32), pt_b0.to(torch.float32))),
        (a1, w1, torch.matmul(pt_a1.to(torch.float32), pt_b1.to(torch.float32))),
    ]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            a, weight, ref = operands[i % 2]
            result = _run_batched_prefetched(device, a, weight, gcb)
            assert_with_pcc(ref, result.reshape(batch, m, n), 0.99)


def test_matmul_decode_batched_prefetched_wrong_slab_order_is_wrong(device):
    """A batch-fast-varying slab order must not accidentally produce the right answer.

    Nothing on the device checks the receiver order, so every other batched-mode test here
    would still pass if the factory happened to consume slabs in the transposed order. This is
    the only test that pins the contract down.
    """
    d0, d1, m, k, n, b_blocks, n_blocks = 1, 4, 32, 512, 1024, 2, 8
    batch = d0 * d1
    pt_a, pt_b, a, weight, gcb = _make_batched_gcb_and_operands(
        device, d0, d1, m, k, n, b_blocks, n_blocks, swap_slab_order=True
    )
    ref = torch.matmul(pt_a.to(torch.float32), pt_b.to(torch.float32))

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        result = _run_batched_prefetched(device, a, weight, gcb)

    passed, message = check_with_pcc_without_tensor_printout(ref, result.reshape(batch, m, n), 0.99)
    assert not passed, f"a transposed slab order still matched the reference: {message}"


def _l1_width_sharded_weight(device, pt_b, k, n, num_b_cores):
    """`pt_b` as a plain L1 width-sharded weight -- the non-prefetcher in1 layout."""
    return ttnn.from_torch(
        pt_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (k, n // num_b_cores),
            core_grid=_num_cores_to_rectangle_core_range_set(num_b_cores, device),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )


def test_matmul_decode_partial_global_cb_rejects_indivisible_slab(device, expect_error):
    """Partial mode: the [Kc, Nc] slab must tile the [K, N] weight exactly.

    Kc=768 against K=1024 leaves a ragged last K-block. The shard count still comes out at
    ceil(1024/768) * (1024/128) = 2 * 8 = 16, so the generic one-shard-per-receiver check is
    satisfied and only the partial mode's own divisibility check can catch it. It has to: the
    factory derives K_blocks by dividing, so a ragged block silently drops weight rows.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb

    m, k, n, kc, nc = 32, 1024, 1024, 768, 128
    num_dram_banks = device.dram_grid_size().x
    num_b_cores = 16

    torch.manual_seed(0)
    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((get_tile_height(m), 32)),
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // 32),
            core_grid=_num_cores_to_rectangle_core_range_set(32, device),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    weight = _dram_nd_sharded(device, pt_b.reshape(1, 1, k, n), (kc, nc), num_dram_banks)

    bank_to_receivers = [
        (b, bank_receivers_strided(b, num_b_cores // num_dram_banks, num_dram_banks, ring_cols=n // nc))
        for b in range(num_dram_banks)
    ]
    gcb = make_matmul_decode_gcb(device, weight, bank_to_receivers, slab_shape=(kc, nc))

    with expect_error(RuntimeError, "Kc dividing K"):
        ttnn.experimental.matmul_decode(a, weight, partial_width_sharded=True, global_cb=gcb)


def test_make_matmul_decode_gcb_rejects_receiver_count_mismatch(device, expect_error):
    """The GCB builder rejects a receiver set the slab shape does not cut the weight into.

    This is the one host-side check that runs before anything is enqueued; without it a short
    receiver set only shows up as blocks nobody has a receiver for, i.e. a hang.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb

    m, k, n, k_blocks, n_blocks = 32, 1024, 1024, 2, 8
    _, _, _, weight, _ = _make_partial_gcb_and_operands(device, m, k, n, k_blocks, n_blocks, build_gcb=False)

    num_dram_banks = device.dram_grid_size().x
    too_few = [(b, bank_receivers_strided(b, 1, num_dram_banks, ring_cols=n_blocks)) for b in range(num_dram_banks)]

    with expect_error(ValueError, r"into 16 shards, but the GCB has 8 receivers"):
        make_matmul_decode_gcb(device, weight, too_few, slab_shape=(k // k_blocks, n // n_blocks))


def test_matmul_decode_batched_global_cb_rejects_wrong_slab_shape(device, expect_error):
    """Batched mode: each weight shard must be the whole [Bc*K, Nc] slab.

    Halving the shard height still yields one shard per receiver, so only the mode-aware
    geometry check catches it. It has to: the compute kernel indexes in1 across the full
    [Bc, K_tiles, Nc_tiles] slab, so a half-height page leaves it reading past its data.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import make_matmul_decode_gcb

    d0, d1, m, k, n, b_blocks, n_blocks = 1, 4, 32, 512, 1024, 2, 8
    bc, nc = (d0 * d1) // b_blocks, n // n_blocks
    _, _, a, _, _ = _make_batched_gcb_and_operands(device, d0, d1, m, k, n, b_blocks, n_blocks, build_gcb=False)

    torch.manual_seed(0)
    num_dram_banks = device.dram_grid_size().x
    pt_b = torch.randn((d0 * d1, k, n), dtype=torch.bfloat16)
    folded = pt_b.reshape(b_blocks, bc, k, n).permute(1, 2, 0, 3).reshape(1, 1, bc * k, b_blocks * n)
    short_slab_weight = _dram_nd_sharded(device, folded, (bc * k // 2, nc), num_dram_banks)

    recv_per_bank = 2 * b_blocks * n_blocks // num_dram_banks
    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=n_blocks)) for b in range(num_dram_banks)
    ]
    gcb = make_matmul_decode_gcb(device, short_slab_weight, bank_to_receivers, slab_shape=(bc * k // 2, nc))

    with expect_error(RuntimeError, r"each weight shard to be \[Bc\*K, Nc\]"):
        ttnn.experimental.matmul_decode(a, short_slab_weight, global_cb=gcb)


@pytest.mark.parametrize("partial_width_sharded", [False, True])
def test_matmul_decode_global_cb_rejects_l1_weight(device, partial_width_sharded, expect_error):
    """With a global_cb the weight must be the DRAM ND-sharded tensor the prefetcher reads."""
    m, k, n = 32, 1024, 2048
    _, pt_b, a, _, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    l1_weight = _l1_width_sharded_weight(device, pt_b, k, n, num_b_cores)

    with expect_error(RuntimeError, "input tensor B to live in DRAM"):
        ttnn.experimental.matmul_decode(a, l1_weight, partial_width_sharded=partial_width_sharded, global_cb=gcb)


@pytest.mark.parametrize("partial_width_sharded", [False, True])
def test_matmul_decode_global_cb_rejects_weight_without_nd_shard_spec(device, partial_width_sharded, expect_error):
    """A weight with no NdShardSpec at all must be rejected before any spec is computed.

    A legacy-sharded weight still carries a derived NdShardSpec, so it reaches the layout check
    in `validate_on_program_cache_miss` intact. An interleaved one does not -- and
    `compute_output_specs` runs first and dereferences that spec on the partial GCB path, so
    without a precondition ahead of it the caller gets a bare bad_optional_access.
    """
    m, k, n = 32, 1024, 2048
    _, pt_b, a, _, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    interleaved_weight = ttnn.from_torch(
        pt_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with expect_error(RuntimeError, "carries no NdShardSpec"):
        ttnn.experimental.matmul_decode(
            a, interleaved_weight, partial_width_sharded=partial_width_sharded, global_cb=gcb
        )


@pytest.mark.parametrize(
    "global_cb_k_blocks, gcb_pages_of_slab",
    [
        # A page is the whole slab, so half a slab is not even one page.
        pytest.param(1, 0.5, id="k_blocks=1-half-a-page"),
        # A page is a quarter slab and the GCB holds exactly one, but streaming needs two.
        pytest.param(4, 0.25, id="k_blocks=4-one-page"),
    ],
)
def test_matmul_decode_global_cb_rejects_undersized_gcb(device, expect_error, global_cb_k_blocks, gcb_pages_of_slab):
    """A GCB too small for the page count the reader needs is rejected on the host.

    The floor is one page when the slab arrives whole and two once it is streamed, because
    the reader holds the page compute is working on while the next one lands; a one-page ring
    would deadlock waiting for a credit it is itself holding. The check lives in the program
    factory and fires before anything is enqueued, so no prefetcher session is needed.
    """
    m, k, n = 32, 1024, 2048
    _, _, a, weight, bank_to_receivers, num_b_cores = _make_gcb_and_operands(
        device, m, k, n, num_a_cores=32, build_gcb=False
    )

    slab_bytes = (k // 32) * ((n // num_b_cores) // 32) * _weight_tile_bytes(weight)
    small_gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
        device, bank_to_receivers, int(slab_bytes * gcb_pages_of_slab)
    )

    with expect_error(RuntimeError, "needs a GCB of at least"):
        ttnn.experimental.matmul_decode(a, weight, global_cb=small_gcb, global_cb_k_blocks=global_cb_k_blocks)


def test_matmul_decode_rejects_k_blocks_not_dividing_k(device, expect_error):
    """A page has to be a whole number of K-rows of the slab, so k_blocks must divide K in tiles."""
    m, k, n = 32, 1024, 2048  # K is 32 tiles, so 5 pages cannot be cut from it.
    _, _, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    with expect_error(RuntimeError, "divisible"):
        ttnn.experimental.matmul_decode(a, weight, global_cb=gcb, global_cb_k_blocks=5)


def test_matmul_decode_rejects_k_blocks_without_global_cb(device, expect_error):
    """global_cb_k_blocks describes GCB pages, so it is meaningless without a GCB."""
    m, k, n = 32, 1024, 2048
    _, _, a, weight, _, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, build_gcb=False)
    l1_weight = ttnn.to_memory_config(weight, ttnn.L1_MEMORY_CONFIG)

    with expect_error(RuntimeError, "global_cb_k_blocks"):
        ttnn.experimental.matmul_decode(a, l1_weight, global_cb_k_blocks=2)


def test_matmul_decode_two_distinct_global_cbs(device):
    """Two GCBs of identical geometry must not alias through the program cache.

    GlobalCircularBuffer hashes on (core mapping, size, buffer type) and not on its
    address, so a cached program could target the first GCB while the prefetcher fills
    the second. The two operand sets are built from different seeds (asserted below),
    so aliasing shows up as a PCC failure against the second reference (or as a hang)
    rather than silently passing.
    """
    m, k, n = 32, 1024, 2048
    pt_a1, pt_b1, a1, w1, gcb1, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, seed=0)
    pt_a2, pt_b2, a2, w2, gcb2, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32, seed=1)
    # Without this the test could not tell aliasing apart from correctness.
    assert not torch.equal(pt_b1, pt_b2), "the two weight sets must differ for this test to mean anything"

    ref1 = pt_a1.to(torch.float32) @ pt_b1.to(torch.float32)
    ref2 = pt_a2.to(torch.float32) @ pt_b2.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(w1, 1)], global_cb=gcb1)
        r1 = ttnn.to_torch(ttnn.experimental.matmul_decode(a1, w1, global_cb=gcb1)).float()
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(w2, 1)], global_cb=gcb2)
        r2 = ttnn.to_torch(ttnn.experimental.matmul_decode(a2, w2, global_cb=gcb2)).float()

    assert_with_pcc(ref1, r1, 0.99)
    assert_with_pcc(ref2, r2, 0.99)


def test_matmul_decode_prefetched_vs_l1_resident_perf(device):
    """Profiler-visible comparison: prefetched weights vs. today's per-call DRAM->L1 copy.

    Run under the profiler to compare the two signposted regions:
        pytest ... -k prefetched_vs_l1_resident --profiler
    """
    from tracy import signpost
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    # Baseline operands: the same weight as a DRAM-interleaved tensor that must be copied
    # into L1 width-sharded form before every matmul -- what LinearDecode does today.
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    l1_weight_config = ttnn.create_sharded_memory_config(
        (k, n // num_b_cores),
        core_grid=b_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    dram_weight = ttnn.from_torch(pt_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        signpost("matmul_decode_prefetched")
        for _ in range(4):
            prefetched_out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        signpost("matmul_decode_l1_copy")
        for _ in range(4):
            l1_weight = ttnn.to_memory_config(dram_weight, l1_weight_config)
            baseline_out = ttnn.experimental.matmul_decode(a, l1_weight)
            l1_weight.deallocate()
        signpost("stop")

        prefetched = ttnn.to_torch(prefetched_out).float()
        baseline = ttnn.to_torch(baseline_out).float()

    assert_with_pcc(ref, prefetched, 0.99)
    assert_with_pcc(ref, baseline, 0.99)
