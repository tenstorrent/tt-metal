# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``LinearDecode(use_prefetcher=True)``: weights delivered by the DRISC tensor prefetcher.

The layer keeps its weight DRAM ND-sharded -- one slab per B core -- and the prefetcher
pushes each slab into the matmul's in1 circular buffer through a GlobalCircularBuffer,
replacing the DRAM->L1 copy the default path does before every call. These tests pin down
that the swap is invisible at the output: same math as torch, and same result as the
L1-copy path on the same weights.

Both weight layouts are covered, because they place slabs differently: full
width-sharded gives each of ``N // 64`` cores a ``[K, N/cores]`` slab, while partial
width-sharded spreads a ``(k_blocks x n_blocks)`` grid of ``[Kc, Nc]`` slabs and reduces
the K-partials across cores. The full width-sharded case also covers several layers
sharing one GCB, which is how a block avoids paying for a buffer per projection.
"""

import pytest
import torch

import ttnn
from models.experimental.deepseek_v4_flash.tt.layers import LinearDecode, make_shared_decode_gcb
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


def _activation(device, pt_x, num_a_cores=32):
    """``pt_x`` as the L1 width(K)-sharded activation ``LinearDecode`` consumes.

    Handed in already sharded so the test exercises the prefetch path rather than
    ``forward``'s reshard fallback.
    """
    m, k = pt_x.shape
    return ttnn.from_torch(
        pt_x,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=ttnn.num_cores_to_corerangeset(num_a_cores, device.compute_with_storage_grid_size(), True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )


def _reference(pt_x, pt_w):
    # LinearDecode is bias-free ``x @ Wᵀ`` on a torch-style [out, in] weight.
    return pt_x.to(torch.float32) @ pt_w.to(torch.float32).t()


@pytest.mark.parametrize("hoist", [False, True], ids=["lazy", "hoisted"])
def test_linear_decode_prefetcher_shared_global_cb_uniform_slabs(device, hoist):
    """Three *identically shaped* layers sharing one GCB, so every slab is the same size.

    Isolates sharing from the page-size changes its mixed-shape sibling also exercises: one
    GCB, three tensors, one page size throughout, so the sender and receiver never have to
    re-derive the ring geometry between transfers. If this passes and the mixed-shape test
    hangs, the page-size transition is the fault rather than sharing a buffer.

    The three weights must differ, or a receiver that re-read a stale slab would still get
    the right numbers and the test would prove nothing.

    ``hoisted`` queues all three requests before any matmul runs, which is what the model
    does when it prefetches the next layer's weights while the current one computes: with
    the ring only one slab deep the senders must backpressure until each matmul frees
    credits. ``lazy`` alternates a transfer with its matmul and never exercises that.
    """
    m, k, n = 32, 1024, 2048
    num_layers = 3

    specs = [{"K": k, "N": n, "n_blocks": 32} for _ in range(num_layers)]
    global_cb = make_shared_decode_gcb(device, specs, ttnn.bfloat16, num_slabs=1)

    torch.manual_seed(0)
    cases = []
    for spec in specs:
        pt_x = torch.randn((m, k), dtype=torch.bfloat16)
        pt_w = torch.randn((n, k), dtype=torch.bfloat16)
        layer = LinearDecode(pt_w, device, use_prefetcher=True, global_cb=global_cb, **spec)
        cases.append((layer, _activation(device, pt_x), _reference(pt_x, pt_w)))

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        if hoist:
            for layer, _, _ in cases:
                layer.fetch_weights()
        for layer, x, ref in cases:
            assert_with_pcc(ref, ttnn.to_torch(layer(x)).float(), 0.99)


def test_linear_decode_prefetcher_shared_global_cb_rejects_mixed_slab_sizes(device, expect_error):
    """Weights of different slab sizes are refused a shared GCB, in the constructor.

    A GCB page is one slab, so differently sized weights change the ring's page size between
    transfers. Both ends are written to handle that -- each re-derives the page geometry per
    transfer and credits any skipped ring tail to the other over NOC -- but measured against
    this path it hangs: these three shapes (128 KB, 256 KB, 256 KB slabs at bf16) wedge the
    device, while ``..._uniform_slabs`` above passes and so does a single weight repeatedly
    wrapping a two-slab ring. The failing case is the one that leaves the pointer mid-ring
    and then realigns it up to a larger page.

    Rejecting it on the host is what keeps that a diagnosable error rather than a hang, so
    this test guards the check itself: without it the layout below is silently accepted.
    """
    specs = [{"K": k, "N": n, "n_blocks": 32} for k, n in [(1024, 2048), (2048, 2048), (1024, 4096)]]

    with expect_error(ValueError, "same slab size"):
        make_shared_decode_gcb(device, specs, ttnn.bfloat16, num_slabs=1)


@pytest.mark.parametrize(
    "m, k, n, k_blocks, n_blocks",
    [
        (32, 1024, 1024, 2, 8),
        (32, 4096, 1024, 2, 8),
    ],
)
def test_linear_decode_prefetcher_partial_width_sharded(device, m, k, n, k_blocks, n_blocks):
    """Partial width-sharded: a ``(k_blocks x n_blocks)`` grid of ``[Kc, Nc]`` slabs.

    Both block counts are > 1 so the receiver order (N fast-varying) genuinely matters:
    with either equal to 1 the two possible orders coincide and a wrong one would still
    give the right answer.
    """
    torch.manual_seed(0)
    pt_x = torch.randn((m, k), dtype=torch.bfloat16)
    pt_w = torch.randn((n, k), dtype=torch.bfloat16)

    layer = LinearDecode(
        pt_w,
        device,
        K=k,
        N=n,
        partial_width_sharded=True,
        k_blocks=k_blocks,
        n_blocks=n_blocks,
        use_prefetcher=True,
    )
    x = _activation(device, pt_x)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        result = ttnn.to_torch(layer(x)).float()

    assert_with_pcc(_reference(pt_x, pt_w), result, 0.99)


@pytest.mark.parametrize("partial_width_sharded", [False, True])
def test_linear_decode_prefetcher_matches_l1_path(device, partial_width_sharded):
    """The prefetched layer and the default DRAM->L1 layer agree on the same weights.

    Stronger than the PCC-vs-torch tests: it holds the layer's own bf16 arithmetic fixed
    and varies only how the weight is delivered, so any slab-ordering or layout mistake
    shows up as a divergence rather than being absorbed by the 0.99 threshold.
    """
    m, k, n = 32, 1024, 1024
    # Both layers get explicit block counts so they place the weight on the same cores and the
    # only difference left is the delivery mechanism. For full width-sharded that means
    # n // 32 cores rather than the n // 64 default, which is the count the L1 path's output
    # config assumes.
    blocks = {"partial_width_sharded": True, "k_blocks": 2, "n_blocks": 8} if partial_width_sharded else {}
    blocks.setdefault("n_blocks", n // 32)

    torch.manual_seed(0)
    pt_x = torch.randn((m, k), dtype=torch.bfloat16)
    pt_w = torch.randn((n, k), dtype=torch.bfloat16)

    prefetched = LinearDecode(pt_w, device, K=k, N=n, use_prefetcher=True, **blocks)
    baseline = LinearDecode(pt_w, device, K=k, N=n, **blocks)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        prefetched_out = ttnn.to_torch(prefetched(_activation(device, pt_x))).float()
    baseline_out = ttnn.to_torch(baseline(_activation(device, pt_x))).float()

    assert torch.equal(prefetched_out, baseline_out), "prefetched weights changed the layer's output"


def test_linear_decode_prefetcher_repeated_invocations(device):
    """Back-to-back calls against the layer's one GCB, alternating two weights.

    Each call consumes exactly one page per receiver, so the GCB read and write pointers
    must stay in lockstep. The two layers must hold *different* weights: the GCB is two
    slabs deep, so a receiver that failed to advance its read pointer would re-read the
    previous slab, and with one repeated weight that slab holds identical data -- PCC
    would pass and the test would prove nothing.
    """
    m, k, n = 32, 1024, 2048
    torch.manual_seed(0)
    pt_x0, pt_w0 = torch.randn((m, k), dtype=torch.bfloat16), torch.randn((n, k), dtype=torch.bfloat16)
    pt_x1, pt_w1 = torch.randn((m, k), dtype=torch.bfloat16), torch.randn((n, k), dtype=torch.bfloat16)
    assert not torch.equal(pt_w0, pt_w1), "the two weights must differ or a stale slab read would go unnoticed"

    cases = [
        (
            LinearDecode(pt_w0, device, K=k, N=n, use_prefetcher=True),
            _activation(device, pt_x0),
            _reference(pt_x0, pt_w0),
        ),
        (
            LinearDecode(pt_w1, device, K=k, N=n, use_prefetcher=True),
            _activation(device, pt_x1),
            _reference(pt_x1, pt_w1),
        ),
    ]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            layer, x, ref = cases[i % 2]
            assert_with_pcc(ref, ttnn.to_torch(layer(x)).float(), 0.99)


def test_linear_decode_prefetcher_rejects_slabs_not_divisible_across_banks(device, expect_error):
    """A weight whose slab count does not divide across the DRAM banks is rejected in the
    constructor.

    ROUND_ROBIN_1D placement only makes slab index equal ring position when every bank
    holds the same number of slabs. Left unchecked, the uneven bank would be paired with
    the wrong receivers -- a hang or wrong results at the first call, far from the cause.
    """
    num_dram_banks = device.dram_grid_size().x
    # n // 64 slabs; pick an n whose slab count is one short of a whole number of banks.
    num_slabs = num_dram_banks * 2 - 1
    k, n = 1024, num_slabs * 64
    pt_w = torch.zeros((n, k), dtype=torch.bfloat16)

    with expect_error(ValueError, "DRAM banks"):
        LinearDecode(pt_w, device, K=k, N=n, use_prefetcher=True)
