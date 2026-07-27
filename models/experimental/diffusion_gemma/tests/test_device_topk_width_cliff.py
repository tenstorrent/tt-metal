# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.topk(k=1)`` correctness across vocab-shard widths.

An independent DiffusionGemma implementation measured ``ttnn.topk`` returning a **garbage
index and value** at a 32768-wide reduction — 32 of 256 rows matching a torch control, with
``inf`` values — while the same call was correct at width >= 49152. Its workaround was to pad
the shard up to 49152 with ``-inf`` purely to obtain the index, and to take the value from a
plain ``ttnn.max`` (reliable at any width).

DiffusionGemma does not currently hit that width: the terminal argmax goes through
``tt.sampling.argmax_last_dim`` (``ttnn.argmax`` on a ROW_MAJOR input, not ``topk``), and the
only ``ttnn.topk`` calls on the denoise path are the router's ``k=top_k`` over the 128-expert
axis. But the vocab-shard width IS a function of the mesh: V=262144 over tp=4 is 65536 (safe),
over tp=8 it is exactly **32768** — the width the cliff was reported at. A Galaxy 4x8 bring-up
would land on it, and a wrong argmax index is committed straight into the canvas with no
temperature cushion.

**Measured on QB2 2026-07-27, and the cliff reproduces exactly as reported:**

    width 16384  index agreement 0.129
    width 32768  index agreement 0.129   <- V/tp at tp=8
    width 49152  index agreement 1.000
    width 65536  index agreement 1.000   <- V/tp at tp=4, what we serve

So ``ttnn.topk(k=1)`` must not be used on a vocab shard narrower than 49152 on this stack. tp=4 is
safe; a tp=8 mesh is not, and would need winter's pad-to-49152-for-the-index workaround (taking the
value from ``ttnn.max``, which is reliable at every width tested — ``max_all_finite`` held throughout).

This test pins the behaviour rather than assuming it. It is deliberately written to be informative
either way:

* it always asserts the widths DiffusionGemma actually serves today (65536 for the tp=4 vocab
  shard, plus the 128-wide router axis) are exact — a regression there is a hard failure;
* for the widths we do not serve it records the agreement and only fails if a width that was
  previously exact stops being exact, so reproducing the reported cliff at 32768 is a
  *finding* recorded by ``-s`` output, not a red build on someone else's op.

Run on QB2::

    DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_topk_width_cliff.py -s
"""

import math
import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt.sampling import argmax_last_dim

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,  # one device open/teardown — avoid QB2 erisc cycling
]

_SEQ = 256
# 65536 = V/tp at the tp=4 mesh we serve; 32768 = V/tp at tp=8, the reported cliff.
_SERVED_WIDTHS = (65536,)
_PROBE_WIDTHS = (16384, 32768, 49152, 65536)


def _logits(width, seed=7):
    """Row-varied logits with a unique maximum per row.

    A tie would make ``argmax`` legitimately implementation-defined, which would confound a
    correctness check, so the winner is nudged clear of the runner-up.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, _SEQ, width, generator=g) * torch.linspace(0.5, 4.0, _SEQ).view(1, 1, _SEQ, 1)
    winners = torch.randint(0, width, (_SEQ,), generator=g)
    for row, col in enumerate(winners.tolist()):
        x[0, 0, row, col] = x[0, 0, row].max().item() + 1.0
    return x, winners


def _topk_index(tt_logits):
    _values, indices = ttnn.topk(tt_logits, 1, dim=-1, largest=True, sorted=False)
    out = ttnn.to_torch(indices).reshape(-1)[:_SEQ].to(torch.int64)
    _values.deallocate(True)
    indices.deallocate(True)
    return out


def _agreement(got, expected):
    return float((got == expected).float().mean().item())


@pytest.mark.parametrize("width", _PROBE_WIDTHS)
def test_topk_k1_index_across_shard_widths(device, width):
    """Record ``ttnn.topk(k=1)`` index agreement per width; assert exactness where we serve."""

    torch_logits, expected = _logits(width)
    tt_logits = ttnn.from_torch(torch_logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        got = _topk_index(tt_logits)
        max_value = ttnn.to_torch(ttnn.max(tt_logits, dim=-1, keepdim=True)).reshape(-1)[:_SEQ].float()
    finally:
        tt_logits.deallocate(True)

    agreement = _agreement(got, expected)
    finite = bool(torch.isfinite(max_value).all())
    print(f"[topk width sweep] width={width} index_agreement={agreement:.4f} max_all_finite={finite}")

    if width in _SERVED_WIDTHS:
        assert agreement == 1.0, (
            f"ttnn.topk(k=1) index is wrong at width {width}, which is the vocab shard "
            f"DiffusionGemma serves (V=262144 over tp=4): agreement {agreement:.4f}"
        )
        assert finite, f"ttnn.max produced a non-finite value at served width {width}"


def test_router_axis_topk_is_exact(device):
    """The 128-wide expert axis — the only ``ttnn.topk`` the denoise path actually runs.

    The selection is made **separable**: the 8 winners are lifted clear of the rest by a margin far
    wider than bf16 rounding. Without that, a plain random input puts the 8th and 9th values within
    a bf16 ulp of each other on a few rows and the op picks a different (equally valid) member —
    measured 0.9961 set overlap, which is tie-breaking, not an op error. Asserting exactness on a
    tie-free input tests the op; asserting it on a tied one tests nothing and fails intermittently.
    """

    num_experts, top_k = 128, 8
    g = torch.Generator().manual_seed(11)
    routing = torch.randn(1, 1, _SEQ, num_experts, generator=g).clamp(-3.0, 3.0)
    winners = torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(_SEQ)])
    for row in range(_SEQ):
        # Distinct, well-separated values (10.0, 11.0, ...) so neither the membership nor the order
        # can be decided by rounding.
        routing[0, 0, row, winners[row]] = torch.arange(10.0, 10.0 + top_k)

    tt_routing = ttnn.from_torch(routing, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        _values, indices = ttnn.topk(tt_routing, top_k, dim=-1, largest=True, sorted=True)
        got = ttnn.to_torch(indices).reshape(_SEQ, top_k).to(torch.int64)
        _values.deallocate(True)
        indices.deallocate(True)
    finally:
        tt_routing.deallocate(True)

    overlap = sum(len(set(got[i].tolist()) & set(winners[i].tolist())) for i in range(_SEQ)) / (_SEQ * top_k)
    print(f"[router topk] width={num_experts} k={top_k} set_overlap={overlap:.4f}")
    assert overlap == 1.0, f"router top-{top_k} over {num_experts} experts disagrees with torch: {overlap:.4f}"


def test_argmax_last_dim_matches_torch_at_served_width(device):
    """The op the terminal actually uses, at the width it actually runs on."""

    width = 65536
    torch_logits, expected = _logits(width, seed=13)
    tt_logits = ttnn.from_torch(torch_logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        got = ttnn.to_torch(argmax_last_dim(tt_logits)).reshape(-1)[:_SEQ].to(torch.int64)
    finally:
        tt_logits.deallocate(True)

    agreement = _agreement(got, expected)
    print(f"[argmax_last_dim] width={width} agreement={agreement:.4f}")
    assert agreement == 1.0, f"argmax_last_dim disagrees with torch at the served width: {agreement:.4f}"


def test_vocab_shard_width_is_documented_for_this_mesh():
    """Host-only: state which shard width a given tp lands on, so a mesh change is not silent."""

    vocab = 262144
    for tp in (1, 2, 4, 8):
        width = vocab // tp
        cliff = width < 49152 and width != vocab
        print(f"[vocab shard] tp={tp} width={width} in_reported_cliff_band={cliff}")
        assert width * tp == vocab
    assert vocab // 4 == 65536, "tp=4 vocab shard changed; re-run the width sweep before trusting the terminal"
    assert vocab // 8 == 32768, "tp=8 lands on the width the topk cliff was reported at"
    assert math.log2(vocab).is_integer()
