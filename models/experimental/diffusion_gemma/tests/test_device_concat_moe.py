# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence of the concat-experts MoE (``DG_MOE_CONCAT``) against a torch oracle.

The concat path (``tt/concat_moe.py``) folds the routing weights into the GeGLU output so the down
projection is one wide matmul:

    out = (geglu(x @ gate_cat, x @ up_cat) * (routing @ expand)) @ down_cat

instead of the per-expert form the reference and the shipped token-gather path compute:

    out = sum_e routing_e * (geglu(x @ W_gate_e, x @ W_up_e) @ W_down_e)

Those are equal by linearity of the down projection, but "equal on paper" is not the claim that
matters — the device path applies the routing weight to a bf16 GeGLU output *before* a single
24576-long reduction, where the per-expert form reduces 192 at a time and applies the routing weight
afterwards. This test measures what that costs, on device, against a torch oracle.

It also pins the two contracts the fold silently depends on:

* **the routing tensor must be exactly zero for unselected experts** — the fold has no other way to
  exclude them, so a router that returned a full softmax would leak all 128 experts into the output;
* **the padded intermediate columns must contribute zero** — ``weights.py`` pads ``I/tp`` up to a
  tile, and the concat matmul computes over the pad, so a nonzero pad would be summed in by
  ``down_cat``.

Run on QB2::

    DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_concat_moe.py -s
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.tt import concat_moe

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,  # one device open/teardown — avoid QB2 erisc cycling
]

# Small but structurally faithful: E a multiple of 32, I and H tile-aligned, S = a real canvas row
# count. The shipped shape is E=128, H=2816, I_dev=192, S=256; this keeps the same ratios at 1/8 the
# expert count so the test runs in seconds.
_E, _H, _I, _S, _TOPK = 16, 256, 64, 256, 4


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _make_weights(seed=3):
    """Torch expert weights in the layout ``gemma4`` produces."""
    return SimpleNamespace(
        gate_proj=_rand(1, _E, _H, _I, seed=seed),  # [1,E,H,I] column-parallel
        up_proj=_rand(1, _E, _H, _I, seed=seed + 1),
        down_proj=_rand(1, _E, _I, _H, seed=seed + 2),  # [1,E,I,H] row-parallel
        intermediate_size_per_device=_I,
    )


def _make_routing(seed=9, zero_unselected=True):
    """Dense ``[1,1,S,E]`` routing, top-k masked — the contract ``concat_experts_forward`` assumes."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, _S, _E, generator=g)
    probs = torch.softmax(logits, dim=-1)
    if not zero_unselected:
        return probs
    topk = torch.topk(probs, _TOPK, dim=-1).indices
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk, 1.0)
    return probs * mask


def _torch_oracle(x, w, routing):
    """Per-expert reference: ``sum_e routing_e * (geglu(x@gate_e, x@up_e) @ down_e)``.

    Shapes come from the weights, not from module constants, so a padded-intermediate variant works
    unchanged.
    """
    num_experts, hidden = w.gate_proj.shape[1], w.gate_proj.shape[2]
    seq = x.shape[-2]
    out = torch.zeros(1, 1, seq, hidden, dtype=torch.float32)
    xf = x.float().reshape(seq, hidden)
    for e in range(num_experts):
        gate = xf @ w.gate_proj[0, e].float()
        up = xf @ w.up_proj[0, e].float()
        # tanh GeLU — DiffusionGemma's configured variant (tt/expert_operations.py:apply_gelu)
        act = torch.nn.functional.gelu(gate, approximate="tanh") * up
        out[0, 0] += (act @ w.down_proj[0, e].float()) * routing[0, 0, :, e : e + 1].float()
    return out


def _to_dev(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _fake_experts(w, device):
    """Minimal stand-in for ``Gemma4Experts``: concat_experts_forward reads only these."""
    return SimpleNamespace(
        weights=SimpleNamespace(
            gate_proj=_to_dev(w.gate_proj, device),
            up_proj=_to_dev(w.up_proj, device),
            down_proj=_to_dev(w.down_proj, device),
            intermediate_size_per_device=int(w.gate_proj.shape[3]),
        ),
        mesh_config=None,  # single device -> no all-reduce, isolating the fold itself
        ccl_manager=None,
    )


def _run_concat(device, w, routing, x):
    experts = _fake_experts(w, device)
    tt_x = _to_dev(x, device)
    tt_routing = _to_dev(routing, device)
    try:
        out = concat_moe.concat_experts_forward(experts, tt_x, tt_routing)
        hidden, seq = w.gate_proj.shape[2], x.shape[-2]
        return ttnn.to_torch(out).float().reshape(1, 1, seq, hidden)
    finally:
        # Order matters: release the concat weights FIRST. ``down_cat`` is a view of
        # ``weights.down_proj``, so freeing the root first and then touching the view would read
        # DRAM the allocator has already reclaimed. ``ConcatExpertWeights.deallocate`` uses
        # ``deallocate(False)`` and so correctly skips the aliasing view.
        cached = getattr(experts, "_dg_concat_weights", None)
        if cached is not None:
            cached.deallocate()
        for t in (tt_x, tt_routing, experts.weights.gate_proj, experts.weights.up_proj, experts.weights.down_proj):
            t.deallocate(True)


def test_concat_matches_per_expert_oracle(device):
    """The headline: does the fold reproduce the per-expert MoE on device?"""

    w = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=21) * 0.1  # keep the GeGLU in a sane range for bf16

    got = _run_concat(device, w, routing, x)
    expected = _torch_oracle(x, w, routing)

    passing, pcc = comp_pcc(expected, got, 0.99)
    rel = ((got - expected).abs().max() / expected.abs().max()).item()
    print(f"[concat vs per-expert oracle] {pcc}  max_rel_err={rel:.4f}")
    assert passing, f"concat MoE disagrees with the per-expert oracle: {pcc}"


def test_fold_requires_zero_for_unselected_experts(device):
    """Pin the contract: a router that does NOT zero unselected experts breaks the fold.

    This is not a bug report against the router — ``_denoise_router_forward`` does mask. It exists so
    that if anyone changes the router to return an unmasked distribution, this fails loudly here
    rather than silently degrading generation quality, which is the failure mode that would be
    hardest to attribute.
    """

    w = _make_weights()
    x = _rand(1, 1, _S, _H, seed=22) * 0.1
    masked = _make_routing(zero_unselected=True)
    unmasked = _make_routing(zero_unselected=False)

    got_masked = _run_concat(device, w, masked, x)
    got_unmasked = _run_concat(device, w, unmasked, x)

    # Against the SAME oracle (the masked, top-k one), the unmasked routing must be visibly wrong.
    expected = _torch_oracle(x, w, masked)
    _, pcc_masked = comp_pcc(expected, got_masked, 0.99)
    diff = (got_unmasked - got_masked).abs().max().item()
    print(f"[fold contract] masked {pcc_masked}  |unmasked - masked|_max={diff:.4f}")
    assert diff > 1e-3, (
        "unselected experts made no difference — either the router mask is being applied somewhere "
        "else, or this test is not exercising the fold"
    )


def test_padded_intermediate_columns_contribute_zero(device):
    """``weights.py`` pads I/tp up to a tile; the concat matmul computes over the pad.

    Zero-pad the intermediate on BOTH gate/up and down and check the result is unchanged. If the pad
    ever carried garbage instead of zeros, ``down_cat`` would sum it into every token.
    """

    w = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=23) * 0.1
    baseline = _run_concat(device, w, routing, x)

    pad = 32
    padded = SimpleNamespace(
        gate_proj=torch.nn.functional.pad(w.gate_proj, (0, pad)),  # [1,E,H,I+pad]
        up_proj=torch.nn.functional.pad(w.up_proj, (0, pad)),
        down_proj=torch.nn.functional.pad(w.down_proj, (0, 0, 0, pad)),  # [1,E,I+pad,H]
        intermediate_size_per_device=_I + pad,
    )
    got = _run_concat(device, padded, routing, x)

    _, pcc = comp_pcc(baseline, got, 0.999)
    print(f"[zero pad invariance] {pcc}")
    assert pcc, f"zero-padded intermediate changed the result: {pcc}"


def test_down_concat_is_a_pure_reshape(device):
    """The memory budget rests on ``[1,E,I,H] -> [1,1,E*I,H]`` being free at bf16 TILE."""

    w = _make_weights()
    source = _to_dev(w.down_proj, device)
    try:
        info = concat_moe.verify_down_concat_is_free(SimpleNamespace(down_proj=source))
        print(f"[down concat] {info}")
        assert info["values_match"], f"down concat is not byte-order preserving: {info}"
    finally:
        source.deallocate(True)


def test_deallocate_does_not_free_the_aliased_down_weights(device):
    """``down_cat`` is a view of ``weights.down_proj``; releasing it must not free the root.

    ``deallocate(True)`` bypasses the not-sole-owner guard and reaches the root holder, so a
    force-free here would release the live row-parallel down weights that prefill and the sparse
    path still read — and the crash would surface inside prefill, far from this module.
    """

    w = _make_weights()
    experts = _fake_experts(w, device)
    down = experts.weights.down_proj
    try:
        concat = concat_moe.concat_weights_for(experts)
        assert concat.down_cat.buffer_address() == down.buffer_address(), (
            "down_cat is no longer a view of down_proj — the 7.7 GiB memory budget in "
            "concat_moe.py assumes it is; re-derive it before changing this"
        )
        concat.deallocate()
        assert down.is_allocated(), "ConcatExpertWeights.deallocate freed the shared down weights"
        # And the root must still be readable, not merely flagged allocated.
        assert torch.isfinite(ttnn.to_torch(down).float()).all()
    finally:
        for t in (experts.weights.gate_proj, experts.weights.up_proj, down):
            t.deallocate(True)
