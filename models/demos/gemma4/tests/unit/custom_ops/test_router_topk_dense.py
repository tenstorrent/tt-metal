# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/router_topk DENSE mode (prefill rows) vs the torch golden and the composite chain of tt/router.py
(softmax, topk, sum, div, zeros_like + scatter, per-expert scale), single chip: 32 / 128 / 1024 rows (1 .. 8 rows
per core), 128 experts, top-8; the ones-scale form; cache hit on fresh tensors."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.router_topk import RouterTopk

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _composite(logits_tt, pes_tt, k):
    probs = ttnn.softmax(logits_tt, dim=-1)
    v, i = ttnn.topk(probs, k=k, dim=-1)
    v = ttnn.div(v, ttnn.sum(v, dim=-1, keepdim=True))
    dense = ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=i, src=v)
    return ttnn.mul(dense, pes_tt)


@pytest.mark.parametrize(
    "S, E, k, with_scale",
    [(32, 128, 8, True), (128, 128, 8, True), (1024, 128, 8, True), (128, 128, 8, False), (128, 64, 4, True)],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_router_topk_dense(mesh_device, device_params, S, E, k, with_scale):
    torch.manual_seed(0)
    logits = (torch.randn(1, 1, S, E) * 2).to(torch.bfloat16)
    pes = ((torch.rand(E) + 0.5) if with_scale else torch.ones(E)).to(torch.bfloat16)
    gold = RouterTopk.golden_dense(logits, pes, k)
    tl = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    pes_rm = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    pes_tile = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ref = ttnn.to_torch(_composite(tl, pes_tile, k)).float()
    out = RouterTopk.op_dense(tl, pes_rm, k)
    assert out.layout == ttnn.TILE_LAYOUT and [int(v) for v in out.shape] == [1, 1, S, E]
    got = ttnn.to_torch(out).float()
    lf = logits.float().reshape(S, E)
    sel = got.reshape(S, E) != 0
    assert (sel.sum(-1) == k).all(), "every row must carry exactly k weights"
    # exact validity of the selection: the smallest selected logit is >= the largest unselected one (a bf16 tie at
    # the k-th boundary may be resolved either way: torch.topk and the kernel break ties differently)
    sel_min = torch.where(sel, lf, torch.full_like(lf, float("inf"))).min(-1).values
    unsel_max = torch.where(sel, torch.full_like(lf, float("-inf")), lf).max(-1).values
    assert (sel_min >= unsel_max).all(), "a selected expert ranks below an unselected one"
    # the weights on the op's own selection: fp32 softmax-normalized over the selected, times the scale
    probs = torch.softmax(lf, dim=-1)
    wk = torch.where(sel, probs, torch.zeros_like(probs))
    wk = wk / wk.sum(-1, keepdim=True) * pes.float()
    p_sel = _pcc(wk, got.reshape(S, E))
    # against the torch golden / the composite chain: identical except on rows with a boundary tie
    same_rows = ((gold.reshape(S, E) != 0) == sel).all(-1)
    p_g, p_r = _pcc(gold, got), _pcc(ref, got)
    print(
        f"[S{S} E{E} k{k} scale={with_scale}] pcc vs own-selection golden {p_sel:.6f} (max|d| {(wk - got.reshape(S, E)).abs().max():.4f}); "
        f"vs torch golden {p_g:.6f}, vs composite {p_r:.6f}; rows with the golden's selection {int(same_rows.sum())}/{S}"
    )
    assert p_sel > 0.9999, p_sel
    # rows whose selection differs from torch's are boundary ties (the validity check above holds for every row);
    # with bf16 logits they are common (1-3 % of random rows), so agreement is reported, not asserted
    assert p_g > 0.999 and p_r > 0.999, (p_g, p_r)
    tl2 = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out2 = RouterTopk.op_dense(tl2, pes_rm, k)
    assert torch.equal(got, ttnn.to_torch(out2).float()), "cache-hit call differs"
