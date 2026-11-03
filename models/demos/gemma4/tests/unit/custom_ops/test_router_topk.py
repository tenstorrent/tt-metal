# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RouterTopk (custom op) vs the HF-order torch golden (softmax, top-k, sum-normalize, per-expert scale, local
expert columns) on a single chip (base 0 and a non-zero base)."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.router_topk import RouterTopk

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _inputs(B, E, seed):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(1, 1, B, E, generator=g) * 3).to(torch.bfloat16)
    pes = (torch.rand(E, generator=g) + 0.5).to(torch.bfloat16)
    return logits, pes


@pytest.mark.parametrize("B, E, e_loc, base", [(32, 128, 32, 0), (32, 128, 32, 64), (1, 128, 32, 96), (8, 128, 32, 32)])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_router_topk_matches_golden(mesh_device, device_params, B, E, e_loc, base):
    k = 8
    logits, pes = _inputs(B, E, seed=B + base)
    w_ref, sp_ref = RouterTopk.golden(logits, pes, base, e_loc, k)
    t_logits = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_pes = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    t_base = ttnn.from_torch(
        torch.full((1, 1, 1, 16), base, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    w, sp = RouterTopk.op(t_logits, t_pes, t_base, e_loc, k)
    w = ttnn.to_torch(w).float()
    sp = ttnn.to_torch(sp).float()
    assert list(w.shape) == [1, 1, e_loc, 32] and list(sp.shape) == [1, 1, 1, e_loc]
    sel_ref, sel = (w_ref != 0), (w != 0)
    flips = int((sel_ref != sel).sum())
    # a mismatch is legitimate only for an exact bf16 tie at the k-th logit (torch.topk and the kernel break
    # ties differently): every differing (e, u) must have the same logit as some selected expert of that user
    ties = 0
    lg = logits.float()[0, 0]  # [B, E]
    diff = sel_ref[0, 0] != sel[0, 0]  # [e_loc, 32]
    for e, u in torch.nonzero(diff).tolist():
        # a tie at the k-th logit of the user's row (the partner may be an expert of another chip)
        kth = torch.topk(lg[u], k).values[-1]
        ties += int(lg[u, base + e] == kth)
    print(
        f"[B{B} base{base}] selection mismatches {flips} (ties {ties}), max|w diff| {(w - w_ref).abs().max():.5f}, sparsity ok {torch.equal(sp, sp_ref)}"
    )
    assert flips == ties, "the selected experts differ from the golden beyond exact ties"
    if flips == 0:
        assert torch.equal(sp, sp_ref)
        assert (w - w_ref).abs().max() <= 1e-2 * w_ref.abs().max() + 1e-3
        pcc = torch.corrcoef(torch.stack([w.flatten(), w_ref.flatten()]))[0, 1].item()
        assert pcc > 0.9999, pcc
    # program-cache hit on fresh buffers; twice (the semaphore must be reset between runs)
    for _ in range(2):
        t2 = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        w2, sp2 = RouterTopk.op(t2, t_pes, t_base, e_loc, k)
        assert torch.equal(ttnn.to_torch(w2).float(), w) and torch.equal(ttnn.to_torch(sp2).float(), sp)


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_router_topk_vs_shipped_path(mesh_device, device_params):
    """The op against the shipped ttnn chain (softmax, topk, sum, div, scatter, mul): same selection, same values
    up to bf16 rounding, on 32 random rows."""
    B, E, e_loc, k = 32, 128, 32, 8
    logits, pes = _inputs(B, E, seed=7)
    t_logits = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_pes_tile = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    probs = ttnn.softmax(t_logits, dim=-1)
    v, i = ttnn.topk(probs, k=k, dim=-1)
    v = ttnn.div(v, ttnn.sum(v, dim=-1, keepdim=True))
    dense = ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=i, src=v)
    dense = ttnn.to_torch(ttnn.mul(dense, t_pes_tile)).float()[0, 0]  # [B, E]
    t_pes = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    t_base = ttnn.from_torch(
        torch.full((1, 1, 1, 16), 32, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    w, sp = RouterTopk.op(t_logits, t_pes, t_base, e_loc, k)
    w = ttnn.to_torch(w).float()[0, 0, :, :B].t()  # [B, e_loc]
    ref = dense[:, 32 : 32 + e_loc]
    flips = int(((w != 0) != (ref != 0)).sum())
    print(f"[vs shipped] selection mismatches {flips}, max|diff| {(w - ref).abs().max():.5f}")
    assert flips == 0
    assert (w - ref).abs().max() <= 2e-2


@pytest.mark.parametrize("B", [32, 8, 1], ids=["b32", "b8", "b1"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_router_topk_packed_matches_golden(mesh_device, device_params, B):
    """PACKED mode (the decode MoE kernel rows): [1, 1, 32, 2k] bf16 ROW_MAJOR = weights | global ids per user row,
    rows >= B weight 0 and ids 0..k-1; selection equal to the golden up to exact bf16 ties; cache hit bitwise."""
    E, k = 128, 8
    logits, pes = _inputs(B, E, seed=100 + B)
    ref = RouterTopk.golden_packed(logits, pes, k)
    t_logits = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t_pes = ttnn.from_torch(
        pes.reshape(1, 1, 1, E), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    t_base = ttnn.from_torch(
        torch.zeros((1, 1, 1, 16), dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    out = RouterTopk.op_packed(t_logits, t_pes, t_base, k)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT and list(out.shape) == [1, 1, 32, 2 * k]
    o = ttnn.to_torch(out).float()
    ids, ids_ref = o[0, 0, :, k:].to(torch.int64), ref[0, 0, :, k:].to(torch.int64)
    w, w_ref = o[0, 0, :, :k], ref[0, 0, :, :k]
    lg = logits.float()[0, 0]
    bad = 0
    for u in range(B):
        if not torch.equal(ids[u].sort().values, ids_ref[u].sort().values):
            kth = torch.topk(lg[u], k).values[-1]
            diff = set(ids[u].tolist()) ^ set(ids_ref[u].tolist())
            bad += int(not all(lg[u, e] == kth for e in diff))
    print(f"[packed B{B}] rows with a non-tie selection mismatch {bad}, max|w diff| {(w - w_ref).abs().max():.5f}")
    assert bad == 0
    assert torch.equal(ids[B:], ids_ref[B:]) and torch.equal(w[B:], w_ref[B:]), "padding rows: weight 0, ids 0..k-1"
    same = ids == ids_ref
    assert (w[same] - w_ref[same]).abs().max() <= 1e-2 * w_ref.abs().max() + 1e-3
    for _ in range(2):
        t2 = ttnn.from_torch(logits, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        assert torch.equal(ttnn.to_torch(RouterTopk.op_packed(t2, t_pes, t_base, k)).float(), o)
