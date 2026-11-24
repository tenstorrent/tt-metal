# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the tp8g4 prefill MoE split (``reference/hy4_moe_tp8g4.py``). No device.

* ``test_split_equals_moe``: the 32 per-chip partials (gather -> 64 sliced experts -> weighted combine), summed over
  slices then groups, equal ``reference/hy4_moe.routed_moe`` (HF ``HYV4MoE``'s routed half) in fp64, on random
  weights at reduced hidden / intermediate with the real 256 experts, top-8 and the clamp in play; also with every
  pick of every token in one group (the imbalance worst case).
* ``test_plan``: the group plans: tile-aligned regions, every pair once, combine slots consistent.
* ``test_slices_match_decode``: ``slice_weights`` is decode's ``prepare.chip_weights(split="tp8g4")`` for the routed
  entries (chip c = slice c // 4 of group c % 4), and BFP4 of a slice equals the slice of BFP4 of the whole expert
  (block exponents never straddle a slice), so the split costs nothing in weight precision against EP32.
* ``test_golden_accuracy`` (``HY4_TP8G4_GOLDEN_LAYERS``, default 1,2,4; skips without the checkpoint / streams): on
  the 4K LongBench golden (``HY4_IHC_REF_STREAMS``), the layer's FFN input from the reference streams, the split with
  exact / BFP4 / two-term BFP4 / BFP8 expert weights against the fp32 reference: routed-output, MoE-output and
  post-layer stream PCC. Rows go to the log (``HY4_TP8G4``) and to ``HY4_TP8G4_REPORT`` (JSON lines). Needs
  ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` (ttnn's host BFP4 conversion opens the board otherwise).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference import hy4_moe as ref
from models.demos.deepseek_v3_d_p.reference import hy4_moe_tp8g4 as t8

CKPT = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
STREAMS = Path(os.environ.get("HY4_IHC_REF_STREAMS", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-4k-ropeint-streams"))
WORK = Path(os.environ.get("HY4_TP8G4_WORK", "/mnt/tt-data/dwadhwani/hy4-test/moe-tp8g4"))
REPORT = os.environ.get("HY4_TP8G4_REPORT")
GOLDEN_LAYERS = [int(v) for v in os.environ.get("HY4_TP8G4_GOLDEN_LAYERS", "1,2,4").split(",")]


def _random_moe(S=512, H=256, I=256, seed=0, one_group=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(S, H, generator=g, dtype=torch.float64)
    gate_up = torch.randn(t8.E, 2 * I, H, generator=g, dtype=torch.float64) * (3.0 / H**0.5)  # some values clamp
    down = torch.randn(t8.E, H, I, generator=g, dtype=torch.float64) / I**0.5
    scores = torch.rand(S, t8.E, generator=g, dtype=torch.float64)
    if one_group:  # every token's 8 picks in group (t % 4)
        mask = torch.full((S, t8.E), -1.0, dtype=torch.float64)
        for t in range(S):
            mask[t, (t % 4) * t8.EPG : (t % 4 + 1) * t8.EPG] = 0
        scores = scores + mask * 10
    top_i = scores.topk(t8.TOPK, dim=-1)[1]
    top_w = torch.rand(S, t8.TOPK, generator=g, dtype=torch.float64)
    top_w = top_w / top_w.sum(-1, keepdim=True) * 2.827
    return x, gate_up, down, top_i, top_w


@pytest.mark.parametrize("one_group", [False, True], ids=["spread", "one_group"])
def test_split_equals_moe(one_group):
    x, gate_up, down, top_i, top_w = _random_moe(one_group=one_group)
    expect = torch.zeros_like(x)
    for e, tok, slot in ref._expert_groups(top_i, t8.E):
        g, u = (x[tok] @ gate_up[e].T).chunk(2, dim=-1)
        expect.index_add_(0, tok, (ref.clamped_swiglu(g, u, t8.LIMIT) @ down[e].T) * top_w[tok, slot, None])
    got = t8.tp8g4_routed(x, top_i, top_w, gate_up, down, dtype=torch.float64)
    clamped = sum(
        float((torch.chunk(x[tok] @ gate_up[e].T, 2, -1)[0] > t8.LIMIT).sum())
        for e, tok, _ in ref._expert_groups(top_i, t8.E)
    )
    assert clamped > 0, "the test input must reach the clamp"
    torch.testing.assert_close(got, expect, rtol=1e-10, atol=1e-10)
    if one_group:
        assert all(p.K == 8 for p in t8.plan(top_i, top_w))


def test_plan():
    _, _, _, top_i, top_w = _random_moe(S=1024)
    plans = t8.plan(top_i, top_w)
    assert sum(p.pairs for p in plans) == top_i.numel()
    for p in plans:
        assert (p.offsets % t8.TILE == 0).all() and p.R % t8.TILE == 0
        assert int((p.rows >= 0).sum()) == p.pairs
        used = p.slot_rows[p.slot_rows >= 0]
        assert used.numel() == p.pairs and used.unique().numel() == p.pairs
        # each slot row belongs to its token
        for k in range(p.K):
            t = (p.slot_rows[k] >= 0).nonzero().flatten()
            assert (p.rows[p.slot_rows[k, t]] == t).all()
        # weights land on the right (token, expert) pair
        for j in range(0, t8.EPG, 7):
            a, m = int(p.offsets[j]), int(p.counts[j])
            for r in range(a, a + m):
                t = int(p.rows[r])
                slot = (top_i[t] == p.group * t8.EPG + j).nonzero().item()
                assert p.weights[r] == top_w[t, slot]


def _mock_ok():
    return bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


@pytest.mark.skipif(not _mock_ok(), reason="host BFP4 conversion needs TT_METAL_MOCK_CLUSTER_DESC_PATH")
def test_slices_match_decode():
    from models.experimental.hy4_preview.fused.moe import layout as L
    from models.experimental.hy4_preview.fused.moe import prepare

    g = torch.Generator().manual_seed(1)
    H, I = 6144, 2048
    gate_up = torch.randn(t8.E, 2 * I, 64, generator=g).to(torch.bfloat16)  # reduced hidden for the layout check
    down = torch.randn(t8.E, 64, I, generator=g).to(torch.bfloat16)
    for chip in (0, 5, 18, 31):
        group, part = t8.group_slice(chip)
        assert (group, part) == prepare.chip_group("tp8g4", chip)
        dg, du, dd = prepare.chip_weights(gate_up, down, "tp8g4", chip)
        mg, mu, md = (w.float() for w in t8.slice_weights(gate_up, down, group, part))
        assert torch.equal(dg[: t8.EPG], mg) and torch.equal(du[: t8.EPG], mu) and torch.equal(dd[: t8.EPG], md)
    # BFP4 blocks: a slice of the BFP4 expert = BFP4 of the slice (gate^T [H, I] tiles, down^T [I, H] tiles)
    w = torch.randn(4, H, I, generator=g).to(torch.bfloat16).float()  # [E, K=H, N=I] as stored for gate / up
    whole = L.bfp4_round(w)
    for part in (0, 3, 7):
        c0, c1 = part * 256, (part + 1) * 256
        assert torch.equal(whole[:, :, c0:c1], L.bfp4_round(w[:, :, c0:c1].contiguous()))
    wd = torch.randn(4, I, 512, generator=g).to(torch.bfloat16).float()  # down^T [K=I, N=H]
    whole = L.bfp4_round(wd)
    for part in (0, 7):
        c0, c1 = part * 256, (part + 1) * 256
        assert torch.equal(whole[:, c0:c1], L.bfp4_round(wd[:, c0:c1].contiguous()))


# ------------------------------------------------------------------------------------------------------- golden
def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def golden_pack(layer: int, ckpt=None) -> dict:
    """The layer's FFN input, picks and fp32 reference outputs from the reference streams (cached under WORK)."""
    path = WORK / "golden" / f"layer_{layer}.pt"
    if path.is_file():
        return torch.load(path, weights_only=True)
    from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import Hy4Checkpoint

    ckpt = ckpt or Hy4Checkpoint(CKPT)
    cfg = ckpt.config
    streams = torch.load(STREAMS / f"streams_L{layer}_attn.pt", map_location="cpu", weights_only=True).float()
    hn, post = t8.golden_moe_inputs(ckpt, streams, layer, cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude)
    p = f"model.layers.{layer}.mlp."
    top_i, top_w = ref.route(
        hn,
        ckpt.get(p + "gate.weight", torch.float32),
        ckpt.get(p + "gate.e_score_correction_bias", torch.float32),
        top_k=cfg.num_experts_per_tok,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    gate_up, down = ckpt.get(p + "experts.gate_up_proj"), ckpt.get(p + "experts.down_proj")
    routed = ref.routed_moe(hn, top_i, top_w, gate_up, down, limit=cfg.swiglu_limit)
    shared = ref.swiglu_mlp(hn, *(ckpt.get(p + f"shared_experts.{n}_proj.weight") for n in ("gate", "up", "down")))
    after = torch.load(STREAMS / f"streams_L{layer}_ffn.pt", map_location="cpu", weights_only=True).float()
    pack = dict(hn=hn, post=post, top_i=top_i, top_w=top_w, routed=routed, shared=shared, streams_attn=streams)
    pack["stream_check_pcc"] = torch.tensor(_pcc(streams + post[..., None] * (routed + shared)[:, None], after))
    pack["streams_ffn"] = after
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, path)
    return pack


@pytest.mark.skipif(not (CKPT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint")
@pytest.mark.skipif(not (STREAMS / "streams_L1_attn.pt").is_file(), reason="no reference streams")
@pytest.mark.skipif(not _mock_ok(), reason="host BFP4 conversion needs TT_METAL_MOCK_CLUSTER_DESC_PATH")
@pytest.mark.parametrize("layer", GOLDEN_LAYERS)
@pytest.mark.timeout(7200)
def test_golden_accuracy(layer):
    from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import Hy4Checkpoint

    ckpt = Hy4Checkpoint(CKPT)
    t0 = time.time()
    pk = golden_pack(layer, ckpt)
    logger.info(
        f"HY4_TP8G4 layer {layer} golden pack {time.time() - t0:.0f}s stream check {float(pk['stream_check_pcc'])}"
    )
    assert float(pk["stream_check_pcc"]) > 0.99999, "the FFN input / outputs do not reproduce the reference streams"
    p = f"model.layers.{layer}.mlp."
    gate_up, down = ckpt.get(p + "experts.gate_up_proj"), ckpt.get(p + "experts.down_proj")
    plans = t8.plan(pk["top_i"], pk["top_w"])
    hops = t8.hop_bytes(plans, pk["top_i"])
    logger.info(f"HY4_TP8G4 layer {layer} hops {json.dumps(hops)}")
    _emit(dict(test="hops", layer=layer, **hops))
    variants = {
        "exact": (None, torch.float32),
        "bfp4": (t8.bfp4_slices, torch.float32),
        "bfp4_x_bf16": (t8.bfp4_slices, "bf16x"),
        "two_term_bfp4": (t8.two_term_bfp4_slices, torch.float32),
        "bfp8": (t8.bfp8_slices, torch.float32),
    }
    only = os.environ.get("HY4_TP8G4_VARIANTS")
    for name, (fn, dt) in variants.items():
        if only and name not in only.split(","):
            continue
        t0 = time.time()
        x = pk["hn"]
        if dt == "bf16x":  # the device's bf16 input (activations otherwise exact)
            x, dt = x.to(torch.bfloat16).float(), torch.float32
        got = t8.tp8g4_routed(x, pk["top_i"], pk["top_w"], gate_up, down, dtype=dt, weights_fn=fn, plans=plans)
        moe = got + pk["shared"]
        after = pk["streams_attn"] + pk["post"][..., None] * moe[:, None]
        row = dict(
            test="accuracy",
            layer=layer,
            variant=name,
            pcc_routed=_pcc(got, pk["routed"]),
            rel_err_routed=_rel(got, pk["routed"]),
            pcc_moe=_pcc(moe, pk["routed"] + pk["shared"]),
            rel_err_moe=_rel(moe, pk["routed"] + pk["shared"]),
            pcc_stream=_pcc(after, pk["streams_ffn"]),
            rel_err_stream_delta=_rel(after - pk["streams_attn"], pk["streams_ffn"] - pk["streams_attn"]),
            max_abs_routed=float((got - pk["routed"]).abs().max()),
            seconds=round(time.time() - t0, 1),
        )
        _emit(row)
        if name == "exact":
            assert row["pcc_routed"] > 0.999999, row


def _emit(row):
    logger.info(f"HY4_TP8G4 {json.dumps(row)}")
    if REPORT:
        with open(REPORT, "a") as f:
            f.write(json.dumps(row) + "\n")
