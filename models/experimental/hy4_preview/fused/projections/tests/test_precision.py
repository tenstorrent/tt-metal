# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision of every projection of a chip's share with the checkpoint's weights of layers 0-5, on the same inputs:

    the programs of fused/projections (custom_mm is LoFi) with the weights stored as BFP8, as two BFP4 terms and as
    one BFP4 term, against exact arithmetic on the checkpoint's weights (the references restricted to the chip), and
    ttnn.matmul with BFP8 weights at HiFi2 with FP32 accumulation (what tt/attention.py runs), chained the same way.

Every stage is fed the reference's input for that stage. One HY4_MK7 line per (layer, output, form): pcc, relative
rms error, gain. The weight streams come from the cache of prepare.py when it has them. Needs $HY4_HF_MODEL."""

import os

import pytest
import torch

import ttnn
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import FID_KW, FID_TAG, check, random_input, tables
from models.experimental.hy4_preview.fused.projections.tests.test_post import post_inputs
from models.experimental.hy4_preview.fused.projections.tests.test_pre import FORMATS, stored_share
from models.experimental.hy4_preview.fused.projections.tests.test_query import query_inputs
from models.experimental.hy4_preview.tt.common import compute_config

LAYERS = [int(v) for v in os.environ.get("HY4_PRECISION_LAYERS", "0,1,2,3,4,5").split(",")]
CHIP = int(os.environ.get("HY4_PRECISION_CHIP", "13"))
_formats = [f for f in os.environ.get("HY4_PRECISION_FORMATS", "bfp8,bfp4x2,bfp4").split(",") if f]


def ttnn_matmul(device, a, wt):
    """a [8, K] (BF16) @ wt [K, N] (BFP8) as tt/attention.py's projections run it: HiFi2, FP32 accumulation."""
    ta = ttnn.from_torch(a.reshape(1, 1, *a.shape).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(
        wt.reshape(1, 1, *wt.shape).float().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    ckc = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True)
    out = ttnn.to_torch(ttnn.matmul(ta, tw, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
    return out.reshape(a.shape[0], wt.shape[1])


def bf16(t):
    return t.to(torch.bfloat16).float()


@pytest.mark.parametrize("layer", LAYERS)
def test_precision_checkpoint(device, layer):
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    lw = prepare.layer_weights(layer)
    w = L.chip_weights(lw, CHIP)
    tag = f"precision L{layer} chip {CHIP}"
    y, scale = random_input(seed=10 + layer)
    q_a, lanes = query_inputs(lw, seed=10 + layer)
    _, _, d_cos, d_sin = tables()
    ctx, gate = post_inputs(w, seed=10 + layer)
    pre_ref = L.pre_golden(y, scale, w, d_cos, d_sin)
    q_ref, _ = L.query_golden(q_a, lanes, d_cos, d_sin, w)
    post_ref, gated_ref = L.post_golden(ctx, gate, w)

    def report(name, form, ref, dev, stored=None):
        """dev against the reference (the checkpoint's weights) and, for the fused programs, against the same
        arithmetic on the weights their streams hold."""
        if not P.MOCK:
            check(f"{tag} {name} [{form}]", ref, dev)
            if stored is not None:
                check(f"{tag} {name} [{form}] vs exact arithmetic on the stored weights", stored, dev)

    for fmt in _formats:
        f, terms = FORMATS[fmt]
        cache = lambda group, name=fmt: prepare.stream_cache(prepare.DEFAULT_OUT, layer, name, CHIP, group)
        fmt = fmt + FID_TAG  # the label of the lines below (not of the cache)
        ws = stored_share(w, f, terms) if not P.MOCK else w
        pre_st = L.pre_golden(y, scale, ws, d_cos, d_sin)
        q_st, _ = L.query_golden(q_a, lanes, d_cos, d_sin, ws)
        post_st, _ = L.post_golden(ctx, gate, ws)
        op = P.pre_program(device, w, fmt=f, terms=terms, cache=cache("pre"), **FID_KW)
        op.set_x(torch.cat((y, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
        op.set_params(scale=scale)
        op()
        if not P.MOCK:
            out, stat, status = op.read()
            op.check_status(status)
            report("q_a", fmt, pre_ref["q_a"], out[:, : 2 * L.TILE], pre_st["q_a"])
            if "pool" in w:
                report(
                    f"pool {L.pool_of(CHIP, lw.full)}",
                    fmt,
                    pre_ref["pool"],
                    out[:, 2 * L.TILE : 3 * L.TILE],
                    pre_st["pool"],
                )
            report(
                "gate (after sigmoid)",
                fmt,
                pre_ref["gate"],
                out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE],
                pre_st["gate"],
            )
            report("q_a lane sums", fmt, pre_ref["lanes"], L.stat_tile_lanes(stat), pre_st["lanes"])
            if "idx_q" in w:
                report(
                    "index query (fused weight, RoPE, Hadamard)",
                    fmt,
                    pre_ref["idx_q_dev"],
                    out[:, P.PRE_IDX * L.TILE :],
                    pre_st["idx_q_dev"],
                )
        op = P.query_program(device, w, fmt=f, terms=terms, cache=cache("query"), **FID_KW)
        op.set_x(torch.cat((q_a, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
        op.set_params(lanes=lanes)
        op()
        if not P.MOCK:
            out, _, status = op.read()
            op.check_status(status)
            dev = out.reshape(L.ROWS, L.HEADS_PER_CHIP, L.LATENT).permute(1, 0, 2)
            report(
                "q_abs (q_b, norm, absorption)",
                fmt,
                q_ref[..., : L.KV_LORA],
                dev[..., : L.KV_LORA],
                q_st[..., : L.KV_LORA],
            )
            report(
                "q_rope (q_b, norm, RoPE)", fmt, q_ref[..., L.KV_LORA :], dev[..., L.KV_LORA :], q_st[..., L.KV_LORA :]
            )
        op = P.post_program(device, w, fmt=f, terms=terms, cache=cache("post"), **FID_KW)
        op.set_x(torch.cat((ctx[0], ctx[1], gate), dim=1))
        op()
        if not P.MOCK:
            out, _, status = op.read()
            op.check_status(status)
            report("o_proj partial (kv_b value side, gate, o_proj)", fmt, post_ref, out, post_st)
    if P.MOCK:
        return

    # the ttnn form of the same matmuls on the same inputs
    form = "ttnn bfp8 HiFi2"
    xn = bf16(y * scale[:, None])
    report("q_a", form, pre_ref["q_a"], ttnn_matmul(device, xn, w["q_a"]))
    if "pool" in w:
        report(f"pool {L.pool_of(CHIP, lw.full)}", form, pre_ref["pool"], ttnn_matmul(device, xn, w["pool"]))
    report("gate (after sigmoid)", form, pre_ref["gate"], torch.sigmoid(ttnn_matmul(device, xn, w["gate"])))
    if "idx_q" in w:
        report("index query (fused weight, before RoPE)", form, pre_ref["idx_q"], ttnn_matmul(device, xn, w["idx_q"]))
    s = L.norm_scalar(lanes)
    qn = bf16(q_a * s[:, None])
    q_abs, q_rope = [], []
    for wq, wk in zip(w["q_b"], w["w_k"]):
        q = ttnn_matmul(device, qn, wq)
        q_abs.append(ttnn_matmul(device, q[:, : L.NOPE], wk))
        q_rope.append(bf16(L.interleaved_rope(q[:, L.NOPE :], d_cos, d_sin)))
    report("q_abs (q_b, norm, absorption)", form, q_ref[..., : L.KV_LORA], torch.stack(q_abs))
    report("q_rope (q_b, norm, RoPE)", form, q_ref[..., L.KV_LORA :], torch.stack(q_rope))
    attn = torch.cat([ttnn_matmul(device, c, wv) for c, wv in zip(ctx, w["w_v"])], dim=1)
    report(
        "o_proj partial (kv_b value side, gate, o_proj)", form, post_ref, ttnn_matmul(device, bf16(attn * gate), w["o"])
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_precision_normalised(device, layer):
    """The pre group without the deferred norm on the checkpoint's weights: x arrives normalised (the root applied
    1 / RMS and the norm's weight, BF16), scale = 1, the streams of `prepare.py --normalised` (<format>_xn: no norm
    weight folded in, pre-rounded terms, so they hold layout.stored of the weights). Against the reference of the
    deferred form and against exact arithmetic on the stored weights."""
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    _, gamma, _ = prepare.checkpoint_layer(layer)
    w_deferred = L.chip_weights(prepare.layer_weights(layer), CHIP)
    w = L.chip_weights(prepare.layer_weights(layer, gamma=False), CHIP)
    y, scale = random_input(seed=10 + layer)
    _, _, d_cos, d_sin = tables()
    xn = bf16(((y.double() * scale.double()[:, None]) * gamma.double()).float())
    cache = prepare.stream_cache(prepare.DEFAULT_OUT, layer, "bfp8_xn", CHIP, "pre")
    op = P.pre_program(device, w, fmt="bfp8", cache=cache, pre_rounded=True, **FID_KW)
    op.set_x(torch.cat((xn, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params()
    op()
    if P.MOCK:
        return
    out, _, status = op.read()
    op.check_status(status)
    ref = L.pre_golden(y, scale, w_deferred, d_cos, d_sin)
    held = {k: L.stored(v.float().contiguous(), "bfp8") if not isinstance(v, list) else v for k, v in w.items()}
    here = L.pre_golden(xn, torch.ones(L.ROWS), held, d_cos, d_sin)
    tag = f"precision L{layer} chip {CHIP} normalised x"
    for name, dev in (
        ("q_a", out[:, : 2 * L.TILE]),
        ("gate (after sigmoid)", out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE]),
    ):
        key = name.split(" ")[0]
        check(f"{tag} {name} [bfp8_xn{FID_TAG}]", ref[key], dev)
        check(f"{tag} {name} [bfp8_xn{FID_TAG}] vs exact arithmetic on the stored weights", here[key], dev)
