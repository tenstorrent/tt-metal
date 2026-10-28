# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Op-level masked-SDPA tests (F1, F9, PLAN v1.12, v1.13, v1.20; CONTRACT §6B, D0 amendment 2).

Every padded length and masked chunk the code can reach (``dataflow.PAD_TABLE`` plus the measured alternatives),
with the model's own SDPA program and compute config in both precisions (and ``exp_approx_mode``, a round-1
candidate). Keys only are masked, with the finite fill: a leading masked region of >= 2 whole k-chunks (where the
length allows) and the pad-key tail. The SDPA output alone is compared with torch fp32; the whole padded output
must be finite. The positive control drops the mask on one masked key block and must fail.
"""

import dataclasses

import pytest
import torch
import ttnn

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.helpers import pcc
from models.experimental.tafsut_forecast.tests.pcc import metrics as M
from models.experimental.tafsut_forecast.tt.common import MASK_FILL, Precision, ckc

NH, DH = df.NH, df.DH
# (bucket context, s_pad, chunk): the table, plus the alternatives D0 Q2 / CONTRACT §7 item 1 measure.
SHAPES = [(b, *df.PAD_TABLE[b]) for b in df.BUCKETS] + [(32768, 1152, 64)]
# Batch per SDPA call that the model really issues at each padded length (D1 ruling 3): the model runs the whole
# batch through one SDPA call, so this is each golden's rows at each forward's bucket. 128: single 1, ragged4 4,
# model16/fallback 16, long_horizon forward 1 512, paper 1024. 256: horizon4 / long_horizon forward 2 (4, 512).
# 384: gift_ragged_odd 16. 640: no case reaches it, so 2 and 32 cover the row. 1088: single_long 1, long_context /
# gift_ragged 32. 1152/64 is the round-1 alternative at long_context's 32. The odd-chunk-count bug shows at B = 16 / 32,
# not at 2 or 64 (jobs 932/933), so batch is part of the trigger.
# D2 item 3 (v1.21): the GIFT predictor issues B = 32 at every bucket, and a ragged final batch (7) at any of them.
MODEL_BATCHES = {
    128: (1, 4, 7, 16, 32, 512, 1024),
    256: (4, 7, 32, 512),
    384: (7, 16, 32),
    640: (2, 7, 32),
    1088: (1, 7, 32),
    1152: (1, 32),
}
CASES = [(c, s, k, B) for c, s, k in SHAPES for B in MODEL_BATCHES[s]]
PRECS = {
    "accuracy": Precision.accuracy(),
    "performance": Precision.performance(),
    "accuracy_exp_approx": dataclasses.replace(Precision.accuracy(), exp_approx=True),
}


def key_valid(ctx_len: int, s_pad: int, chunk: int, B: int = 2) -> torch.Tensor:
    """``[B, s_pad]``: row 0 with a leading masked run of 2 chunks + 17 keys, or every context key when the context
    is shorter (the all-NaN row: REG and future stay valid, so this is the most whole chunks the length allows);
    row 1 with one masked interior patch; both with the pad tail masked."""
    S = ctx_len // df.PATCH + df.NFUT + 1
    P = ctx_len // df.PATCH
    kv = torch.zeros(B, s_pad, dtype=torch.bool)
    kv[:, :S] = True
    lead = min(2 * chunk + 17, P)
    kv[0, :lead] = False
    if B > 1:
        kv[1, 5] = False
    for b in range(2, B):  # ragged rows: GIFT-style leading padding of every length
        kv[b, : (b * 37) % (P + 1)] = False
    return kv


def run_sdpa(device, q, k, v, bias, chunk, prec: Precision, q_chunk: int | None = None):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk or chunk,
        k_chunk_size=chunk,
        exp_approx_mode=prec.exp_approx,
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        up(q),
        up(k),
        up(v),
        attn_mask=up(bias),
        is_causal=False,
        scale=1.0,
        program_config=pc,
        compute_kernel_config=ckc(prec.sdpa_fidelity),
    )
    return ttnn.to_torch(out).float()


@pytest.mark.parametrize("prec", list(PRECS))
@pytest.mark.parametrize("ctx_len,s_pad,chunk,B", CASES, ids=[f"{s}-c{k}-B{B}" for _, s, k, B in CASES])
def test_sdpa_masked(device, ctx_len, s_pad, chunk, prec, B):
    p = PRECS[prec]
    assert df.chunk_count_ok(s_pad, chunk)
    g = torch.Generator().manual_seed(s_pad + chunk)
    q, k, v = (torch.randn(B, NH, s_pad, DH, generator=g) * 0.6 for _ in range(3))  # logit std ~2.9 at scale 1.0
    kv = key_valid(ctx_len, s_pad, chunk, B)
    bias = torch.where(kv, 0.0, MASK_FILL)[:, None, None, :].expand(B, 1, s_pad, s_pad).contiguous()
    qb, kb, vb = (t.to(torch.bfloat16).float() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=bias, scale=1.0)
    out = run_sdpa(device, q, k, v, bias, chunk, p)
    assert torch.isfinite(out).all(), "non-finite SDPA output (padded rows included)"
    S = ctx_len // df.PATCH + df.NFUT + 1
    got, exp = out[:, :, :S], ref[:, :, :S]
    pc = pcc(got, exp)
    err = float((got - exp).abs().max())
    # the same call at a different valid chunk: chunking must not change the result (n/a when no other chunk is valid)
    alt = [c for c in (32, 64, 128, 256) if c != chunk and df.chunk_count_ok(s_pad, c)]
    if alt:
        other = run_sdpa(device, q, k, v, bias, alt[-1], p)[:, :, :S]
        chunk_diff = f"|c{chunk}-c{alt[-1]}| {float((got - other).abs().max()):.4f}"
        assert float((got - other).abs().max()) < 0.1
    else:
        chunk_diff = "other chunk n/a"
    # positive control (v1.13): drop the mask on one whole masked key block of row 0
    ctrl = bias.clone()
    blk = (~kv[0]).nonzero().flatten()[: min(chunk, 32)]
    ctrl[0, :, :, blk] = 0.0
    bad = run_sdpa(device, q, k, v, ctrl, chunk, p)[:, :, :S]
    ctrl_err = float((bad - exp).abs().max())
    rows = M.row_pccs(exp.reshape(B, -1), got.reshape(B, -1))
    print(
        f"[sdpa] {s_pad}-c{chunk}-{prec}-B{B}: worst row {float(rows.min()):.6f} pcc {pc:.6f} max_err {err:.4f} {chunk_diff} "
        f"control max_err {ctrl_err:.4f} lead {int((~kv[0, :S]).sum())} whole_chunks {int((~kv[0, :S]).sum()) // chunk}"
    )
    assert pc > 0.999 and err < 0.1 and float(rows.min()) > 0.999
    assert ctrl_err > 5 * err, "positive control passed: the mask is not being applied"


READ_CASES = [(c, s, k, B) for c, s, k in SHAPES[: len(df.BUCKETS)] for B in MODEL_BATCHES[s]]


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
@pytest.mark.parametrize("ctx_len,s_pad,chunk,B", READ_CASES, ids=[f"{s}-c{k}-B{B}" for _, s, k, B in READ_CASES])
def test_sdpa_readout(device, ctx_len, s_pad, chunk, prec, B):
    """The last layer's read-out call (tt/model.py layer(last=True)): 32 query rows (the future tokens, one q chunk)
    against every key, mask ``[B, 1, 32, s_pad]``, k chunk from the table. Same masks and control as above."""
    from models.experimental.tafsut_forecast.tt.model import NREAD

    p = PRECS[prec]
    P = ctx_len // df.PATCH
    g = torch.Generator().manual_seed(s_pad + chunk + B)
    q = torch.randn(B, NH, NREAD, DH, generator=g) * 0.6
    k, v = (torch.randn(B, NH, s_pad, DH, generator=g) * 0.6 for _ in range(2))
    kv = key_valid(ctx_len, s_pad, chunk, B)
    bias = torch.where(kv, 0.0, MASK_FILL)[:, None, None, :].expand(B, 1, NREAD, s_pad).contiguous()
    qb, kb, vb = (t.to(torch.bfloat16).float() for t in (q, k, v))
    exp = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=bias, scale=1.0)
    got = run_sdpa(device, q, k, v, bias, chunk, p, q_chunk=NREAD)
    assert torch.isfinite(got).all()
    pc, err = pcc(got, exp), float((got - exp).abs().max())
    rows = M.row_pccs(exp.reshape(B, -1), got.reshape(B, -1))
    ctrl = bias.clone()
    ctrl[0, :, :, (~kv[0]).nonzero().flatten()[: min(chunk, 32)]] = 0.0
    ctrl_err = float((run_sdpa(device, q, k, v, ctrl, chunk, p, q_chunk=NREAD) - exp).abs().max())
    print(
        f"[sdpa-read] {s_pad}-c{chunk}-{prec}-B{B} (P {P}): worst row {float(rows.min()):.6f} pcc {pc:.6f} "
        f"max_err {err:.4f} control max_err {ctrl_err:.4f}"
    )
    assert pc > 0.999 and err < 0.1 and float(rows.min()) > 0.999
    assert ctrl_err > 5 * err, "positive control passed: the mask is not being applied"


def test_sdpa_rejects_broadcast_mask(device, expect_error):
    """D0 amendment 4: ttnn SDPA needs the mask's query dim equal to Q's; a [B, 1, 1, S] mask is rejected."""
    q = torch.randn(1, NH, 128, DH)
    bias = torch.zeros(1, 1, 1, 128)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Mask sequence length must match Q sequence length"):
        ttnn.transformer.scaled_dot_product_attention(
            up(q), up(q), up(q), attn_mask=up(bias), is_causal=False, scale=1.0
        )


@pytest.mark.parametrize("s_pad,chunk", [(384, 128), (1088, 64)])
def test_sdpa_odd_chunk_count_bug(device, s_pad, chunk):
    """Regression record of the ttnn bug the chunk table avoids: an odd q-chunk count at B = 16 is wrong even
    unmasked. If this starts passing, the bug is fixed and the table can be revisited."""
    B = 16
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, NH, s_pad, DH, generator=g) * 0.6 for _ in range(3))
    bias = torch.zeros(B, 1, s_pad, s_pad)
    qb, kb, vb = (t.to(torch.bfloat16).float() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=bias, scale=1.0)
    out = run_sdpa(device, q, k, v, bias, chunk, Precision.accuracy())
    worst = float(M.row_pccs(ref.reshape(B, -1), out.reshape(B, -1)).min())
    print(f"[sdpa-bug] {s_pad}-c{chunk}-B{B}: worst row {worst:.4f}")
    assert worst < 0.999, "the odd-chunk-count SDPA bug is gone"
