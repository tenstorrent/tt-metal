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
from models.experimental.tafsut_forecast.tt.common import MASK_FILL, Precision, ckc

NH, DH = df.NH, df.DH
# (bucket context, s_pad, chunk): the table, plus the alternatives D0 Q2 / CONTRACT §7 item 1 measure.
SHAPES = [(b, *df.PAD_TABLE[b]) for b in df.BUCKETS] + [(3072, 256, 128), (32768, 1152, 128)]
PRECS = {
    "accuracy": Precision.accuracy(),
    "performance": Precision.performance(),
    "accuracy_exp_approx": dataclasses.replace(Precision.accuracy(), exp_approx=True),
}


def key_valid(ctx_len: int, s_pad: int, chunk: int) -> torch.Tensor:
    """``[2, s_pad]``: row 0 with a leading masked run of 2 chunks + 17 keys, or every context key when the context
    is shorter (the all-NaN row: REG and future stay valid, so this is the most whole chunks the length allows);
    row 1 with one masked interior patch; both with the pad tail masked."""
    S = ctx_len // df.PATCH + df.NFUT + 1
    P = ctx_len // df.PATCH
    kv = torch.zeros(2, s_pad, dtype=torch.bool)
    kv[:, :S] = True
    lead = min(2 * chunk + 17, P)
    kv[0, :lead] = False
    kv[1, 5] = False
    return kv


def run_sdpa(device, q, k, v, bias, chunk, prec: Precision):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=prec.exp_approx,
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        up(q), up(k), up(v), attn_mask=up(bias), is_causal=False, scale=1.0, program_config=pc,
        compute_kernel_config=ckc(prec.sdpa_fidelity),
    )
    return ttnn.to_torch(out).float()


@pytest.mark.parametrize("prec", list(PRECS))
@pytest.mark.parametrize("ctx_len,s_pad,chunk", SHAPES, ids=[f"{s}-c{c}" for _, s, c in SHAPES])
def test_sdpa_masked(device, ctx_len, s_pad, chunk, prec):
    p = PRECS[prec]
    g = torch.Generator().manual_seed(s_pad + chunk)
    q, k, v = (torch.randn(2, NH, s_pad, DH, generator=g) * 0.6 for _ in range(3))  # logit std ~2.9 at scale 1.0
    kv = key_valid(ctx_len, s_pad, chunk)
    bias = torch.where(kv, 0.0, MASK_FILL)[:, None, None, :].expand(2, 1, s_pad, s_pad).contiguous()
    qb, kb, vb = (t.to(torch.bfloat16).float() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=bias, scale=1.0)
    out = run_sdpa(device, q, k, v, bias, chunk, p)
    assert torch.isfinite(out).all(), "non-finite SDPA output (padded rows included)"
    S = ctx_len // df.PATCH + df.NFUT + 1
    got, exp = out[:, :, :S], ref[:, :, :S]
    pc = pcc(got, exp)
    err = float((got - exp).abs().max())
    big = max(c for c in (32, 64, 128, 256, 512) if s_pad % c == 0)
    forced = run_sdpa(device, q, k, v, bias, big, p)[:, :, :S]
    capped = float((got - forced).abs().max())
    # positive control (v1.13): drop the mask on one whole masked key block of row 0
    ctrl = bias.clone()
    blk = (~kv[0]).nonzero().flatten()[: min(chunk, 32)]
    ctrl[0, :, :, blk] = 0.0
    bad = run_sdpa(device, q, k, v, ctrl, chunk, p)[:, :, :S]
    ctrl_err = float((bad - exp).abs().max())
    print(f"[sdpa] {s_pad}-c{chunk}-{prec}: pcc {pc:.6f} max_err {err:.4f} |capped-forced{big}| {capped:.4f} "
          f"control max_err {ctrl_err:.4f} lead {int((~kv[0, :S]).sum())} whole_chunks {int((~kv[0, :S]).sum()) // chunk}")
    assert pc > 0.999 and err < 0.1
    assert ctrl_err > 5 * err, "positive control passed: the mask is not being applied"


def test_sdpa_rejects_broadcast_mask(device):
    """D0 amendment 4: ttnn SDPA needs the mask's query dim equal to Q's; a [B, 1, 1, S] mask is rejected."""
    q = torch.randn(1, NH, 128, DH)
    bias = torch.zeros(1, 1, 1, 128)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(Exception):
        ttnn.transformer.scaled_dot_product_attention(up(q), up(q), up(q), attn_mask=up(bias), is_causal=False,
                                                      scale=1.0)
