# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.attn_residual.test.1): h_mid [S * 4, H] = post * attn_out + comb^T @ in, token-major (row 4s + n
is stream n of token s); post = attn_hc[:, 4:8], comb = attn_hc[:, 8:24] row-major 4x4, out stream m = post[m] *
attn_out + sum_n comb[n, m] in[n]. Both terms are about as large as the output (layer 0 norms: in 48.8, post term
49.4, comb term 48.8, out 42.9; layer 1: 34.2 / 26.6 / 25.5 / 36.0). At layer 0 the four streams are identical copies
of the embedding and comb columns sum to 1, so comb^T @ in = in whatever comb is: an identity comb or a comb read from
the wrong slot is invisible there. Measured on this golden (s4096 chunk 1, 2048 tokens), layer 0 PCC / rel L2 / norm
ratio (layer 1 rel): comb not transposed 0.9987 / 0.064 / [0.95, 1.11] (0.26), post x1.02 0.9998 / 0.023 (0.015),
comb x1.05 0.9987 / 0.057 (0.035), last token zeroed 0.99993 / 0.012 / [0, 1.005] (0.016), last 32 tokens zeroed
0.9928 / 0.12, identity comb 0.999996 / 0.0030 (layer 1 0.50): all pass PCC 0.99. Caught by PCC: post dropped,
reversed or from the pre slot, residual or attn_out dropped, stream-major rows, attn_out shifted a row.
Device noise: fp32 reference vs the bf16 golden rel 0.0032, worst row 0.010; bf16 products and partial sums 0.0048 /
[0.995, 1.005] / per-stream <= 0.0065 / worst row 0.014; bfp8 output 0.0082 (so the output must stay bf16 or better).
Extra checks, on layer 0 and on the same module (the step has no weights) on layer 1's golden, whose streams differ:
rel L2 <= 0.01, per-token norm ratio in [0.985, 1.015], per-stream rel L2 <= 0.015, worst row rel L2 <= 0.05, and
each term on its own against the golden inputs: delta = out - comb^T @ in vs post * attn_out and out - post * attn_out
vs comb^T @ in, projection coefficient in [0.98, 1.02] and rel L2 <= 0.02 (bf16 0.003; comb not transposed 0.056,
identity comb at layer 1 0.67).
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "attn_residual"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
DISTINCT_LAYER = 1  # a kda_dense layer whose streams differ (layer 0's are identical, hiding comb bugs)
N = 4  # hc_mult
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
RATIO = (0.985, 1.015)  # per-row ||got|| / ||want||
MAX_STREAM_REL = 0.015  # rel L2 over the rows of one stream
MAX_ROW_REL = 0.05  # worst row ||got - want|| / ||want||
TERM_COEF = (0.98, 1.02)  # <residual of the other term, term> / ||term||^2
MAX_TERM_REL = 0.02  # ||residual of the other term - term|| / ||term||


def _terms(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(post * attn_out, comb^T @ in) in fp32, [S * N, H] each."""
    h = x.shape[-1]
    post = hc[:, N : 2 * N].float()
    comb = hc[:, 2 * N :].float().reshape(-1, N, N)
    pterm = (post.unsqueeze(-1) * y.float().unsqueeze(-2)).reshape(-1, h)
    cterm = torch.matmul(comb.transpose(-1, -2), x.float().view(-1, N, h)).reshape(-1, h)
    return pterm, cterm


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor, inputs: list[torch.Tensor]) -> list[str]:
    assert out.numel() == want.numel(), f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    h = w.shape[-1]
    fails = []

    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)).max().item()
    gs, ws = got.view(-1, N, h), w.view(-1, N, h)
    srel = [((gs[:, n] - ws[:, n]).norm() / ws[:, n].norm()).item() for n in range(N)]
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"worst_row_rel_{STEP}_{tag}", row)
    metrics.record(f"worst_stream_rel_{STEP}_{tag}", max(srel))
    print(
        f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO}) "
        f"worst row {row:.4f} (<= {MAX_ROW_REL}) per-stream {[round(s, 5) for s in srel]} (<= {MAX_STREAM_REL})"
    )
    if rel > MAX_REL_L2:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {MAX_REL_L2}")
    if lo < RATIO[0] or hi > RATIO[1]:
        fails.append(f"{tag}: per-row norm ratio [{lo:.4f}, {hi:.4f}] outside {RATIO}")
    if row > MAX_ROW_REL:
        fails.append(f"{tag}: worst row rel L2 {row:.4f} > {MAX_ROW_REL}")
    if max(srel) > MAX_STREAM_REL:
        fails.append(f"{tag}: per-stream rel L2 {srel} > {MAX_STREAM_REL} (stream order or one stream wrong)")

    # Each term on its own: take the other (exact, from the golden inputs) out of the device output.
    x, hc, y = inputs
    pterm, cterm = _terms(x, hc, y)
    for name, term, other in (("post_term", pterm, cterm), ("comb_term", cterm, pterm)):
        d = got - other
        coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
        trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
        metrics.record(f"{name}_coef_{STEP}_{tag}", coef)
        metrics.record(f"{name}_rel_l2_{STEP}_{tag}", trel)
        print(f"{tag}: {name} coef={coef:.4f} (in {TERM_COEF}) rel={trel:.4f} (<= {MAX_TERM_REL})")
        if not TERM_COEF[0] <= coef <= TERM_COEF[1]:
            fails.append(f"{tag}: {name} coefficient {coef:.4f} outside {TERM_COEF}")
        if trel > MAX_TERM_REL:
            fails.append(f"{tag}: {name} rel L2 {trel:.4f} > {MAX_TERM_REL}")
    return fails


def _inputs(gl, st) -> list[torch.Tensor]:
    return [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    assert tuple(st.inputs) == ("in", "attn_hc", "attn_out"), f"unexpected step inputs {st.inputs}"
    gl = g.layer(c, LAYER)
    inputs = _inputs(gl, st)
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    fails = _checks(f"L{LAYER:02d}", out, want, inputs)

    # Same module on a layer with distinct streams (comb order and transposition are only visible there).
    assert S.block_type_of(DISTINCT_LAYER) == S.block_type_of(LAYER)
    gd = g.layer(c, DISTINCT_LAYER)
    inputs_d = _inputs(gd, st)
    out_d = fn(reference_ctx(ref, LAYER, g, c), device_ctx(DISTINCT_LAYER, g, c), *inputs_d)
    fails += _checks(f"L{DISTINCT_LAYER:02d}_distinct_streams", out_d, gd[st.output], inputs_d)
    assert not fails, "; ".join(fails)
