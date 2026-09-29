# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.ffn_residual.test.1): out [S * 4, H] = post * mlp_out + comb^T @ h_mid, token-major (row 4s + n
is stream n of token s); post = ffn_hc[:, 4:8], comb = ffn_hc[:, 8:24] row-major 4x4, out stream m = post[m] *
mlp_out + sum_n comb[n, m] h_mid[n]. Same math and module as attn_residual, but h_mid's streams already differ at
layer 0, so comb order is visible there. Norms (layer 0 / layer 1): h_mid 42.9 / 36.0, post term 27.2 / 10.0, comb
term 40.6 / 34.8, out 34.2 / 33.3. Measured on this golden (s4096 chunk 1, 2048 tokens), layer 0 PCC / rel L2 / norm
ratio (layer 1 rel): comb not transposed 0.9951 / 0.098 / [0.59, 1.39] (0.038), post x1.02 0.99987 / 0.016 /
[0.989, 1.036] (0.0066), post x1.01 0.99996 / 0.0087 / [0.9925, 1.0205] (0.0041), comb x1.02 0.99988 / 0.024 (0.021),
mlp_out x1.01 = post x1.01, last token zeroed 0.99993 / 0.012 / [0, 1.005], last row zeroed 0.99998 / 0.0062: all pass
PCC 0.99. Caught by PCC: identity comb (0.974), post from the pre slot or reversed, mlp_out dropped or halved,
stream-major rows, mlp_out shifted a row.
Device noise: fp32 reference vs the bf16 golden rel 0.0034 / [0.9949, 1.0052] / worst row 0.0095 / per-stream 0.0045
(the golden's own bf16 rounding sets this floor); bf16 inputs, fp32 mix, bf16 out 0.0037; all-bf16 mix 0.0044 /
[0.9945, 1.0053] / worst row 0.0108 / per-stream 0.0057 / term rel <= 0.0082 (layer 1); 7-bit-mantissa output 0.0047 /
[0.9921, 1.0023] / term rel 0.011 / coef 0.9974.
Extra checks, on layer 0 and on the same module (the step has no weights) on layer 1's golden: rel L2 <= 0.01,
per-token norm ratio in [0.99, 1.01] (post x1.01 1.0205), per-stream rel L2 <= 0.012, worst row rel L2 <= 0.03 (post
x1.02 0.053, x1.01 0.031), and each term on its own against the golden inputs: out - comb^T @ h_mid vs post * mlp_out
and out - post * mlp_out vs comb^T @ h_mid, projection coefficient in [0.99, 1.01] and rel L2 <= 0.015 (comb x1.02
0.020 / 0.030, post x1.02 0.020, comb not transposed 0.083 / 0.12).
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
STEP = "ffn_residual"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
DISTINCT_LAYER = 1  # a second kda_dense layer: the weightless module must not depend on the layer
N = 4  # hc_mult
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want||
MAX_STREAM_REL = 0.012  # rel L2 over the rows of one stream
MAX_ROW_REL = 0.03  # worst row ||got - want|| / ||want||
TERM_COEF = (0.99, 1.01)  # <residual of the other term, term> / ||term||^2
MAX_TERM_REL = 0.015  # ||residual of the other term - term|| / ||term||


def _terms(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(post * mlp_out, comb^T @ h_mid) in fp32, [S * N, H] each."""
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
    assert tuple(st.inputs) == ("h_mid", "ffn_hc", "mlp_out"), f"unexpected step inputs {st.inputs}"
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

    # Same (weightless) module on layer 1's golden: another set of streams and coefficients.
    assert S.block_type_of(DISTINCT_LAYER) == S.block_type_of(LAYER)
    gd = g.layer(c, DISTINCT_LAYER)
    inputs_d = _inputs(gd, st)
    out_d = fn(reference_ctx(ref, LAYER, g, c), device_ctx(DISTINCT_LAYER, g, c), *inputs_d)
    fails += _checks(f"L{DISTINCT_LAYER:02d}", out_d, gd[st.output], inputs_d)
    assert not fails, "; ".join(fails)
