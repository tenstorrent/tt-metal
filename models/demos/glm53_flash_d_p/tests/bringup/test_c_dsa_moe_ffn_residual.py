# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_residual of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.ffn_residual.test.1): out [S * 4, H] = post * mlp_out + comb^T @ h_mid, token-major (row 4s + n
is stream n of token s); post = ffn_hc[:, 4:8], comb = ffn_hc[:, 8:24] row-major 4x4, out stream m = post[m] *
mlp_out + sum_n comb[n, m] h_mid[n]. Same weightless module as attn_residual. At layer 3 the streams differ, so comb
bugs show here (no second-layer run). Unlike attn_residual, ffn post is not saturated (up to 0.40) and mlp_out is
large, so the post term is 0.70 of the output (row norms: h_mid 1.07, mlp_out 59.7, post term 1.32, comb term 0.99,
out 2.10).
Measured on this golden (s4096 chunk 1, 2048 tokens), PCC / rel L2 / norm ratio: post x1.02 0.99998 / 0.014 / max
1.019, post x1.05 0.99992 / 0.035, comb not transposed 0.99947 / 0.033 / per-stream 0.997, identity comb 0.99982 /
0.033, comb x1.01 0.99999 / 0.0050 / max 1.013, comb x1.005 0.999996 / 0.0033, h_mid x1.01 = comb x1.01, last token
zeroed 0.99985 / 0.018, last row zeroed 0.999997 / 0.0025 / min 0, last 32 tokens zeroed 0.9927 / 0.12, post x0.5
0.9822 (fails PCC): all but the last pass PCC 0.99. Caught by PCC: post from the pre slot, post reversed, stream-major
rows, mlp_out or h_mid dropped, mlp_out shifted a row.
Noise vs the golden: the fp32 reference on the bf16 golden inputs is at rel 0.0025 / [0.9970, 1.0030] / worst row
0.0040 / per-stream 0.0027 (bf16 out 0.0029 / row 0.0044 / stream 0.0031; all-bf16 mix 0.0036 / [0.9960, 1.0029] /
0.0051 / 0.0040; 0.3% noise 0.0039 / 0.0050 / 0.0040). Checks vs the golden: rel L2 <= 0.006, ratio [0.993, 1.007]
(post x1.01 1.009, comb x1.01 1.013), worst row <= 0.01 (post x1.02 0.020, comb x1.01 0.013), per-stream <= 0.006
(last row zeroed 0.0091, comb x1.01 0.010).
Checks vs the fp32 CPU step on the same golden inputs (removes the golden's floor; bf16 out 0.0017 / [0.9998, 1.0004]
/ row 0.0018 / stream 0.0017, all-bf16 mix 0.0026 / [0.9978, 1.0008] / 0.0034 / 0.0029, 0.3% noise 0.0030): rel
<= 0.0035 (post x1.005 0.0039), ratio [0.997, 1.003] (post x1.005 1.0045, comb x1.005 1.0058, truncating bf16 out
0.9965), worst row <= 0.006, per-stream <= 0.005 (last row zeroed 0.0088). Each term on its own (out minus the exact
other term, vs the term): post term coefficient [0.998, 1.002] (post x1.005 1.005, truncating out 0.9963) and rel
<= 0.007 (bf16 out 0.0024, all-bf16 0.0037, 0.3% noise 0.0043; post x1.01 0.010); comb term coefficient [0.998,
1.002] (comb x1.005 1.005, post x1.005 1.0042) and rel <= 0.009 (bf16 out 0.0038, all-bf16 0.0060, 0.3% noise
0.0069; comb x1.01 0.011). Limits are written `not x <= lim` so NaN fails.
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
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
# vs the golden output
MAX_REL_L2 = 0.006  # ||got - want|| / ||want||
RATIO = (0.993, 1.007)  # per-row ||got|| / ||want||
MAX_ROW_REL = 0.01  # worst row ||got - want|| / ||want||
MAX_STREAM_REL = 0.006  # rel L2 over the rows of one stream
# vs the fp32 CPU step on the same golden inputs
CPU_MAX_REL_L2 = 0.0035
CPU_RATIO = (0.997, 1.003)
CPU_MAX_ROW_REL = 0.006
CPU_MAX_STREAM_REL = 0.005
POST_COEF = (0.998, 1.002)  # <out - comb term, post term> / ||post term||^2
POST_MAX_REL = 0.007  # ||out - comb term - post term|| / ||post term||
COMB_COEF = (0.998, 1.002)  # <out - post term, comb term> / ||comb term||^2
COMB_MAX_REL = 0.009  # ||out - post term - comb term|| / ||comb term||


def _terms(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(post * mlp_out, comb^T @ h_mid) in fp32, [S * N, H] each."""
    h = x.shape[-1]
    post = hc[:, N : 2 * N].float()
    comb = hc[:, 2 * N :].float().reshape(-1, N, N)
    pterm = (post.unsqueeze(-1) * y.float().unsqueeze(-2)).reshape(-1, h)
    cterm = torch.matmul(comb.transpose(-1, -2), x.float().view(-1, N, h)).reshape(-1, h)
    return pterm, cterm


def _within(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi  # False on NaN


def _whole(tag: str, got: torch.Tensor, w: torch.Tensor, max_rel, ratio, max_row, max_stream) -> list[str]:
    h = w.shape[-1]
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
        f"{tag}: rel_l2={rel:.5f} (<= {max_rel}) norm ratio [{lo:.4f}, {hi:.4f}] (in {ratio}) "
        f"worst row {row:.4f} (<= {max_row}) per-stream {[round(s, 5) for s in srel]} (<= {max_stream})"
    )
    fails = []
    if not rel <= max_rel:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {max_rel}")
    if not (_within(lo, *ratio) and _within(hi, *ratio)):
        fails.append(f"{tag}: per-row norm ratio [{lo:.4f}, {hi:.4f}] outside {ratio}")
    if not row <= max_row:
        fails.append(f"{tag}: worst row rel L2 {row:.4f} > {max_row}")
    if not max(srel) <= max_stream:
        fails.append(f"{tag}: per-stream rel L2 {srel} > {max_stream} (stream order or one stream wrong)")
    return fails


def _term_checks(tag: str, got: torch.Tensor, inputs: list[torch.Tensor]) -> list[str]:
    """Each term on its own: take the other (exact, from the golden inputs) out of the output."""
    pterm, cterm = _terms(*inputs)
    fails = []
    for name, term, other, (clo, chi), max_rel in (
        ("post_term", pterm, cterm, POST_COEF, POST_MAX_REL),
        ("comb_term", cterm, pterm, COMB_COEF, COMB_MAX_REL),
    ):
        d = got - other
        coef = ((d * term).sum() / (term * term).sum().clamp_min(1e-30)).item()
        trel = ((d - term).norm() / term.norm().clamp_min(1e-30)).item()
        metrics.record(f"{name}_coef_{STEP}_{tag}", coef)
        metrics.record(f"{name}_rel_l2_{STEP}_{tag}", trel)
        print(f"{tag}: {name} coef={coef:.5f} (in [{clo}, {chi}]) rel={trel:.5f} (<= {max_rel})")
        if not _within(coef, clo, chi):
            fails.append(f"{tag}: {name} coefficient {coef:.5f} outside [{clo}, {chi}]")
        if not trel <= max_rel:
            fails.append(f"{tag}: {name} rel L2 {trel:.4f} > {max_rel}")
    return fails


def _checks(
    tag: str, got: torch.Tensor, want: torch.Tensor, cpu: torch.Tensor, inputs: list[torch.Tensor]
) -> list[str]:
    fails = _whole(tag, got, want.float(), MAX_REL_L2, RATIO, MAX_ROW_REL, MAX_STREAM_REL)
    fails += _whole(f"{tag}_vs_cpu", got, cpu, CPU_MAX_REL_L2, CPU_RATIO, CPU_MAX_ROW_REL, CPU_MAX_STREAM_REL)
    fails += _term_checks(tag, got, inputs)
    return fails


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    assert tuple(st.inputs) == ("h_mid", "ffn_hc", "mlp_out"), f"unexpected step inputs {st.inputs}"
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    assert torch.isfinite(got).all(), "non-finite output"

    # Sharp checks: the fp32 CPU step on the same golden inputs, and each term on its own.
    cpu = ref.component(LAYER, STEP)(reference_ctx(ref, LAYER, g, c), *inputs).float().reshape(want.shape)
    fails = _checks(f"L{LAYER:02d}", got, want, cpu, inputs)
    assert not fails, "; ".join(fails)
