# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_residual of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.attn_residual.test.1): h_mid [S * 4, H] = post * attn_out + comb^T @ in, token-major (row 4s + n
is stream n of token s); post = attn_hc[:, 4:8], comb = attn_hc[:, 8:24] row-major 4x4, out stream m = post[m] *
attn_out + sum_n comb[n, m] in[n]. Same weightless module as kda_dense attn_residual. At layer 3 the streams already
differ, so comb bugs are visible here (no second-layer run). But post is saturated near 0 (<= 0.024), so the post
term is only 4.3% of the output (row norms: in 1.10, attn_out 16.5, post term 0.037, comb term 1.06, out 1.07).
Measured on this golden (s4096 chunk 1, 2048 tokens), PCC / rel L2 / norm ratio: attn_out dropped 0.99907 / 0.043,
post x0.5 0.99977 / 0.022, post x1.05 0.999995 / 0.0032 / max 1.039, post x1.02 0.999997 / 0.0026 / max 1.017,
comb not transposed 0.99975 / 0.023, identity comb 0.99944 / 0.041, comb x1.01 0.999997 / 0.010 / [1.002, 1.014],
comb x1.005 0.999997 / 0.0056, last token zeroed 0.99940 / 0.035, last row zeroed 0.999997 / 0.0025 / min 0, attn_out
shifted a row 0.99943 / 0.034, post reversed 0.9982 / 0.061: all pass PCC 0.99. Caught by PCC: post from the pre slot
(0.27), stream-major rows (0.08).
Noise vs the golden: the golden was made from the fp32 attn_hc, so the fp32 reference on the bf16 golden inputs is
already at rel 0.0024 / [0.9956, 1.0035] / worst row 0.0077 / per-stream 0.0029 (bf16 out 0.0028, all-bf16 mix 0.0030
/ worst row 0.0086 / per-stream 0.0040). Checks vs the golden: rel L2 <= 0.006, ratio [0.99, 1.01] (post x1.02 1.017,
comb x1.01 1.014), worst row <= 0.015 (post x1.02 0.019), per-stream <= 0.008 (post x1.02 0.013).
Checks vs the fp32 CPU step on the same golden inputs (removes the golden's floor; bf16 out 0.0015 / [0.9983, 1.0004]
/ worst row 0.0021, all-bf16 mix 0.0018 / [0.9977, 1.0009] / 0.0043, 0.3% noise 0.0030): rel <= 0.0035, ratio
[0.997, 1.003] (comb x1.002 1.0039, comb x1.005 1.0059, post x1.02 1.0147), worst row <= 0.008. Each term on its own
(out minus the exact other term, vs the term): comb term coefficient [0.998, 1.002] and rel <= 0.0035 (comb x1.005
1.005 / 0.0054; a truncating bf16 output 0.9970); post term coefficient [0.99, 1.01] (post x1.02 1.02) and rel
<= 0.08 (bf16 out 0.034, all-bf16 0.041, 0.3% noise 0.070; comb x1.005 0.12, comb not transposed 0.52). Limits are
written `not x <= lim` so NaN fails.
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
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
N = 4  # hc_mult
# vs the golden output
MAX_REL_L2 = 0.006  # ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-row ||got|| / ||want||
MAX_ROW_REL = 0.015  # worst row ||got - want|| / ||want||
MAX_STREAM_REL = 0.008  # rel L2 over the rows of one stream
# vs the fp32 CPU step on the same golden inputs
CPU_MAX_REL_L2 = 0.0035
CPU_RATIO = (0.997, 1.003)
CPU_MAX_ROW_REL = 0.008
COMB_COEF = (0.998, 1.002)  # <out - post term, comb term> / ||comb term||^2
COMB_MAX_REL = 0.0035  # ||out - post term - comb term|| / ||comb term||
POST_COEF = (0.99, 1.01)  # <out - comb term, post term> / ||post term||^2
POST_MAX_REL = 0.08  # ||out - comb term - post term|| / ||post term|| (bf16 output rounding alone gives 0.034)


def _terms(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(post * attn_out, comb^T @ in) in fp32, [S * N, H] each."""
    h = x.shape[-1]
    post = hc[:, N : 2 * N].float()
    comb = hc[:, 2 * N :].float().reshape(-1, N, N)
    pterm = (post.unsqueeze(-1) * y.float().unsqueeze(-2)).reshape(-1, h)
    cterm = torch.matmul(comb.transpose(-1, -2), x.float().view(-1, N, h)).reshape(-1, h)
    return pterm, cterm


def _within(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi  # False on NaN


def _whole(tag: str, got: torch.Tensor, w: torch.Tensor, max_rel, ratio, max_row, max_stream=None) -> list[str]:
    h = w.shape[-1]
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)).max().item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"worst_row_rel_{STEP}_{tag}", row)
    msg = (
        f"{tag}: rel_l2={rel:.5f} (<= {max_rel}) norm ratio [{lo:.4f}, {hi:.4f}] (in {ratio}) "
        f"worst row {row:.4f} (<= {max_row})"
    )
    fails = []
    if not rel <= max_rel:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {max_rel}")
    if not (_within(lo, *ratio) and _within(hi, *ratio)):
        fails.append(f"{tag}: per-row norm ratio [{lo:.4f}, {hi:.4f}] outside {ratio}")
    if not row <= max_row:
        fails.append(f"{tag}: worst row rel L2 {row:.4f} > {max_row}")
    if max_stream is not None:
        gs, ws = got.view(-1, N, h), w.view(-1, N, h)
        srel = [((gs[:, n] - ws[:, n]).norm() / ws[:, n].norm()).item() for n in range(N)]
        metrics.record(f"worst_stream_rel_{STEP}_{tag}", max(srel))
        msg += f" per-stream {[round(s, 5) for s in srel]} (<= {max_stream})"
        if not max(srel) <= max_stream:
            fails.append(f"{tag}: per-stream rel L2 {srel} > {max_stream} (stream order or one stream wrong)")
    print(msg)
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


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    assert tuple(st.inputs) == ("in", "attn_hc", "attn_out"), f"unexpected step inputs {st.inputs}"
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
    tag = f"L{LAYER:02d}"
    fails = _whole(tag, got, want.float(), MAX_REL_L2, RATIO, MAX_ROW_REL, MAX_STREAM_REL)

    # Sharp checks: the fp32 CPU step on the same golden inputs, and each term on its own.
    cpu = ref.component(LAYER, STEP)(reference_ctx(ref, LAYER, g, c), *inputs).float().reshape(want.shape)
    fails += _whole(f"{tag}_vs_cpu", got, cpu, CPU_MAX_REL_L2, CPU_RATIO, CPU_MAX_ROW_REL)
    fails += _term_checks(tag, got, inputs)
    assert not fails, "; ".join(fails)
