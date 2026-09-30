# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: moe_add of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.moe_add.test.1): mlp_out [S, H] = experts_out + shared_out, replicated, no weights. Row norms on
this golden (s4096 chunk 1, 2048 tokens): experts 47.6, shared 22.1, out 59.7; shared / experts per row 0.16..4.5.
Measured, PCC / rel L2 vs golden / per-row norm ratio: (a + b) / 2 0.999997 / 0.50 / 0.50, shared x1.01 0.999993 /
0.0044 / max 1.009, experts x1.01 0.999993 / 0.0086, last row zeroed 0.999982 / 0.0060 / min 0, shared missing on the
last 32 rows 0.9989 / 0.048 / min 0.22, one 1024-column shard x1.01 0.999988 / 0.0055: all pass PCC 0.99. Caught by
PCC: a dropped addend (0.93 / 0.60), a - b (0.68), shared shifted a row (0.89), one shard of shared missing (0.98).
Noise: the golden was made from fp32 experts / shared outputs, so the fp32 sum of the bf16 golden inputs is already
at rel 0.0022 / [0.9998, 1.0002] / worst row 0.0024 vs the golden (bf16 RNE output 0.0028, truncating bf16 output
0.0037 / [0.9974, 0.9984] / 0.0040). Checks vs the golden: rel <= 0.005, ratio [0.995, 1.005], worst row <= 0.01
(last row zeroed 1.0, (a + b) / 2 0.50).
Checks vs the fp32 sum of the same golden inputs (bf16 out 0.0018 / [0.9998, 1.0001] / 0.0019, truncating
0.0028 / [0.9974, 0.9983] / 0.0031): rel <= 0.0035 (experts x1.005 0.0042), ratio [0.997, 1.003] (shared x1.005
1.0045, experts x1.005 1.0048), worst row <= 0.006 (shared x1.01 0.0093, last 32 rows' shared missing 0.85). Each
addend on its own (out minus the exact other addend, vs the addend): experts coefficient [0.997, 1.003] and rel
<= 0.004 (truncating 0.9975 / 0.0034; experts x1.005 1.005 / 0.005); shared coefficient [0.995, 1.005] and rel
<= 0.009 (truncating 0.9967 / 0.0075; shared x1.01 1.010 / 0.010). Limits are written `not x <= lim` so NaN fails.
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
STEP = "moe_add"
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
# vs the golden output
MAX_REL_L2 = 0.005  # ||got - want|| / ||want||
RATIO = (0.995, 1.005)  # per-row ||got|| / ||want||
MAX_ROW_REL = 0.01  # worst row ||got - want|| / ||want||
# vs the fp32 sum of the same golden inputs
CPU_MAX_REL_L2 = 0.0035
CPU_RATIO = (0.997, 1.003)
CPU_MAX_ROW_REL = 0.006
EXPERTS_COEF = (0.997, 1.003)  # <out - shared, experts> / ||experts||^2
EXPERTS_MAX_REL = 0.004  # ||out - shared - experts|| / ||experts||
SHARED_COEF = (0.995, 1.005)  # <out - experts, shared> / ||shared||^2
SHARED_MAX_REL = 0.009  # ||out - experts - shared|| / ||shared||


def _within(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi  # False on NaN


def _whole(tag: str, got: torch.Tensor, w: torch.Tensor, max_rel, ratio, max_row) -> list[str]:
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)
    lo, hi = r.min().item(), r.max().item()
    row = ((got - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)).max().item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"worst_row_rel_{STEP}_{tag}", row)
    print(
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
    return fails


def _addend_checks(tag: str, got: torch.Tensor, experts: torch.Tensor, shared: torch.Tensor) -> list[str]:
    """Each addend on its own: take the other (exact, from the golden inputs) out of the output."""
    fails = []
    for name, term, other, (clo, chi), max_rel in (
        ("experts", experts, shared, EXPERTS_COEF, EXPERTS_MAX_REL),
        ("shared", shared, experts, SHARED_COEF, SHARED_MAX_REL),
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
    assert tuple(st.inputs) == ("experts_out", "shared_out"), f"unexpected step inputs {st.inputs}"
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
    fails = _whole(tag, got, want.float(), MAX_REL_L2, RATIO, MAX_ROW_REL)

    # Sharp checks: the fp32 sum of the same golden inputs, and each addend on its own.
    experts, shared = inputs
    fails += _whole(f"{tag}_vs_cpu", got, experts + shared, CPU_MAX_REL_L2, CPU_RATIO, CPU_MAX_ROW_REL)
    fails += _addend_checks(tag, got, experts, shared)
    assert not fails, "; ".join(fails)
