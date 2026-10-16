# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type swa_engram (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Review (C.swa_engram.experts test 1): the same step as swa layer 0, reviewed the same way (test_c_swa_experts.py,
DS41-22). The golden is the fp32 reference's output stored in bf16, its inputs (ffn_norm, dense router weights) bf16
too, and the experts quantize x and h to fp8 around fp4 GEMMs, so the fp32 CPU step on the golden inputs is itself
rel 0.0203, worst row 0.0539, row norm ratio [0.971, 1.016] from the golden (layer 1). The built-in checks add that
error to the vs-golden rel and column limits only; here also to the worst-row and row-ratio limits (vs the golden
only: 0.0724 -> 0.126, 0.0536 -> 0.083), since a correct device's own worst row (layer 0: 0.039 vs the CPU) on top
of the golden's 0.054 would leave no margin. The tight checks stay as built in: vs the CPU step on the same inputs
(rel 0.0175, worst row 0.0724, ratio 0.0536, columns, bias) and the second inputs (layer24, mixed, small, big).
1 % output noise is not caught by limits: the bf16 precision model's error vs the CPU step is larger (rel 0.0146 vs
0.0100, worst row 0.060 vs 0.030), and the output (an fp32 sum of 6 experts' GEMMs) has no grid or subspace to check
(6 x 2304 expert columns span the 5120-dim hidden). Owner rule: accuracy against the noise floor.

Review (C.swa_engram.experts test, attempt 1): the freeze sweep let noise1e-2 through, as on layer 0. Added layer 0's
row-scaling check (DS41-22, test_c_swa_experts.py) on every input, vs the CPU step (evaluate below):
    row scaling   floating-point error is relative: a row's error grows with the row. Fit log ||got_i - cpu_i|| =
                  a + b log ||cpu_i|| over the rows; b >= ROW_SLOPE_MIN. Measured values: BREADCRUMBS. Skipped when
                  fewer than ROW_SLOPE_ROWS rows differ (an exact module has no error to fit).
The sweep (BRINGUP_IMPL=mutations, CC.sweep) calls this evaluate, so it runs with the check added.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
STEP = "experts"
LAYER = 1
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)
ROW_SLOPE_MIN = 0.35  # slope of log row error on log row norm vs the CPU step (bf16 model >= 0.69, 1 % noise ~0)
ROW_SLOPE_ROWS = 32  # fewer rows with an error than this: nothing to fit, the check passes


def row_slope(got: torch.Tensor, ref: torch.Tensor) -> tuple[float | None, int]:
    """(b, rows) of the least-squares fit log ||got_i - ref_i|| = a + b log ||ref_i|| over the rows i where both are
    nonzero (float64); b None when the shapes differ or a value is not finite."""
    if got.shape != ref.shape:
        return None, 0
    g, r = got.double().reshape(-1, ref.shape[-1]), ref.double().reshape(-1, ref.shape[-1])
    if not bool(torch.isfinite(g).all()):
        return None, 0
    en, rn = (g - r).norm(dim=1), r.norm(dim=1)
    ok = (en > 0) & (rn > 0)
    n = int(ok.sum().item())
    if n < ROW_SLOPE_ROWS:
        return float("nan"), n
    x, y = rn[ok].log(), en[ok].log()
    x, y = x - x.mean(), y - y.mean()
    return ((x * y).sum() / (x * x).sum().clamp_min(1e-300)).item(), n


class _Expect(CC.Expect):
    """CC.Expect whose vs-golden worst-row and row-norm-ratio limits also add the fp32 CPU step's own error vs the
    bf16 golden (as CC.Expect.check already does for the rel and column limits)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.golden_fe = CC.float_errors(self.cpu["golden"], self.want) if self.kind == "float" else None

    def check(self, tag, got, ref, case, vs_golden, quiet):
        if not (vs_golden and self.kind == "float"):
            return super().check(tag, got, ref, case, vs_golden, quiet)
        rl = self.rel_lim[case.name] + self.golden_err
        cl = self.col_lim[case.name]
        if cl is not None and self.golden_col_err is not None:
            cl = cl + self.golden_col_err
        if cl is not None:
            cl = cl.clamp_min(self.lim["component_col_factor"] * rl)
        L = CC.float_limits(self.lim, 1.0, rl, self.model_err[case.name], False, cl)
        ge = self.golden_fe
        L["row"] += ge["row"]
        L["ratio"] += max(1 - ge["ratio_min"], ge["ratio_max"] - 1)
        e, bad = CC.float_fails(got, ref, L)
        if not quiet:
            for k, v in e.items():
                CC.metrics.record(f"{tag}_{k}", v)
            nums = " ".join(f"{k}={v:.6g}" for k, v in e.items())
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: {nums}" + (f" ({'; '.join(bad)})" if bad else ""))
        return not bad

    def evaluate(self, outs: dict, quiet=False, stop_early=False) -> list[str]:
        """The built-in checks plus the row-scaling check on every input, vs the CPU step on the same inputs."""
        failed = super().evaluate(outs, quiet=quiet, stop_early=stop_early)
        if self.kind != "float" or (failed and stop_early):
            return failed
        for case in self.cases:
            got, tag = outs[case.name], f"rowscale_{self.st.name}_L{self.layer:02d}_{case.name}"
            if isinstance(got, Exception):
                continue  # the built-in checks already failed this input
            b, n = row_slope(got, self.cpu[case.name])
            bad = None
            if b is None:
                bad = f"shape {tuple(got.shape)} or non-finite values, want {tuple(self.cpu[case.name].shape)}"
            elif n >= ROW_SLOPE_ROWS and not b >= ROW_SLOPE_MIN:
                bad = f"row error does not scale with the row: slope {b:.3f} < {ROW_SLOPE_MIN} (added noise?)"
            if bad:
                failed.append(tag)
            if not quiet:
                if b is not None and n >= ROW_SLOPE_ROWS:
                    metrics.record(tag, b)
                print(f"{'ok  ' if not bad else 'FAIL'} {tag}: slope={b} rows={n}" + (f" ({bad})" if bad else ""))
            if failed and stop_early:
                break
        return failed


@mesh_parametrize
def test_component(mesh_device):
    base = CC.Expect
    CC.Expect = _Expect  # testing/component._run_auto builds its Expect through CC.Expect
    try:
        assert run_component_test(S, STEP, LAYER, mesh_device, COMPARE, THRESHOLD, checks=CHECKS)
    finally:
        CC.Expect = base
