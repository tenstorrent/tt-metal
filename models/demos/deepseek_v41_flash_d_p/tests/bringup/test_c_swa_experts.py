# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: experts of block type swa (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Review (C.swa.experts test 1): the built-in checks, with one change on the vs-golden comparison. The golden is the
fp32 reference's output stored in bf16, and its inputs (ffn_norm, dense router weights) are stored in bf16 too. The
experts quantize x and h to fp8 (blocks of 32) around fp4 GEMMs, so the input rounding alone flips whole e4m3 steps
(DS41-1, DS41-14): the fp32 CPU step on the golden inputs is rel 0.0169, worst row 0.0798, row norm ratio
[0.976, 1.058] from the golden, and inputs jittered by half a bf16 ulp move it as far (rel 0.0168, worst row
0.054-0.075, ratio down to 0.945). The built-in checks add that error of the CPU step itself to the vs-golden rel and
column limits, not to the worst-row and row-ratio limits, so the CPU reference failed the test. Here they are added
the same way (vs the golden only). The tight checks stay as built in: vs the CPU step on the same inputs (rel, worst
row, ratios, columns, bias at the precision model's limits) and the second inputs (layer24, mixed, small, big).
"""

from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
STEP = "experts"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
CHECKS = "auto"  # also the built-in checks by output kind and the second inputs (testing/component_checks.py)


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


@mesh_parametrize
def test_component(mesh_device):
    base = CC.Expect
    CC.Expect = _Expect  # testing/component._run_auto builds its Expect through CC.Expect
    try:
        assert run_component_test(S, STEP, LAYER, mesh_device, COMPARE, THRESHOLD, checks=CHECKS)
    finally:
        CC.Expect = base
