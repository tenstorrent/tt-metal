# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 15: block type swa_engram (layer 1) with experts swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (swa_engram) with these steps on the device and the rest on the CPU reference:
    engram_rows
    engram_kv
    engram_add
    attn_hc
    attn_collapse
    attn_norm
    q_a
    kv
    attention
    attn_residual
    ffn_hc
    ffn_collapse
    ffn_norm
    router
    experts

Reviewed (S.swa_engram.15.test.1). Two changes to the rendered test: test 14's router checks, carried over
unchanged (the rendered test failed on the CPU reference only at swap_router_vs_golden, PCC 0.984586 < 0.99,
DS41-32; test 14's review below), and the experts' checks (DS41-33):

- experts vs golden: the CPU reference is PCC 0.990793 (device 0.990515) against the fixed 0.99, because 12.4 % of
  the rows run another top-6 set than the golden (the router flips above) and so compute another function. Now as
  the router: PCC over all rows >= 0.98 (a loose bound, 2 x the CPU chain's 1 - 0.9908), and on the rows whose
  routing (the swapped router's output, the experts' input) keeps the golden's experts rel, worst row, row norm ratio
  (2 x the CPU chain's) and median ratio bias (swap_step_bias 0.01).
- experts vs the CPU step on the same inputs: the framework's fixed worst row 0.05 and row norm ratio 1 +- 0.02
  failed the bf16 control (row 0.0469, ratio [0.9659, 1.0129]); now 2 x the control (EXPERTS_CPU_LIMITS), the
  framework's rel and bias kept. 1 % noise passed every check (rel 0.0100 < 0.058); added the experts component
  test's row-scaling check (DS41-22): slope of log row error on log row norm vs the CPU step >= 0.35.

Measured (layer 1, the component rung's last dumped chunk, 2048 rows; mutate:<kind> with BRINGUP_MUTATE_STEP=experts;
"golden" = vs the golden on the rows that keep the golden's experts; "cpu" = vs the CPU step on the same inputs):

    variant          golden: pcc rel    row    ratio          bias     | cpu: rel row    ratio          slope | test
    reference (fp32) 0.9908      0.0450 0.0770 [0.954, 1.036] -1.3e-4  | 0        0      [1, 1]         -     | pass
    bf16 (control)   0.9908      0.0454 0.0780 [0.953, 1.020] -6.3e-4  | 0.0288   0.0469 [0.966, 1.013] 0.61  | pass
    device (2x2)     0.9905      0.0452 0.0852 [0.978, 1.026] +2.2e-4  | 0.0024   0.0137 [0.999, 1.002] 0.71  | pass
    stub (zeros)     0           -      -      -              -        | 1.0      1.0    [0, 0]         -     | FAIL
    noise1e-2        0.9907      0.0460 0.0772 [0.954, 1.036] -7.8e-5  | 0.0100   0.0183 [0.999, 1.001] 0.00  | FAIL *
    scale1.02 / 0.98 0.9908      0.049  0.081  [0.973, 1.057] +/-0.020 | 0.0200   0.0200 [1.02, 1.02]   1.00  | FAIL
    halfswap         -0.0016     1.41   1.46   [0.954, 1.036] -1.3e-4  | 1.42     1.46   [1, 1]         1.00  | FAIL
    rowshift         0.1012      1.34   2.73   [0.309, 2.581] -2.8e-3  | 1.34     2.72   [0.309, 2.576] 0.56  | FAIL
    quarterzero      0.8506      0.513  1.0    [0, 1.020]     -1.9e-3  | 0.513    1.0    [0, 1]         1.00  | FAIL
    sign             0.7345      0.720  0.861  [0.954, 1.036] -1.3e-4  | 0.719    0.859  [1, 1]         1.05  | FAIL

bf16 = mutate:bf16; the mistakes are mutate:<kind>. Rows that keep the golden's experts: 87.6 % (reference,
mutations), 87.2 % (device); the stub's router keeps none.

* only the row-scaling check.

Reviewed (S.swa_engram.14.test.1). One change to the rendered test: the router's checks.

The rendered test gated swap_router_vs_golden at the router component test's PCC 0.99, which the CPU reference
itself fails: 0.984586 (DS41-30, DS41-32). The golden block input is stored in bf16, and layer 1's is a rounded fp32
stream (layer 0's out), so the fp32 CPU chain starts one bf16 rounding off the golden run; the fp8 quantizers (DS41-1)
take ffn_norm to PCC 0.99985 vs the golden, and that flips near-tie top-6 choices of the 96 experts in 12.4 % of the
rows (K-39). PCC over the dense routing weights [2048, 96] counts every flip as error; no device step can be closer
to the golden than the CPU chain.

So the router vs the golden is split by routing (K-39, testing/component.py:step_errors): the fraction of rows whose
expert set differs from the golden's (reselected), and on the remaining rows rel L2, worst row, row norm ratio and
median ratio bias; PCC over all rows stays as a loose bound. And the router vs the CPU step on the same inputs gets a
tight rel L2 on the rows with the same experts (ROUTER_CPU_REL): the framework's limit there is 2 x the router's
component error, 0.0493, which comes from its PCC on golden inputs and so mostly from flips (limit 0.0987), and 1 %
noise (0.0097) passed it. Every other check of the rendered test is unchanged: the framework's router vs the CPU step
(PCC 0.99, rel, worst row 0.05, ratio 1 +- 0.02, bias 0.01, reselected <= 0.02), the other 13 steps' checks, the
block out (pcc_swap_out >= 0.98) and the tail check (swap_router_out).

Measured (layer 1, the component rung's last dumped chunk, 2048 rows; mutate:<kind> = the CPU router's output
altered, BRINGUP_MUTATE_STEP=router; "golden" = vs the golden split by routing, "cpu rel" = vs the CPU step on the
same inputs, rows with the same experts):

    variant              golden: pcc  reselected  rel      row     ratio             bias       | cpu rel  | test
    reference (fp32)            0.9846  0.1245      0.00513  0.0293  [0.9947, 1.0052]  -1.7e-5   | 0        | pass
    mutate:bf16 (control)       0.9848  0.1235      0.00538  0.0284  [0.9939, 1.0049]  0         | 0.00156  | pass
    device (this box, 2x2)      0.9845  0.1284      0.00528  0.0299  [0.9953, 1.0046]  +6e-6     | 0.00012  | pass
    stub (zeros)                0       1.0         inf      inf     [0, inf]          -1        | inf      | FAIL
    mutate:noise1e-2            0.9845  0.1245      0.01099  0.0309  [0.9943, 1.0049]  +3e-5     | 0.00971  | FAIL *
    mutate:scale1.02 / 0.98     0.9846  0.1245      0.0206   0.036   [0.975, 1.025]    +/-0.020  | -        | FAIL
    mutate:halfswap / rowshift  <0.27   1.0         inf      inf     -                 -1        | inf      | FAIL
    mutate:quarterzero          0.8441  0.3413      0.00503  0.0293  [0.9961, 1.0052]  -1.4e-5   | -        | FAIL
    mutate:sign                 0.7512  0.1245      0.680    1.83    [0.9947, 1.0052]  -1.7e-5   | -        | FAIL

* only these two checks: every framework check passes 1 % noise.
Limits: about 2 x the CPU chain's distance to the golden, and the framework's swap_step_row / _ratio / _bias.
"""

from unittest import mock

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import component
from models.demos.common.bringup.testing.component import run_swap_test, step_errors
from models.demos.common.bringup.testing.harness import compare, mesh_parametrize, spec

S = spec()
BLOCK_TYPE = "swa_engram"
SWAPPED = [
    "engram_rows",
    "engram_kv",
    "engram_add",
    "attn_hc",
    "attn_collapse",
    "attn_norm",
    "q_a",
    "kv",
    "attention",
    "attn_residual",
    "ffn_hc",
    "ffn_collapse",
    "ffn_norm",
    "router",
    "experts",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
CHECKS = "steps"  # also gate every swapped step's own output (testing/component.py); None = block out only

# router vs golden, split by routing (K-39), on the swapped router's output and the golden's L01.router [2048, 96]
ROUTER_GOLDEN_PCC = 0.97  # PCC over all rows, a loose bound: 2 x the CPU chain's 1 - 0.9846
ROUTER_GOLDEN_RESELECT = 0.25  # rows whose top-6 expert set is not the golden's: 2 x the CPU chain's 0.1245
ROUTER_GOLDEN_REL = 0.0103  # rel L2 over the rows that keep the golden's experts: 2 x the CPU chain's 0.00513
ROUTER_GOLDEN_ROW = 0.05  # worst such row, rel L2: thresholds.swap_step_row (CPU chain 0.029)
ROUTER_GOLDEN_RATIO = 0.02  # every such row's norm ratio within 1 +- this: swap_step_ratio (CPU chain 0.0053)
ROUTER_GOLDEN_BIAS = 0.01  # median row norm ratio within 1 +- this: swap_step_bias (CPU chain 2e-5)
# router vs the CPU step on the same inputs, rows that keep the CPU's experts: the framework's limit is 2 x the
# component error (0.0493, from PCC, so mostly golden-input flips: 0.0987), which 1 % noise (0.0097) passes
ROUTER_CPU_REL = 0.004  # bf16 control 0.0016, device 0.00012
# experts vs golden, split by the routing the swapped experts received (the swapped router's output) against the
# golden's: a reselected row runs other experts, so its output is another function, not a precision error (K-39)
EXPERTS_GOLDEN_PCC = 0.98  # PCC over all rows, a loose bound: 2 x the CPU chain's 1 - 0.9908
EXPERTS_GOLDEN_REL = 0.09  # rel L2 over the rows that keep the golden's experts: 2 x the CPU chain's 0.0450
EXPERTS_GOLDEN_ROW = 0.154  # worst such row, rel L2: 2 x the CPU chain's 0.0770
EXPERTS_GOLDEN_RATIO = 0.092  # every such row's norm ratio within 1 +- this: 2 x the CPU chain's 0.0461
EXPERTS_GOLDEN_BIAS = 0.01  # median row norm ratio within 1 +- this: swap_step_bias (CPU chain 1.3e-4)
# experts vs the CPU step on the same inputs: row scaling (DS41-22, test_c_swa_engram_experts.py): floating-point
# error grows with the row, added noise does not; the framework's rel limit (2 x component error 0.029) passes 1 % noise
# experts vs the CPU step on the same inputs, the framework's fixed worst row 0.05 and row norm ratio 1 +- 0.02 are
# tighter than a correct bf16 step (mutate:bf16 control: row 0.0469, ratio [0.9659, 1.0129]; the experts component
# test's own precision-model limits: row 0.0724, ratio 0.0536); 2 x the control, as rel (2 x component error 0.029)
EXPERTS_CPU_LIMITS = {"swap_step_row": 0.094, "swap_step_ratio": 0.068}  # bias (0.01) and rel stay the framework's
ROW_SLOPE_MIN = 0.35  # slope of log row error on log row norm vs the CPU step
ROW_SLOPE_ROWS = 32  # fewer rows with an error than this: nothing to fit, the check passes

_ROUTING: dict = {}  # the swapped router's output and the golden's, kept by the router's check (checked first)

_check_step = component._check_step


class _Limits:
    """The spec with some thresholds replaced (testing/component.py:_check_step reads them through s.threshold)."""

    def __init__(self, s, over: dict):
        self._s, self._over = s, over

    def threshold(self, key: str, default: float) -> float:
        return float(self._over[key]) if key in self._over else self._s.threshold(key, default)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _limits_failed(e: dict, lim: dict) -> list[str]:
    dev = max(1 - e["ratio_min"], e["ratio_max"] - 1)
    bad = [
        f"reselected {e['reselected']:.4f} > {lim['reselected']}" if not e["reselected"] <= lim["reselected"] else "",
        f"rel {e['rel']:.5f} > {lim['rel']}" if not e["rel"] <= lim["rel"] else "",
        f"worst row {e['row']:.5f} > {lim['row']}" if not e["row"] <= lim["row"] else "",
        (
            f"row norm ratio [{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] outside 1 +- {lim['ratio']}"
            if not dev <= lim["ratio"]
            else ""
        ),
        (
            f"median row norm ratio off by {e['bias']:+.5f} (limit {lim['bias']})"
            if not abs(e["bias"]) <= lim["bias"]
            else ""
        ),
    ]
    return [b for b in bad if b]


def _report(tag: str, what: str, e: dict, bad: list[str]) -> bool:
    for k, v in e.items():
        metrics.record(f"{tag}_{k}", v)
    print(
        f"{'ok  ' if not bad else 'FAIL'} {tag}: {what}: reselected={e['reselected']:.4f} rel={e['rel']:.6f} "
        f"row={e['row']:.6f} ratio=[{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] bias={e['bias']:+.6f}"
        + (f" ({'; '.join(bad)})" if bad else "")
    )
    return not bad


def _router_checks(got: torch.Tensor, cpu: torch.Tensor, golden: torch.Tensor) -> bool:
    _, ok = compare("swap_router_vs_golden", got, golden, "pcc", ROUTER_GOLDEN_PCC)
    e = step_errors(got, golden)
    lim = dict(
        reselected=ROUTER_GOLDEN_RESELECT,
        rel=ROUTER_GOLDEN_REL,
        row=ROUTER_GOLDEN_ROW,
        ratio=ROUTER_GOLDEN_RATIO,
        bias=ROUTER_GOLDEN_BIAS,
    )
    ok = _report("swap_router_golden", "vs the golden, split by routing", e, _limits_failed(e, lim)) and ok
    e = step_errors(got, cpu)
    bad = [f"rel {e['rel']:.5f} > {ROUTER_CPU_REL}"] if not e["rel"] <= ROUTER_CPU_REL else []
    return _report("swap_router_cpu", "vs the CPU step, rows with the same experts", e, bad) and ok


def _moved(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Rows whose top-k of |got| is not want's nonzero set (k per row), as testing/component.py:step_errors."""
    nz = want != 0
    rank = (-got.abs()).argsort(1).argsort(1)
    return ((rank < nz.sum(1, keepdim=True)) != nz).any(1)


def row_slope(got: torch.Tensor, ref: torch.Tensor) -> tuple[float | None, int]:
    """(b, rows) of the least-squares fit log ||got_i - ref_i|| = a + b log ||ref_i|| over the rows i where both are
    nonzero (float64); b None when the shapes differ or a value is not finite (test_c_swa_engram_experts.py)."""
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


def _experts_checks(got: torch.Tensor, cpu: torch.Tensor, golden: torch.Tensor) -> bool:
    _, ok = compare("swap_experts_vs_golden", got, golden, "pcc", EXPERTS_GOLDEN_PCC)
    rg, rw = _ROUTING.get("got"), _ROUTING.get("golden")
    if rg is None or rw is None or rg.shape[0] != got.shape[0]:
        print("FAIL swap_experts_golden: no router output / golden routing to split the rows by")
        return False
    moved = _moved(rg, rw)
    if bool(moved.all()):
        print("FAIL swap_experts_golden: no row keeps the golden's experts")
        return False
    e = {"reselected": moved.float().mean().item(), **step_errors(got[~moved], golden[~moved])}
    lim = dict(
        reselected=ROUTER_GOLDEN_RESELECT,
        rel=EXPERTS_GOLDEN_REL,
        row=EXPERTS_GOLDEN_ROW,
        ratio=EXPERTS_GOLDEN_RATIO,
        bias=EXPERTS_GOLDEN_BIAS,
    )
    ok = (
        _report("swap_experts_golden", "vs the golden, rows that keep the golden's experts", e, _limits_failed(e, lim))
        and ok
    )
    b, n = row_slope(got, cpu)
    bad = None
    if b is None:
        bad = f"shape {tuple(got.shape)} or non-finite values, want {tuple(cpu.shape)}"
    elif n >= ROW_SLOPE_ROWS and not b >= ROW_SLOPE_MIN:
        bad = f"row error does not scale with the row: slope {b:.3f} < {ROW_SLOPE_MIN} (added noise?)"
    if b is not None and n >= ROW_SLOPE_ROWS:
        metrics.record("rowscale_swap_experts", b)
    print(
        f"{'ok  ' if not bad else 'FAIL'} rowscale_swap_experts: vs the CPU step: slope={b} rows={n}"
        + (f" ({bad})" if bad else "")
    )
    return not bad and ok


def _checked(s, block_type, layer, st, got, cpu, golden, bridged):
    """The framework's step check; for the router and the experts without their PCC-vs-golden gate, which
    _router_checks / _experts_checks replace, and for the experts with EXPERTS_CPU_LIMITS."""
    if st.name not in ("router", "experts") or golden is None:
        return _check_step(s, block_type, layer, st, got, cpu, golden, bridged)
    lim = _Limits(s, EXPERTS_CPU_LIMITS) if st.name == "experts" else s
    ok = _check_step(lim, block_type, layer, st, got, cpu, None, bridged)
    if isinstance(cpu, Exception) or got.numel() != golden.numel() or not torch.isfinite(got.float()).all():
        return False
    got = got.float().reshape(golden.shape)
    if st.name == "router":
        _ROUTING.update(got=got, golden=golden.float())
        return _router_checks(got, cpu.float().reshape(golden.shape), golden.float()) and ok
    return _experts_checks(got, cpu.float().reshape(golden.shape), golden.float()) and ok


@mesh_parametrize
def test_swap(mesh_device):
    _ROUTING.clear()
    with mock.patch.object(component, "_check_step", _checked):
        assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
