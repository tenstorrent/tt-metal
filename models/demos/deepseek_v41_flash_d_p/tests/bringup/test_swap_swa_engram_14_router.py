# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 14: block type swa_engram (layer 1) with router swapped in last.

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

_check_step = component._check_step


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


def _checked(s, block_type, layer, st, got, cpu, golden, bridged):
    """The framework's step check; for the router without its PCC-vs-golden gate, which _router_checks replaces."""
    if st.name != "router" or golden is None:
        return _check_step(s, block_type, layer, st, got, cpu, golden, bridged)
    ok = _check_step(s, block_type, layer, st, got, cpu, None, bridged)
    if isinstance(cpu, Exception) or got.numel() != golden.numel() or not torch.isfinite(got.float()).all():
        return False
    got = got.float().reshape(golden.shape)
    return _router_checks(got, cpu.float().reshape(golden.shape), golden.float()) and ok


@mesh_parametrize
def test_swap(mesh_device):
    with mock.patch.object(component, "_check_step", _checked):
        assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
