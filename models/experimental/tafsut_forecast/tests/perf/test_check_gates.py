# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""check_gates.py unit tests (TiRex-2 651e8c4af93 pattern, on synthetic records; no device)."""

import copy

from models.experimental.tafsut_forecast.tests.perf.check_gates import LOWER_IS_BETTER, check

GATES = {
    "floors": {"*/accuracy": {"pcc_normalized": 0.999, "abs_err_max": 0.5, "spread_ratio_p50": [0.98, 1.02]}},
    "drift": {"pcc_spread": {"abs": 0.001}, "spread_ratio_p50": {"two_sided": True, "abs": 0.01}},
}
REC = {
    "name": "paper",
    "precision": "accuracy",
    "git_head": "x",
    "git_dirty": "",
    "pcc_normalized": 0.9999,
    "abs_err_max": 0.1,
    "spread_ratio_p50": 1.0,
    "pcc_spread": 0.9999,
}


def test_lower_is_better_set():
    assert {"inactive_max_err", "abs_err_max", "flat_max_err", "median_err", "spread_ratio_max"} <= LOWER_IS_BETTER


def test_passes():
    assert not check([REC], [REC], GATES)


def test_floor_and_drift_fail():
    assert check([dict(REC, pcc_normalized=0.99)], [REC], GATES)
    assert check([dict(REC, abs_err_max=0.9)], [REC], GATES)
    assert check([dict(REC, spread_ratio_p50=1.015)], [REC], GATES)  # in range, outside two-sided drift
    assert check([dict(REC, pcc_spread=0.99)], [REC], GATES)


def test_not_applicable_skips_floor_only():
    assert not check(
        [dict(REC, spread_ratio_p50=None)], [dict(REC, spread_ratio_p50=None)], {"floors": GATES["floors"], "drift": {}}
    )


def test_missing_drift_metric_fails():
    r = copy.deepcopy(REC)
    del r["pcc_spread"]
    assert any("pcc_spread" in f for f in check([r], [REC], GATES))


def test_dirty_or_unstamped_fails():
    assert any("dirty" in f for f in check([dict(REC, git_dirty=" M tt/model.py")], [REC], GATES))
    r = {k: v for k, v in REC.items() if k not in ("git_head", "git_dirty")}
    assert any("stamp" in f for f in check([r], [REC], GATES))


def _g1():
    import json
    from pathlib import Path

    doc = Path(__file__).resolve().parents[2] / "doc" / "optimize"
    gates = json.loads((doc / "gates_g1_proposed.json").read_text())
    r0 = [json.loads((doc / "iter_00" / f).read_text()) for f in ("bench.json", "suite.json")]
    return gates, r0[0] + r0[1]


def test_g1_proposal_passes_round0():
    """v1.17: the measured round-0 bench and suite records pass every proposed floor."""
    gates, r0 = _g1()
    assert not check(r0, None, gates)


def test_g1_proposal_fails_broken():
    """v1.9 / v1.17: every broken output fails at least one proposed floor, on every case, in both precisions."""
    from models.experimental.tafsut_forecast.reference import dataflow as df
    from models.experimental.tafsut_forecast.tests.pcc import metrics as M
    from models.experimental.tafsut_forecast.tests.perf import score_broken as SB
    from models.experimental.tafsut_forecast.tests.perf.check_gates import _floors, passes

    gates, _ = _g1()
    passed = []
    for shape in SB.BENCH + SB.SUITE:
        gh, prep, take, gq = SB.case_inputs(shape)  # skips when the goldens are missing
        for name, (head, _) in SB.broken(gh).items():
            m = M.gate_metrics(head, gh, df.postprocess(head.float(), prep, take).transpose(1, 2), gq, prep.scale)
            for prec in ("accuracy", "performance"):
                fl = _floors(gates, shape, prec)
                if all(m.get(k) is None or passes(k, m[k], f) for k, f in fl.items()):
                    passed.append(f"{shape}/{prec}/{name}")
    assert not passed, passed
