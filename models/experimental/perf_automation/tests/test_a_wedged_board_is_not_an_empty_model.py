# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: the op-signature probe must tell a dead board apart from a model with nothing to count.

_op_sig_probe prints `PERF_OP_SIGS=[]` and exits 0 whenever the perf test dispatched no op. It
does that for a model whose ops the walk cannot see -- and it does exactly the same thing when the
test never ran because the mesh fixture could not bring the board up. Both reached _coverage_layers
as the same empty set.

Measured 2026-09-25, reproduced from the run's own command line:

    ERROR at setup of test_main_perf[wormhole_b0-mesh_device0-device_params0]
    RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 14521888976,
                  ETH core e8-6 (NOC0) to advance. Stuck at 0xaabb0024
    1 error in 51.55s
    PERF_OP_SIGS=[]

The run reported "the op-signature probe found nothing", removed the depth cap, and spent twenty
rounds with no measured profiling window and no per-op levers. `eth heartbeat` was ALREADY in
DEAD_BOARD_SIGS and the text was ALREADY in the probe's output -- but nothing in cc_optimize/run.py
ever asked. Recovery had been wired into the paths that run after this one (perf_mcp's baseline,
agent/probes' capture); the coverage probe runs at discovery, before either, and a wedge had never
been seen that early.

Two things are pinned here: the probe reclaims and retries on that evidence, and when it still
comes back empty the reason says the BOARD died rather than blaming the model.
"""

from __future__ import annotations

import inspect

import pytest

from models.experimental.perf_automation.cc_optimize import run as run_mod

# The wedge this box actually produces. Used verbatim so the test fails if the signature list ever
# stops recognising it -- the predicate and the retry are only useful together.
WEDGE = (
    "E       RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 14521888976, "
    "ETH core e8-6 (NOC0) to advance. Stuck at 0xaabb0024\n"
    "PERF_OP_SIGS=[]\n"
    "PERF_OP_SIG_SEQUENCE=[]\n"
)
BENIGN_EMPTY = "1 passed in 12.00s\nPERF_OP_SIGS=[]\nPERF_OP_SIG_SEQUENCE=[]\n"
GOOD = '1 passed\nPERF_OP_SIGS=["matmul((32, 32),)"]\nPERF_OP_SIG_SEQUENCE=["matmul((32, 32),)"]\n'


@pytest.fixture()
def probe(monkeypatch, tmp_path):
    """_run_op_sigs with every device-touching collaborator stubbed, recording what it did."""
    calls: dict = {"runs": [], "reclaims": []}

    def _fake_step(cmd, cwd, env, devices, timeout_s, label="", **kw):
        calls["runs"].append(label)
        out = calls["outputs"].pop(0)
        return (None, "") if out is None else (0, out)

    def _fake_reclaim(devices, error_text="", after_kill=False):
        calls["reclaims"].append(error_text)
        return "reclaimed device"

    monkeypatch.setattr(run_mod, "_run_device_step", _fake_step)
    monkeypatch.setattr(run_mod, "_reclaim_device", _fake_reclaim)
    monkeypatch.setattr(run_mod, "cc_env", lambda root, devices: {})
    monkeypatch.setattr(run_mod, "_measure_backstop", lambda root: 60)
    monkeypatch.setattr(run_mod, "adaptive_timer", lambda *a, **k: 60)
    monkeypatch.setattr(run_mod, "_python_bin", lambda root: "python")

    def _go(outputs, retries=None):
        calls["outputs"] = list(outputs)
        if retries is not None:
            monkeypatch.setattr(run_mod, "_OP_SIG_WEDGE_RETRIES", retries)
        result = run_mod._run_op_sigs(tmp_path, {}, "all", "node.py", "case", 0)
        return result, calls

    return _go


def test_a_wedged_probe_is_reclaimed_and_retried(probe):
    """The failure this whole change is about: empty output whose text names a dead board."""
    (sigs, raw, seq), calls = probe([WEDGE, GOOD])
    assert len(calls["runs"]) == 2, "the probe was not retried after the reclaim"
    assert len(calls["reclaims"]) == 1, "the board was never reclaimed"
    assert WEDGE in calls["reclaims"][0], "the reclaim was not given the probe's own evidence"
    assert sigs, "the retry's signatures were thrown away"


def test_an_empty_probe_on_a_live_board_is_not_a_wedge(probe):
    """A model with nothing to count must not cost a device reset."""
    (sigs, raw, seq), calls = probe([BENIGN_EMPTY])
    assert calls["reclaims"] == [], "a healthy empty probe reset the board"
    assert len(calls["runs"]) == 1
    assert sigs is None and raw == BENIGN_EMPTY and seq == []


def test_a_probe_that_works_first_time_is_left_alone(probe):
    (sigs, raw, seq), calls = probe([GOOD])
    assert len(calls["runs"]) == 1 and calls["reclaims"] == []
    assert sigs == {"matmul((32, 32),)"} and seq == ["matmul((32, 32),)"]


def test_the_retry_budget_is_bounded(probe):
    """A board that stays dead must not be retried forever -- one extra attempt, then give up."""
    (sigs, raw, seq), calls = probe([WEDGE, WEDGE, WEDGE])
    assert len(calls["runs"]) == 2, "the budget was exceeded"
    assert sigs is None
    assert raw == WEDGE, "the evidence must survive so the caller can name the reason"


def test_the_budget_can_be_switched_off(probe):
    """0 restores the old single-shot behaviour for anyone who needs it."""
    (sigs, _r, _s), calls = probe([WEDGE, GOOD], retries=0)
    assert len(calls["runs"]) == 1 and calls["reclaims"] == []
    assert sigs is None


def test_a_timeout_still_returns_the_three_tuple_it_always_did(probe):
    """_run_device_proc already reclaimed on its way out and captured nothing to reason about.
    Unchanged on purpose: all four callers unpack three values."""
    (sigs, raw, seq), calls = probe([None])
    assert (sigs, raw, seq) == (None, "", [])
    assert calls["reclaims"] == [], "the timeout path reclaims inside _run_device_proc, not twice"


def test_the_probe_reuses_the_shared_recovery(nothing=None):
    """No parallel reset logic and no second copy of the predicate: the same _dr()/_reclaim_device
    the rest of this file already uses."""
    src = inspect.getsource(run_mod._run_op_sigs)
    assert "_dr().is_dead_board(" in src
    assert "_reclaim_device(" in src
    assert "tt-smi" not in src, "the probe must not issue resets itself"


def test_the_output_is_parsed_in_one_place():
    """The retry loop must not grow a second copy of the PERF_OP_SIGS parse."""
    src = inspect.getsource(run_mod)
    assert src.count('startswith("PERF_OP_SIGS=")') == 1, "the parse was duplicated"
    assert "_parse_op_sigs(" in inspect.getsource(run_mod._run_op_sigs)


def test_a_dead_board_is_named_distinctly_from_an_empty_probe():
    """ "probe_failed" reads as a defect in the model. When the board died, say so."""
    src = inspect.getsource(run_mod._coverage_layers)
    assert "_dr().is_dead_board(raw)" in src, "the final exit never consults the evidence"
    assert 'facts["no_window"] = "board_wedged"' in src
    assert 'facts["no_window"] = "probe_failed"' in src, "the live-board reason must stay"


def test_every_none_exit_still_states_a_reason():
    """The invariant test_no_window_means_no_window pins: one stated reason per None exit."""
    src = inspect.getsource(run_mod._coverage_layers)
    assert src.count("return None, facts") == src.count('facts["no_window"]')


def test_the_depth_bridge_can_explain_the_new_reason():
    """A reason run.py can emit and before_loop cannot spell reaches the operator as
    "reason not reported by the coverage probe" -- which is what this change exists to end."""
    from pathlib import Path

    bl = Path(inspect.getsourcefile(run_mod)).resolve().parents[1] / "agent" / "before_loop.py"
    src = bl.read_text()
    blk = src[src.index("NO WINDOW MEANS NO WINDOW") : src.index("_bl_depth = _bridge_depth_env(")]
    assert '"board_wedged"' in blk, "the bridge cannot name the reason run.py now reports"
    i = blk.index('"board_wedged"')
    assert "LOST" in blk[i : i + 400], "a lost measurement must not read as a decision"


# ------------------------------------------------------------------ and the reclaim must ACT on it
#
# Detecting the wedge is half of it. Measured 2026-09-25 with the retry above already in place:
#
#   coverage probe dispatched no ops and its output names a dead board -- reclaiming and
#   retrying (attempt 2 of 2): reclaimed device (killed holders none) + no reset issued
#
# recover()'s telemetry veto cancelled the reset because every chip was reporting a die
# temperature -- which a stuck ETH fabric does not change -- so recover() returned True, the caller
# was told the board had come back, and the retry hit the identical wedge 180 s later.


@pytest.fixture()
def reset_gate(monkeypatch):
    """recover() with its collaborators stubbed; returns the targets a reset was issued against."""
    from agent import device_recovery as dr
    from agent import probes

    def _go(error_text, temps=(60.0, 61.0, 60.5, 62.0), dead=()):
        issued: list = []
        monkeypatch.setattr(probes, "board_telemetry", lambda: (list(temps), list(dead)))
        monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
        monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
        monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["all"])
        monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
        ok = dr.recover("test", lambda tgt: issued.append(tgt), error_text=error_text)
        return ok, issued

    return _go


def test_a_stuck_eth_fabric_is_reset_even_though_every_chip_is_warm(reset_gate):
    """The fault this change exists for: the ARC is fine, the fabric is not."""
    ok, issued = reset_gate(WEDGE)
    assert issued == ["all"], "the telemetry veto still cancelled a reset backed by hard evidence"
    assert ok is True


def test_a_timeout_with_no_signature_still_cancels_the_reset(reset_gate):
    """The 2026-08-17 protection, unchanged: a slow op must not cost four chips."""
    ok, issued = reset_gate("coverage probe KILLED after 600s (hard limit)")
    assert issued == [], "a reset fired on a board with no evidence that anything is wrong"
    assert ok is True


def test_no_evidence_at_all_leaves_the_veto_in_charge(reset_gate):
    for evidence in ("", None):
        ok, issued = reset_gate(evidence)
        assert issued == [], repr(evidence)


def test_a_silent_chip_still_resets_without_any_signature(reset_gate):
    """The veto only ever cancels; a chip that cannot report still gets its reset."""
    ok, issued = reset_gate("some unrelated failure", temps=(60.0,), dead=["hwmon3"])
    assert issued == ["all"]


def test_the_evidence_check_reuses_the_one_predicate():
    """No second signature list: the same is_dead_board the probe, the baseline and note_crash use."""
    from agent import device_recovery as dr

    src = inspect.getsource(dr.recover)
    assert "is_dead_board(error_text)" in src
    assert "_board_needs_reset()" in src, "the telemetry veto must still be consulted"
    for sig in dr.DEAD_BOARD_SIGS:
        assert sig not in src, f"{sig!r} was copied into recover() instead of being asked for"
