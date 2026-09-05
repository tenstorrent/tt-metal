# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The stop gate and the report must divide by the SAME pinned baseline bytes.

THE DEFECT: the write-once ledger anchor exists because the optimize loop REVERTS the model directory
between attempts, so perf_target_inputs.json describes whichever vintage is on disk. The report read
the anchor; the stop gate (_select_perf_target -> compute_target) read the facts. So one run had two
ceilings -- and the gate's moved, because a lever that shrinks weights shrinks the facts, which is
exactly the "optimized build lowers its own target" failure the anchor was introduced to stop.

IN_BAND is what terminates a run, so the gate is the side that must not drift.
"""

import sys
from pathlib import Path

_CC = Path(__file__).resolve().parents[1] / "cc_optimize"
if str(_CC) not in sys.path:
    sys.path.insert(0, str(_CC))
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import perf_mcp  # noqa: E402

from agent import perf_target  # noqa: E402

_GB = 1e9


def _led_mod():
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("led_gate_ut", _CC / "measurements.py")
    mod = ilu.module_from_spec(spec)
    sys.modules["led_gate_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def _facts(params):
    return {"total_params": int(params), "unit": "token", "weight_bytes": int(params)}


def test_the_gate_divides_by_the_anchor_not_the_reverted_facts(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    led = _led_mod()

    # BASELINE pinned at 8 GB per token.
    led.anchor(led.KIND_ACTIVE_BYTES, 8000.0, depth="token", mode="bytes_mb", source="test", model="m")
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")
    monkeypatch.setattr(perf_mcp, "_ENV", {"dram_bw_gbps": 512.0})

    # ...and the facts on disk now claim HALF that (a bf4 lever landed, or a revert restored a
    # truncated vintage). The gate must still judge against the baseline.
    monkeypatch.setattr(perf_mcp, "_load_perf_target_inputs", lambda: _facts(4e9))
    target, scope, has_ceiling = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})

    assert (scope, has_ceiling) == ("model", True)
    assert target.active_bytes == int(8 * _GB), "gate followed the facts instead of the anchor"
    assert round(target.theoretical_rate, 1) == 64.0  # spec 512/8, NOT 128.0 from 4 GB
    assert target.bytes_source == "anchored baseline bytes"


def test_with_nothing_pinned_the_gate_still_computes_from_the_facts(tmp_path, monkeypatch):
    """The anchor is an override, not a requirement: a first-ever profile has nothing pinned yet."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "empty.jsonl"))
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")
    monkeypatch.setattr(perf_mcp, "_ENV", {"dram_bw_gbps": 512.0})
    monkeypatch.setattr(perf_mcp, "_load_perf_target_inputs", lambda: _facts(8e9))

    target, _scope, _has = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})
    assert round(target.theoretical_rate, 1) == 64.0
    assert "params rule" in target.bytes_source


def test_the_gate_and_the_report_agree_on_one_run(tmp_path, monkeypatch):
    """Same anchor, same divisor, same ceiling -- the invariant the split broke."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    led = _led_mod()
    led.anchor(led.KIND_ACTIVE_BYTES, 7505.0, depth="token", mode="bytes_mb", source="test", model="m")
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")
    monkeypatch.setattr(perf_mcp, "_ENV", {"dram_bw_gbps": 512.0})
    monkeypatch.setattr(perf_mcp, "_load_perf_target_inputs", lambda: _facts(4e9))

    gate, _s, _h = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})

    import importlib.util as ilu

    spec = ilu.spec_from_file_location("sm_gate_ut", _CC / "summary.py")
    sm = ilu.module_from_spec(spec)
    sys.modules["sm_gate_ut"] = sm
    spec.loader.exec_module(sm)
    snap = {
        "has_unit_ceiling": True,
        "theoretical_rate": gate.theoretical_rate,
        "band": [gate.band[0], gate.band[1]],
        "active_bytes": gate.active_bytes,
        "peak_bw_gbps": 512.0,
        "bw_fraction": gate.bw_fraction,
        "tp_degree": 1,
        "perf_layers": "all",
        "unit": "token",
    }
    txt = "\n".join(sm._roofline_lines(snap, None, {"per_token_ms": 17.0}, "m", "main"))

    # 7.505 GB anchored -> (512*0.8)/7.505 = 54.6, in BOTH places, and never the 4 GB facts' 102.4.
    assert round(gate.theoretical_rate, 1) == 68.2
    # The band column no longer repeats the unit on every cell ("40.9 - 54.6", not "... 54.6 tok/s/u").
    # What matters is unchanged: the ANCHORED value is what renders, never the 4 GB facts' 102.4.
    assert "54.6" in txt, txt
    assert "102.4" not in txt, txt
    assert "102.4" not in txt, txt


def test_a_module_level_run_is_unaffected(tmp_path, monkeypatch):
    """Per-module runs use the module's own floor and carry no band; the anchor must not smuggle a
    model-level ceiling into them."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    led = _led_mod()
    led.anchor(led.KIND_ACTIVE_BYTES, 8000.0, depth="token", mode="bytes_mb", source="test", model="m")
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")

    target, scope, has_ceiling = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})
    assert (scope, has_ceiling) == ("module", False)
    assert target.band == (0.0, 0.0)


def test_losing_the_facts_file_does_not_downgrade_the_gate_to_the_floor(tmp_path, monkeypatch):
    """THE DEFECT (found by stress-testing the drift): perf_target_inputs.json is UNTRACKED, so the
    optimize loop's revert can delete it. With no facts the gate skipped the ceiling branch and fell to
    target_from_floor_ms -- a band-less ms floor, on which the band stop can never fire -- while the
    report went on showing the pinned ceiling. The anchored row carries the unit it was pinned under,
    so the baseline is rebuildable without the file."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    led = _led_mod()
    led.anchor(led.KIND_ACTIVE_BYTES, 8000.0, depth="token", mode="bytes_mb", source="test", model="m")
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")
    monkeypatch.setattr(perf_mcp, "_ENV", {"dram_bw_gbps": 512.0})
    monkeypatch.setattr(perf_mcp, "_load_perf_target_inputs", lambda: None)  # file gone

    target, scope, has_ceiling = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})
    assert (scope, has_ceiling) == ("model", True), "downgraded to the floor form"
    assert round(target.theoretical_rate, 1) == 64.0  # spec 512/8; not 1000/15.6 = 64.1 from the floor
    assert target.band[0] > 0, "a floor target carries no band, so the band stop could never fire"
    assert target.unit == "token", "the unit must travel with the pinned bytes"


def test_a_step_unit_anchor_rebuilds_as_steps_not_tokens(tmp_path, monkeypatch):
    """The unit is recovered from the anchored row, so a diffusion model does not silently become
    per-token when its facts file disappears."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    led = _led_mod()
    led.anchor(led.KIND_ACTIVE_BYTES, 2000.0, depth="step", mode="bytes_mb", source="test", model="m")
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path / "m")
    monkeypatch.setattr(perf_mcp, "_ENV", {"dram_bw_gbps": 512.0})
    monkeypatch.setattr(perf_mcp, "_load_perf_target_inputs", lambda: {})

    target, _s, _h = perf_mcp._select_perf_target({"modeled_floor_ms": 15.6})
    assert target.unit == "step"
    assert round(target.theoretical_rate, 1) == 256.0  # (512*0.8)/2
