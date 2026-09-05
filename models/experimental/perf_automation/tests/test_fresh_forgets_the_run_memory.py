# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`--fresh` forgets what a run remembers, including the value nothing else clears.

A pinned number records WHAT it is and never WHICH RULE produced it, so a value from a superseded
formula outlives the fix. Measured on Voxtral-Mini-3B, 2026-08-14 -- the ceiling anchor held:

    {"kind": "active_bytes", "value_ms": 3611.4831, "mode": "bytes_mb",
     "source": "checkpoint bytes + HF config"}

3611.48 MB is total_params x 1.0, the placeholder width from before the ceiling divided by the width
the loader actually chose. compute_target takes `bytes_per_unit` ahead of every other source, so the
corrected rule (measured 2.0 B/param -> 7.223 GB) never ran and the run published 141.8 tok/s/u
against a true ~71 -- the model reading as twice as close to the wall as it is, which is the input to
can_stop. Clearing the coverage and knob caches did not touch it: it lives in the persistent ledger,
which is why a run started "from scratch" inherited a ceiling pinned days earlier.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _state(tmp):
    d = Path(tmp) / "state"
    d.mkdir(parents=True, exist_ok=True)
    for n in (
        "perf_measurements_m_main.jsonl",
        "perf_mcp_baseline_m_main.json",
        "perf_mcp_full_pipeline_baseline_1cq_m_main.json",
        "perf_mcp_throughput_m_main.json",
        "perf_mcp_knob_cache.json",
        "perf_mcp_thermal_profile.json",
        "cc_kernlog_m_main.json",
        "perf_mcp_board_topology.json",
        "tt_device_recovery_model_main.json",
    ):
        (d / n).write_text("{}")
    (d / "perf_mcp_profile_cache").mkdir(exist_ok=True)
    return d


def test_the_ceiling_anchor_is_cleared(tmp_path):
    """THE ONE NOTHING ELSE CLEARS. The ledger is where the write-once anchor lives."""
    from agent.fresh_start import wipe

    d = _state(tmp_path)
    removed = {p.name for p in wipe(d)}
    assert "perf_measurements_m_main.jsonl" in removed
    assert not (d / "perf_measurements_m_main.jsonl").exists()


def test_hardware_facts_are_kept(tmp_path):
    """Board topology and the device-recovery record describe the MACHINE, not this attempt, and
    re-deriving them costs device time for the same answer."""
    from agent.fresh_start import KEEP, wipe

    d = _state(tmp_path)
    wipe(d)
    for keep in KEEP:
        assert (d / keep).exists(), keep


def test_every_carried_measurement_is_cleared(tmp_path):
    """A baseline, a full-pipeline best-so-far or a throughput dict left behind is a number from the
    old tool that the new run would silently reuse."""
    from agent.fresh_start import wipe

    d = _state(tmp_path)
    wipe(d)
    for gone in (
        "perf_mcp_baseline_m_main.json",
        "perf_mcp_full_pipeline_baseline_1cq_m_main.json",
        "perf_mcp_throughput_m_main.json",
        "perf_mcp_knob_cache.json",
        "cc_kernlog_m_main.json",
        "perf_mcp_profile_cache",
    ):
        assert not (d / gone).exists(), gone


def test_a_dry_run_removes_nothing(tmp_path):
    from agent.fresh_start import wipe

    d = _state(tmp_path)
    planned = wipe(d, dry_run=True)
    assert planned
    assert (d / "perf_measurements_m_main.jsonl").exists()


def test_it_never_touches_model_source(tmp_path):
    """Only generated state. A mistake here costs a slow run, never a lost edit."""
    from agent.fresh_start import plan

    demo = Path(tmp_path) / "demo"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("x = 1\n")
    (demo / "perf_target_inputs.json").write_text("{}")
    names = {p.name for p in plan(None, model_dir=demo)}
    assert "perf_target_inputs.json" in names
    assert "pipeline.py" not in names


def test_the_flag_is_wired_into_optimize():
    src = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    assert 'getattr(args, "fresh", False)' in src, "the flag is declared but never acted on"
    assert src.index('getattr(args, "fresh", False)') < src.index("result = run_cc("), "cleared after the run starts"
    cli = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    assert '"--fresh"' in cli
    # NAMES MUST BE IN SCOPE. The first wiring referenced `tt_root`, which does not exist in that
    # function: --fresh printed "skipped: name 'tt_root' is not defined", failed safe, and the run
    # continued on the very state it was asked to clear. A flag that silently does nothing is worse
    # than no flag, because the operator believes the run is clean.
    import ast as _ast

    src_o = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "optimize.py").read_text()
    tree = _ast.parse(src_o)
    fn = next(
        n
        for n in _ast.walk(tree)
        if isinstance(n, _ast.FunctionDef) and "getattr(args, 'fresh', False)" in _ast.unparse(n)
    )
    bound = {t.id for n in _ast.walk(fn) if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Store) for t in [n]}
    bound |= {a.arg for a in fn.args.args}
    used = _ast.unparse(fn)
    i = used.index("getattr(args, 'fresh', False)")
    for name in ("run_root", "run_demo"):
        assert name in bound, "%s is not bound in the function that clears state" % name
    assert "tt_root" not in used[i : i + 900], "the clear still references a name that is not in scope"
