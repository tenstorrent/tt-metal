"""Regression locks for the audit-round-2 fixes (2026-07-25), grouped as in the plan.

Each test pins ONE behaviour that was wrong and is now right. Every case was reproduced against
source before the fix; the docstrings say what the wrong behaviour produced, so a future change
that reintroduces it fails with the reason rather than a bare assertion.

Hermetic: no device, no agent, no claude subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _fresh_perf_mcp(tmp_path):
    run = tmp_path / "models/experimental/perf_automation/runs/2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"config": {"timeout": 10800, "metric": "device_ms"}, "perf_test_resolved": {"path": "t.py"}})
    )
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": 146.72}) + "\n"
    )
    keys = ("PERF_MCP_MANIFEST", "PERF_MCP_KERNEL_LOG", "TMPDIR")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["PERF_MCP_MANIFEST"] = str(run / "manifest.json")
    os.environ["PERF_MCP_KERNEL_LOG"] = str(tmp_path / "kernlog.json")
    os.environ["TMPDIR"] = str(tmp_path)
    try:
        spec = importlib.util.spec_from_file_location("perf_mcp_r2_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["perf_mcp_r2_ut"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ------------------------------------------------------------------ gates that failed open


def test_partial_skip_no_longer_passes_the_pcc_gate():
    """`1 passed, 1 skipped` bypassed the skip guard (it required NO `passed`), reopening the
    SKIP-mislabel class the guard exists for: the trivial case passes, the real e2e case skips."""
    from agent import pcc_runner

    out = pcc_runner._verdict_from_output("== 1 passed, 1 skipped ==", threshold=0.99)
    assert out["status"] != "ok", f"a partially-skipped run was accepted as correct: {out}"


def test_worst_pcc_is_judged_not_the_last_one():
    """parse_pcc took the LAST match, so a printed THRESHOLD line banked as the measured value and
    a per-layer sweep was judged only on its final entry."""
    from agent.pcc_runner import parse_pcc

    assert parse_pcc("PCC: 0.5123\nexpected pcc: 0.99\n") == pytest.approx(
        0.5123
    ), "the echoed threshold was banked as the measured PCC"


def test_unknown_dtype_is_recorded_not_silently_assumed():
    """A miss fell to 2.0 bytes, so a bf8_b model's ceiling was understated ~2x and the run could
    stop declaring success from a string-key miss."""
    from agent import perf_target

    assert perf_target._bytes_per_elem("DataType.BFLOAT8_B") == perf_target.BYTES_PER_ELEM["bfloat8_b"]
    assert perf_target._bytes_per_elem("bfp8_b") == perf_target.BYTES_PER_ELEM["bfloat8_b"]
    perf_target._bytes_per_elem("some_new_format")
    assert "some_new_format" in perf_target.unknown_dtypes(), "an unknown dtype left no trace"


def test_at_floor_requires_most_of_the_profile_to_be_modeled():
    """`at_floor` was true whenever the modeled set had no open ops -- with degraded roofline inputs
    that can be 1 op out of hundreds, and the run was certified 'no reachable gain left'."""
    from agent import roofline

    assert hasattr(roofline, "residual_report")
    src = (_ROOT / "agent" / "roofline.py").read_text()
    assert "len(open_ops) == 0 and len(modeled) > 0,  # nothing" not in src, "at_floor still ignores n_unmodeled"


def test_unknown_arch_raises_instead_of_inheriting_blackhole():
    """An unknown/renamed arch silently got Blackhole peaks -> wrong floor, wrong ceiling, wrong
    DONE verdict. parse_env_snapshot already raised on the same input."""
    src = (_ROOT / "agent" / "environment.py").read_text()
    assert 'ARCH_FACTS.get(arch, ARCH_FACTS["blackhole"])' not in src, "unknown arch still inherits Blackhole peaks"


# ------------------------------------------------------------------ lever burning / ladder


def test_any_unrecognised_host_failure_is_a_measurement_failure_not_a_wedge(tmp_path):
    """It was a 4-substring ALLOW-LIST with 'device wedge' as the fall-through, so every other
    host-side extraction fault burned the lever, bumped the crash counter and reset the board."""
    m = _fresh_perf_mcp(tmp_path)
    for host_side in (
        "tt-perf-report exited 1",
        "KeyError: 'Global Call Count'",
        "shutil.copyfile failed: [Errno 28] No space left on device",
        "subprocess.TimeoutExpired: command timed out",
    ):
        assert m._is_measurement_failure(host_side), f"{host_side!r} still classified as a device wedge"
    for device_fault in ("Segmentation fault (core dumped)", "TT_FATAL @ tt_cluster.cpp:281", "hang detected"):
        assert not m._is_measurement_failure(device_fault), f"{device_fault!r} must stay a wedge"


def test_recall_knobs_returns_the_levers_the_prompt_asks_for(tmp_path):
    """The tool's own instructions mandate recall_knobs(op_class='decode') at the kv-cache gate, but
    'decode' is not router vocabulary, route() raised by design and the except swallowed it -- so
    the mandatory prior-knowledge lookup reported 'nothing catalogued' every single time."""
    m = _fresh_perf_mcp(tmp_path)
    assert m.recall_knobs("decode")["count"] > 0, "the mandated kv-cache recipe lookup is still empty"
    assert m.recall_knobs("totally_bogus_class")["count"] > 0, "an unknown op_class still returns a silent empty"


def test_route_block_typos_are_reported_not_turned_into_wildcards():
    """The query side raises on vocab drift; the index side did not, so a typo'd dimension became a
    WILDCARD and that lever matched EVERY bucket."""
    from agent import router

    idx = router.build_index(str(_ROOT / "GUIDELINES"))
    assert idx, "no levers indexed"
    assert hasattr(router, "index_warnings") and hasattr(router, "all_entries")
    assert router.index_warnings(idx) == [], f"route-block defects present: {router.index_warnings(idx)}"


def test_revert_removes_files_the_edit_created(tmp_path):
    """`git checkout <sha> -- <path>` only rewrites TRACKED files, so a lever that CREATED a file
    survived every revert and its edit stayed in the next measurement -- the documented
    'revert no-ops -> infinite wedge' trap. Pre-existing untracked artifacts must survive."""
    import subprocess

    from agent import gitio

    d = tmp_path / "repo"
    d.mkdir()
    for args in (["init", "-q", "."], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=d, check=True)
    (d / "m").mkdir()
    (d / "m/tracked.py").write_text("x=1\n")
    (d / "m/artifact.json").write_text("{}\n")
    subprocess.run(["git", "add", "m/tracked.py"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=d, check=True)

    baseline = gitio.untracked_under(d, "m")
    (d / "m/tracked.py").write_text("x=2\n")
    (d / "m/new_kernel.py").write_text("kernel\n")
    gitio.checkout(d, gitio.head_sha(d), "m")
    removed = gitio.remove_new_untracked(d, baseline, "m")

    assert (d / "m/tracked.py").read_text().strip() == "x=1", "tracked file not restored"
    assert not (d / "m/new_kernel.py").exists(), "the file the edit CREATED survived the revert"
    assert (d / "m/artifact.json").exists(), "a pre-existing untracked artifact was destroyed"
    assert removed == ["m/new_kernel.py"]


def test_op_sig_probe_always_returns_three_values():
    """The device-timeout path returned a 2-tuple while all four callers unpack 3, so a timeout
    raised ValueError and the stop message blamed 'a build/env/version mismatch'."""
    src = (_ROOT / "cc_optimize" / "run.py").read_text()
    body = src.split("def _run_op_sigs(", 1)[1].split("\ndef ", 1)[0]
    assert 'return None, ""\n' not in body, "the timeout path still returns a 2-tuple"


# ------------------------------------------------------------------ false facts


def test_parallelism_is_matched_in_the_order_the_producer_emits(tmp_path):
    """The regex required 'DP=... TP=...' while the only producer emits TP first, so it never
    matched once and even an 8-chip TP=8 run printed TP=1 x DP=1 as a measured fact."""
    spec = importlib.util.spec_from_file_location("ccrun_r2", _ROOT / "cc_optimize" / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    facts = m._parse_facts("[full-pipeline-gate] PERF_SCORECARD mesh=1x4 TP=4 DP=1 shard=True", {"ttnn.matmul(x)"})
    assert (facts["tp"], facts["dp"]) == (4, 1), f"TP/DP still unparsed: {facts}"
    assert facts["shard_active"] is True
    assert facts["parallelism_known"] is True
    assert (
        m._parse_facts("no scorecard line here", set())["parallelism_known"] is False
    ), "absent parallelism is still reported as a measured 1x1"


def test_eager_is_not_banked_as_a_trace_measurement():
    """bool('eager') is True, so TRACE_REPLAY_PATH=eager/none/skipped counted as traced; and the
    non-whitespace pattern truncated 'trace 1cq' -- the spelling the tool's own skeleton emits --
    to the bare word 'trace'."""
    from agent.perf_test_gen import _parse_trace_path as f

    for eager in ("TRACE_REPLAY_PATH=eager", "TRACE_REPLAY_PATH=none", "TRACE_REPLAY_PATH=skipped"):
        assert f(eager) is None, f"{eager} still counts as a trace measurement"
    assert f("TRACE_REPLAY_PATH=trace") is None, "a bare 'trace' is not a real trace+Ncq path"
    assert f("TRACE_REPLAY_PATH=trace 1cq") == "trace+1cq"
    assert f("TRACE_REPLAY_PATH=trace+1cq") == "trace+1cq"


def test_degenerate_device_output_scores_zero_pcc_not_one():
    """A constant/all-zero device output gave denom==0 -> PCC 1.0 = perfect, and matmul_sweep then
    picked the fastest BROKEN config as its PCC-gated recommendation."""
    torch = pytest.importorskip("torch")
    from cc_optimize.tp_fracture import _pcc

    ref = torch.randn(64, 64)
    assert _pcc(ref, torch.zeros(64, 64)) == 0.0, "an all-zero device output still scores a perfect PCC"
    assert _pcc(ref, ref) == pytest.approx(1.0), "an identical tensor must still score 1.0"


def test_unmatched_memory_string_is_unknown_not_a_real_category():
    """Blank/unknown returned the REAL routing category dram_interleaved, so L1/sharded ops were
    routed as DRAM-interleaved and the shard lever was offered against already-sharded tensors."""
    from agent.tracy_tool import normalize_memory

    assert normalize_memory("") == "unknown"
    assert normalize_memory("L1") == "l1_interleaved"
    assert normalize_memory("DRAM") == "dram_interleaved"


def test_half_set_mesh_env_does_not_silently_open_one_by_one(monkeypatch, capsys):
    """int("") raised and both sides were discarded, so a planned TP=4 mesh opened as 1x1 and that
    single-chip measurement was reported as the planned topology."""
    from agent.perf_adapter import resolve_mesh_shape

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "1")
    monkeypatch.delenv("TT_PERF_MESH_COLS", raising=False)
    assert resolve_mesh_shape(1, 4) == (1, 4), "a half-set mesh env still collapses to the bare default"
    # An unparseable value must NOT abort the run (that would make a bad env var fatal); it falls
    # back to the source default and says so, so the topology is never silently wrong-and-quiet.
    monkeypatch.setenv("TT_PERF_MESH_COLS", "not-a-number")
    assert resolve_mesh_shape(2, 2) == (2, 2)
    assert "WARNING" in capsys.readouterr().err, "the fallback was silent"


def test_promotion_requires_positive_evidence():
    """Every missing field defaulted to 'kept win', so an edit that was never measured could be
    distilled by an LLM into a permanent GUIDELINES lever every future run trusts."""
    src = (_ROOT / "agent" / "promote.py").read_text()
    assert 'd.get("result", "keep") == "keep"' not in src, "a missing result still defaults to a kept win"
    assert "before is None or after is None" not in src, "a missing measurement still passes the comparison"
