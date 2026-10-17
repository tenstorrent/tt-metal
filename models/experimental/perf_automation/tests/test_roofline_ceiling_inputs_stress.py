# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the roofline ceiling must get correct INPUTS — right weight size, right depth label,
and a snapshot that actually exists.

Three independent bugs made the report print a garbage ceiling (13798 tok/s/u [2-layer], util n/a):

  s1  WEIGHT SIZE: _model_weight_bytes summed _captured/ + _stubs/ TEST FIXTURES as if they were
      model weights, so a tiny non-weight total short-circuited the real checkpoint (XTTS: 102
      captured *.pt = 133 MB masked the true 1.868 GB). The scan must skip fixture dirs.
  s2  DEPTH LABEL: the config/anchored ceiling divides by DEPTH-INDEPENDENT full-model bytes, so its
      perf_layers must be "all" -- stamping the profiler window (TT_PERF_LAYERS=2) made the report's
      depth guard falsely mismatch it against the all-layer measurement and blank utilization.
  s3  PERSISTENCE: the snapshot was written ONLY by profile_model, which the agent never calls, so the
      report usually read a missing file -> NO_BAND. termination_check (called every step) must write it.

Everything model-agnostic: no model name, no byte number, no arch value is baked into the fixes.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _load_run():
    spec = importlib.util.spec_from_file_location("cc_run_ceiling_stress", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- s1
def test_s1_weight_scan_skips_captured_and_stubs(tmp_path):
    """A real weight file is counted; _captured/_stubs fixtures are not."""
    m = _load_run()
    (tmp_path / "weights.pth").write_bytes(b"\x00" * 5000)  # real weight
    cap = tmp_path / "_captured" / "g_p_t"
    cap.mkdir(parents=True)
    (cap / "args.pt").write_bytes(b"\x00" * 900)  # fixture
    (cap / "output.pt").write_bytes(b"\x00" * 900)  # fixture
    stub = tmp_path / "_stubs"
    stub.mkdir()
    (stub / "x.last_good_sharded.pt").write_bytes(b"\x00" * 700)  # fixture snapshot
    got = m._model_weight_bytes(str(tmp_path))
    assert got == 5000, f"expected only the real weights.pth (5000), got {got} (fixtures leaked in)"


def test_s1_all_fixtures_falls_through_to_hf(tmp_path, monkeypatch):
    """If the demo ships ONLY fixtures (no real local weights), the total is 0 so it falls through to
    the HF-cache lookup -- it must NOT return the fixture bytes."""
    m = _load_run()
    cap = tmp_path / "_captured" / "attn"
    cap.mkdir(parents=True)
    (cap / "args.pt").write_bytes(b"\x00" * 4242)
    called = {}

    def _fake_hf(mid):
        called["mid"] = mid
        return 999999

    monkeypatch.setattr(m, "_resolve_model_id", lambda *a, **k: "some/model")
    monkeypatch.setattr(m, "_hf_cache_weight_bytes", _fake_hf)
    got = m._model_weight_bytes(str(tmp_path), hint="some/model")
    assert got == 999999, f"fixtures-only demo did not fall through to HF cache: got {got}"
    assert called.get("mid") == "some/model"


def test_s1_no_model_name_or_bytes_hardcoded_in_scan():
    src = (_CC / "run.py").read_text()
    i = src.index("def _model_weight_bytes(")
    j = src.index("\ndef ", i + 1)
    body = src[i:j]
    assert "_captured" in body and "_stubs" in body, "the scan must skip the fixture dirs"
    # strip comment lines: an explanatory comment may cite the model/number that revealed the bug; the
    # LOGIC must not bake any of them in.
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    for junk in ("xtts", "133582652", "2086199391", "288", "512"):
        assert junk not in code.lower(), f"model/arch-specific value hardcoded in logic: {junk}"


# --------------------------------------------------------------------------- s2 / s3 (source-level: the
# perf_mcp module needs a live manifest to import, so assert the wiring at the source, like the other
# deferred-behaviour tests in this suite.)
def test_s2_config_ceiling_labelled_all_not_the_profiler_window():
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("def _persist_throughput(")
    j = src.index("\ndef ", i + 1)
    body = src[i:j]
    # the config/anchored ceiling (is_llm) must be labelled "all"; only the floor form keeps the window
    assert '"perf_layers": "all" if is_llm else _win' in body, (
        "the config ceiling must be stamped perf_layers='all' (it divides by depth-independent full-model "
        "bytes); stamping TT_PERF_LAYERS falsely trips the report's depth guard and blanks utilization"
    )
    # the floor anchor must still carry the REAL profiling window, not "all"
    assert "depth=_win," in body, "the depth-scoped FLOOR anchor must keep the real profiling window"


def test_s3_snapshot_persisted_from_termination_check():
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("def termination_check(")
    j = src.index("\ndef ", i + 1)
    body = src[i:j]
    assert "_persist_throughput(rep)" in body, (
        "termination_check must persist the roofline snapshot -- it is the tool the agent actually calls "
        "every step; profile_model (the old sole writer) is never invoked in the normal loop, so the "
        "report read a missing file and fell to NO_BAND"
    )
