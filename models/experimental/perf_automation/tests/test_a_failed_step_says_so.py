"""A step that could not do its job must say so, not return a default and carry on.

Three places did the latter, and each cost real time to find:

* profile_model persisted the roofline inputs inside a bare `except: pass`. When that failed the
  report rendered its floor-only form -- no band, no per-stage rows, no fidelity ladder -- which
  reads as a different KIND of report rather than a broken one. A day went into finding it.
* the fallback line named the missing VALUE ("no weight-bytes input") rather than the missing STEP,
  so it pointed at the model instead of at the write that never happened.
* tracy stops instrumenting after 32K source locations and saves what it has, losing roughly a
  third of the rows on a full-model forward. The run still exits 0 with a CSV, so the op breakdown
  was rendered as if it described the whole run.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_summary_quiet", PERF / "cc_optimize" / "summary.py")
_summary = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_summary)

_BUCKETS = [
    {
        "id": "matmul",
        "device_ms": 100.0,
        "count": 10,
        "bound": "slow",
        "top_ops": [{"op_code": "Matmul", "shape": "32x32x32", "device_ms": 100.0}],
    }
]


def _op_breakdown(prof):
    return "\n".join(_summary._baseline_bucket_lines(prof) or [])


def test_a_truncated_capture_declares_itself_beside_its_numbers():
    text = _op_breakdown({"buckets": _BUCKETS, "capture_truncated": "Instrumentation failure"})
    assert "INCOMPLETE" in text
    assert "Instrumentation failure" in text, "the reason must be quoted, not paraphrased"
    assert text.index("INCOMPLETE") < text.index("op class"), "the warning must precede the table"


def test_a_complete_capture_says_nothing_extra():
    """A false warning on every clean run would train the reader to ignore it."""
    text = _op_breakdown({"buckets": _BUCKETS})
    assert "INCOMPLETE" not in text
    assert _op_breakdown({"buckets": _BUCKETS, "capture_truncated": None}).count("INCOMPLETE") == 0


def test_the_truncation_reason_travels_with_the_profile():
    """Left in a log it is not read; the renderer must be able to see it."""
    from agent.tracy_tool import _capture_truncated_reason

    assert callable(_capture_truncated_reason)
    src = (PERF / "agent" / "tracy_tool.py").read_text()
    assert '"capture_truncated": _why' in src
    assert "_why = _capture_truncated_reason(profiles_dir)" in src, "the reason must be read from the log"


def test_the_roofline_fallback_names_the_step_not_just_the_value():
    """The old text named only the missing VALUE. It must still do that -- two sibling branches are
    told apart by that phrase -- and additionally name the write that never happened."""
    src = (PERF / "cc_optimize" / "summary.py").read_text()
    assert "no weight-bytes input" in src, "the phrase that distinguishes this branch must survive"
    assert "roofline inputs\n" in src or "roofline inputs " in src
    assert "were never persisted" in src, "the missing STEP must be named, not just the value"
    assert "roofline inputs NOT persisted" in src, "must point at the log line that explains it"


def test_the_persist_failure_is_reported_rather_than_swallowed():
    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("_persist_throughput(rep, prof)")
    tail = src[i : i + 1200]
    assert "roofline inputs NOT persisted" in tail, "the exception is still swallowed silently"
    assert "except Exception:\n        pass" not in tail
