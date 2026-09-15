"""Unit tests for the prior-attempt explicit framing (Bundle item #1).

Step 4's vision_neck Tier 2 test showed that iter 1 (haiku) and iter 2
(sonnet) produced the IDENTICAL IndexError. Sonnet rewrote the same
wrong reshape because it never saw haiku's exact failing line — only
an auto-derived diagnosis summary.

This module covers:
  - _extract_failing_stub_excerpt: parses traceback for last frame in
    the stub, returns (line_number, source_excerpt) with the failing
    line marked.
  - _format_attempt_history_block: renders a focused PRIOR ATTEMPT
    FAILED HERE block for the most recent entry, with the failing-line
    excerpt and the literal traceback tail.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def test_extract_returns_zero_for_empty_traceback() -> None:
    """No traceback → no excerpt. Should never raise."""
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "vision_neck.py"
        stub.write_text("x = 1\ny = 2\n")
        line, excerpt = cli._extract_failing_stub_excerpt("", stub)
        assert line == 0
        assert excerpt == ""


def test_extract_returns_zero_when_stub_missing() -> None:
    """Stub file missing → no excerpt."""
    tb = 'File "/tmp/nonexistent.py", line 5, in forward\n    x = bad()\n'
    line, excerpt = cli._extract_failing_stub_excerpt(tb, Path("/tmp/nonexistent.py"))
    assert line == 0


def test_extract_finds_failing_line_in_stub() -> None:
    """Should locate the LAST frame in the traceback that references the
    stub, return the line number, and produce a marked excerpt."""
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "vision_neck.py"
        stub.write_text(
            "def forward(self, x):\n"
            "    # line 2\n"
            "    a = self._helper_a(x)\n"
            "    # line 4\n"
            "    b = ttnn.reshape(a, [0, 11, 11, 3, 3, -1])\n"
            "    # line 6\n"
            "    return b\n"
        )
        tb = f'  File "{stub}", line 5, in forward\n    b = ttnn.reshape(a, [0, 11, 11, 3, 3, -1])\nRuntimeError: invalid shape\n'
        line, excerpt = cli._extract_failing_stub_excerpt(tb, stub)
        assert line == 5
        assert ">>> " in excerpt
        # The failing line itself should be in the excerpt
        assert "ttnn.reshape(a, [0, 11, 11, 3, 3, -1])" in excerpt
        # The line number prefix should be present (formatted to 5 wide)
        assert "    5 >>> " in excerpt


def test_extract_uses_last_frame_when_multiple_match() -> None:
    """If the traceback has multiple frames in the same stub (e.g.,
    helper calls itself), use the LAST (deepest) one."""
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "vision_neck.py"
        stub.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 20)))
        tb = (
            f'  File "{stub}", line 3, in forward\n'
            "    x = recurse()\n"
            f'  File "{stub}", line 12, in recurse\n'
            "    crash()\n"
            "Error: boom\n"
        )
        line, _ = cli._extract_failing_stub_excerpt(tb, stub)
        assert line == 12, "should pick the deepest frame in the stub"


def test_extract_ignores_unrelated_traceback_frames() -> None:
    """Frames pointing at other files (not the stub) must be ignored.
    Only frames in the stub itself contribute the failing line."""
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "vision_neck.py"
        stub.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 10)))
        tb = (
            '  File "/usr/lib/python3.10/something.py", line 99, in inner\n'
            "    raise RuntimeError\n"
            f'  File "{stub}", line 4, in forward\n'
            "    inner()\n"
            "RuntimeError\n"
        )
        line, _ = cli._extract_failing_stub_excerpt(tb, stub)
        assert line == 4


def test_extract_handles_line_beyond_stub_size() -> None:
    """If the traceback's line is past EOF of the stub (stale traceback
    against newer file), return safely with line=0."""
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "vision_neck.py"
        stub.write_text("only_three_lines\nare_here\nin_this_stub\n")
        tb = f'  File "{stub}", line 999, in forward\n    crash\n'
        line, excerpt = cli._extract_failing_stub_excerpt(tb, stub)
        assert line == 0
        assert excerpt == ""


def test_write_attempt_log_stores_failing_line_fields() -> None:
    """`_write_attempt_log` must persist the failing-line number and
    excerpt into the JSON entry so the next iter can read them."""
    import json

    with tempfile.TemporaryDirectory() as td:
        demo_dir = Path(td)
        # Build a fake stub file that pytest tracebacks would point at
        stub_path = demo_dir / "_stubs" / "vision_neck.py"
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub_path.write_text(
            "def forward(self, x):\n"
            "    a = self._helper(x)\n"
            "    b = ttnn.reshape(a, [0])  # this crashes\n"
            "    return b\n"
        )
        tb = (
            f'  File "{stub_path}", line 3, in forward\n'
            "    b = ttnn.reshape(a, [0])\n"
            "RuntimeError: invalid shape\n"
        )
        cli._write_attempt_log(
            demo_dir=demo_dir,
            component_name="vision_neck",
            iter_n=1,
            stub_path=stub_path,
            exemplar_used=None,
            model_used="haiku",
            failure_class="SHAPE",
            failure_signature="abc123",
            traceback_excerpt=tb,
        )
        log_path = cli._attempt_log_dir(demo_dir, "vision_neck") / "iter_001.json"
        assert log_path.exists()
        entry = json.loads(log_path.read_text())
        assert entry["failing_line"] == 3
        assert "ttnn.reshape(a, [0])" in entry["failing_line_excerpt"]
        assert ">>> " in entry["failing_line_excerpt"]


def test_format_block_no_history_returns_first_try_message() -> None:
    """Empty history is the common case for iter 1; must not raise."""
    out = cli._format_attempt_history_block([])
    assert "first try" in out.lower() or "no prior" in out.lower()


def test_format_block_renders_prior_attempt_section_when_excerpt_present() -> None:
    """The new PRIOR ATTEMPT FAILED HERE block must appear when the last
    history entry has a failing_line_excerpt + traceback_excerpt."""
    history = [
        {
            "iter": 1,
            "model_used": "haiku",
            "exemplar_used": "(none)",
            "ops_used": ["ttnn.reshape", "ttnn.permute"],
            "sharding_strategy": [],
            "failure_class": "SHAPE",
            "diagnosis": "shape op produced invalid target",
            "next_step": "review reshape arithmetic",
            "failing_line": 5,
            "failing_line_excerpt": (
                "    3     a = self._helper(x)\n"
                "    4     # comment\n"
                "    5 >>> b = ttnn.reshape(a, [0, 11, 11, 3, 3, -1])\n"
                "    6     return b\n"
            ),
            "traceback_excerpt": (
                'File "vision_neck.py", line 5, in forward\n'
                "    b = ttnn.reshape(a, [0, 11, 11, 3, 3, -1])\n"
                "RuntimeError: shape '[0, 11, 11, 3, 3, -1]' is invalid for input of size 691200\n"
            ),
        }
    ]
    out = cli._format_attempt_history_block(history)
    assert "YOUR PRIOR ATTEMPT FAILED AT THIS EXACT LINE" in out
    assert "ttnn.reshape(a, [0, 11, 11, 3, 3, -1])" in out
    assert "RuntimeError" in out, "literal error must be in the prompt"
    assert "iter-1" in out, "must reference which iter failed"


def test_format_block_skips_prior_attempt_section_when_excerpt_missing() -> None:
    """If the last entry has no failing-line excerpt (e.g., older log
    written before the field was added), the new section must be
    silently skipped — backwards-compat."""
    history = [
        {
            "iter": 1,
            "model_used": "sonnet",
            "exemplar_used": "(none)",
            "ops_used": ["ttnn.matmul"],
            "sharding_strategy": [],
            "failure_class": "PCC_ONLY",
            "diagnosis": "pcc below target",
            "next_step": "review numerical magnitude",
            # No failing_line / failing_line_excerpt fields
        }
    ]
    out = cli._format_attempt_history_block(history)
    assert "YOUR PRIOR ATTEMPT FAILED AT THIS EXACT LINE" not in out
    # But existing rendering must still work
    assert "Iter 1" in out
    assert "diagnosis" in out.lower()


def test_format_block_uses_only_last_entry_excerpt() -> None:
    """If history has 3 entries, only the MOST RECENT one's excerpt
    appears in the focused PRIOR ATTEMPT block — earlier ones are too
    stale to be useful (the stub has been overwritten since)."""
    history = [
        {
            "iter": 1,
            "model_used": "haiku",
            "exemplar_used": "(none)",
            "ops_used": [],
            "sharding_strategy": [],
            "failure_class": "SHAPE",
            "diagnosis": "d1",
            "next_step": "",
            "failing_line": 5,
            "failing_line_excerpt": "    5 >>> first_attempt_code\n",
            "traceback_excerpt": "Err1",
        },
        {
            "iter": 2,
            "model_used": "sonnet",
            "exemplar_used": "(none)",
            "ops_used": [],
            "sharding_strategy": [],
            "failure_class": "SHAPE",
            "diagnosis": "d2",
            "next_step": "",
            "failing_line": 7,
            "failing_line_excerpt": "    7 >>> second_attempt_code\n",
            "traceback_excerpt": "Err2",
        },
    ]
    out = cli._format_attempt_history_block(history)
    assert "second_attempt_code" in out, "must show the most recent excerpt"
    # The first attempt's excerpt should NOT appear in the focused PRIOR section.
    # (It still appears in the per-iter summary lines above.)
    prior_section_idx = out.find("YOUR PRIOR ATTEMPT FAILED")
    assert prior_section_idx != -1
    prior_section = out[prior_section_idx:]
    assert "first_attempt_code" not in prior_section
