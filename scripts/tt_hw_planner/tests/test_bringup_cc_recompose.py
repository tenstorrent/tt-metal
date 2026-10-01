"""Pin: recompose-restore gives a previously-terminal parent a clean slate.

A composite that fell back to CPU or was harness-skipped sits in the gate's
terminal set. Recompose restores it as a whole-module target, but unless the
terminal marks are cleared the gate skips it (rounds=0) and it can never
graduate. ``_clear_terminal_state_for_recompose`` removes those marks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner._cli_helpers.bringup_cc import (  # noqa: E402
    _clear_terminal_state_for_recompose,
)


def _write_state(demo: Path, st: dict) -> Path:
    demo.mkdir(parents=True, exist_ok=True)
    p = demo / ".bringup_cc_state.json"
    p.write_text(json.dumps(st))
    return p


def test_clears_harness_skipped_and_reason(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    _write_state(
        demo,
        {
            "harness_skipped": ["g_p_t", "other"],
            "harness_skip_reason": {"g_p_t": "HF ref raised TypeError", "other": "x"},
            "attempts": {"g_p_t": 12, "other": 3},
            "consecutive_same_class": {"g_p_t": 6},
        },
    )
    cleared = _clear_terminal_state_for_recompose(demo, "g_p_t")
    assert "harness_skipped" in cleared
    st = json.loads((demo / ".bringup_cc_state.json").read_text())
    assert st["harness_skipped"] == ["other"]
    assert "g_p_t" not in st["harness_skip_reason"]
    assert st["harness_skip_reason"]["other"] == "x"
    assert "g_p_t" not in st["attempts"]
    assert st["attempts"]["other"] == 3
    assert "g_p_t" not in st["consecutive_same_class"]


def test_clears_fallback(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    _write_state(demo, {"fallback": ["g_p_t"], "attempts": {"g_p_t": 5}})
    cleared = _clear_terminal_state_for_recompose(demo, "g_p_t")
    assert "fallback" in cleared
    st = json.loads((demo / ".bringup_cc_state.json").read_text())
    assert st["fallback"] == []
    assert "g_p_t" not in st.get("attempts", {})


def test_noop_when_parent_not_terminal(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    _write_state(demo, {"harness_skipped": ["other"], "attempts": {"other": 2}})
    cleared = _clear_terminal_state_for_recompose(demo, "g_p_t")
    assert cleared == []
    st = json.loads((demo / ".bringup_cc_state.json").read_text())
    assert st["harness_skipped"] == ["other"]


def test_missing_state_file_is_safe(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    demo.mkdir()
    assert _clear_terminal_state_for_recompose(demo, "g_p_t") == []
