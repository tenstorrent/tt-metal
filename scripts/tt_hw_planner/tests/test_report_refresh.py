"""A stale, old-format bring-up section is modernized in place when a later
phase (emit-e2e / optimize) refreshes it — without disturbing sibling sections.

Guards the cross-phase report-drift bug: each phase only rewrites its own
marker block, so a report assembled across tool versions could show an old
bring-up block (emoji, "Converged after ?") next to a fresh optimize block.
refresh_bringup_section() re-renders the bring-up block in the current
professional format, derives the outcome from on-disk state, and leaves every
other section byte-for-byte intact.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rr():
    return importlib.import_module("scripts.tt_hw_planner.run_report")


def _seed_demo(demo: Path) -> None:
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "bringup_status.json").write_text(
        json.dumps(
            {
                "backend": {"name": "tt_transformers / simple_text_demo"},
                "category": "LLM",
                "new_model_type": "llama",
                "pipeline_tag": "text-generation",
                "components": [{"name": "attn", "status": "NEW"}],
            }
        )
    )


_STALE_OPTIMIZE = "<!-- BEGIN optimize -->\n" "# Optimize (perf)\n\nDO NOT TOUCH THIS BLOCK\n" "<!-- END optimize -->"


def test_refresh_modernizes_bringup_and_preserves_optimize(tmp_path):
    rr = _rr()
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    _seed_demo(demo)
    # a stale, old-format bring-up block next to a fresh optimize block
    stale_bringup = (
        "<!-- BEGIN bringup -->\n"
        "# Bring-up run report — `org/m`\n\n"
        "**Converged** after ? iteration(s).\n\n"
        "| Module | on device? |\n|---|---|\n| attn | ✅ |\n"
        "<!-- END bringup -->"
    )
    (demo / "RUN_REPORT.md").write_text(stale_bringup + "\n\n" + _STALE_OPTIMIZE + "\n")

    rr.refresh_bringup_section(demo)
    out = (demo / "RUN_REPORT.md").read_text()

    # bring-up block modernized: no emoji, no bare "?", professional glyphs
    bringup_block = out.split("<!-- BEGIN bringup -->", 1)[1].split("<!-- END bringup -->", 1)[0]
    assert "✅" not in bringup_block
    assert "after ? iteration" not in bringup_block
    assert "## Module placement (all components)" in bringup_block
    assert "[ ok ]" in bringup_block or "[ cpu ]" in bringup_block or "[wait]" in bringup_block
    # model_id recovered from the old title
    assert "org/m" in bringup_block
    # ranked siblings (#4) now present
    assert "## Sibling candidates (ranked)" in bringup_block
    # the optimize section is untouched
    assert "DO NOT TOUCH THIS BLOCK" in out


def test_refresh_creates_bringup_when_report_absent(tmp_path):
    rr = _rr()
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    _seed_demo(demo)
    rr.refresh_bringup_section(demo, model_id="org/m")
    out = (demo / "RUN_REPORT.md").read_text()
    assert "<!-- BEGIN bringup -->" in out and "## Module placement" in out
