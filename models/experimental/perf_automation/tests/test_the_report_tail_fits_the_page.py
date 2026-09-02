"""The report's closing sections must be laid out like the rest of it.

Three tail sections -- limitations, reproduce, legend -- ended the report in unruled prose while
every section above them is drawn to a fixed width, and the legend was a single 191-character line
against a 100-wide page. Every terminal folded it at whatever width it happened to be, so a report
that is otherwise carefully aligned finished on a ragged paragraph.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_summary_tail", PERF / "cc_optimize" / "summary.py")
_summary = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_summary)

_SRC = (PERF / "cc_optimize" / "summary.py").read_text(encoding="utf-8")


def test_the_page_width_is_named_once_not_copied():
    """Three places held their own copy of the number and a fourth had none."""
    assert _summary._REPORT_W == 100
    assert '─" * 100' not in _SRC, "a hardcoded rule width is still present"


def test_the_legend_is_wrapped_to_the_page():
    """The legend text must be split by the renderer, not by whatever terminal shows it."""
    assert "textwrap.wrap(" in _SRC
    assert "width=_REPORT_W - 2" in _SRC, "the wrap must follow the page width, not a literal"


def test_the_tail_sections_are_ruled_like_every_other_section():
    for heading in ("Limitations / suggested manual next steps", "Reproduce", "Legend"):
        i = _SRC.index('lines.append("%s")' % heading)
        after = _SRC[i : i + 220]
        assert '"─" * _REPORT_W' in after, "%s is not ruled" % heading


def test_the_legend_still_explains_every_mark_the_table_can_print():
    """A legend that drops a mark is worse than a long one."""
    for mark in ("✓win", "·try", "·wedge", "— = not attempted"):
        assert mark in _SRC, mark


def test_no_rendered_legend_line_overflows_the_page():
    import textwrap

    body = (
        "✓win = new best so far   ·try = measured no-gain   " "·wedge = wedged/crashed when tried   — = not attempted"
    )
    lines = textwrap.wrap(
        body,
        width=_summary._REPORT_W - 2,
        initial_indent="  marks:   ",
        subsequent_indent="           ",
        break_long_words=False,
    )
    assert lines, "the legend rendered empty"
    assert max(len(x) for x in lines) <= _summary._REPORT_W
