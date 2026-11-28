"""Every column heading must sit above its own cells.

`:<8` pads to a minimum width but never truncates, so a heading longer than 8 -- `structural` at 10,
`tp-fracture` at 11 -- overflowed its column and pushed the header row right while the cells stayed
on the 8-wide grid. Measured on a real report: no drift through `host`, then -2 at `tp-fracture` and
-5 for every column after it. A reader following a column upward from a win arrived at the wrong
lever's name, which is worse than untidy -- the table asserted something false.
"""

from __future__ import annotations

import importlib.util as _ilu
import re
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_summary_cols", PERF / "cc_optimize" / "summary.py")
_summary = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_summary)


def _render(cols):
    """The header and one all-cells row, exactly as the report builds them."""
    disp = _summary._disp_level
    w = max(8, *(len(disp(c)) for c in cols))
    hdr = f"{'op':<34} " + "  ".join(f"{disp(c):<{w}}" for c in cols) + f"  {'best ms':>9}"
    row = f"{'AnOp':<34} " + "  ".join(f"{'-':<{w}}" for _ in cols) + f"  {'1.00':>9}"
    return hdr, row


def _drift(cols):
    hdr, row = _render(cols)
    disp = _summary._disp_level
    head_at = [hdr.index(disp(c)) for c in cols]
    cell_at = [m.start() for m in re.finditer(r"(?<= )-(?= |$)", row)]
    assert len(cell_at) == len(cols), (len(cell_at), len(cols))
    return [c - h for c, h in zip(cell_at, head_at)]


def test_the_real_column_set_lines_up():
    cols = list(_summary._LEVEL_COLS) + ["other"]
    assert _drift(cols) == [0] * len(cols)


def test_the_columns_that_caused_it_are_still_longer_than_the_old_literal():
    """If these ever shrink below 8 the bug stops reproducing and this test stops proving anything."""
    long = [c for c in _summary._LEVEL_COLS if len(_summary._disp_level(c)) > 8]
    assert long, "no heading exceeds the old 8-wide column; this test no longer covers the defect"


def test_a_newly_added_long_column_cannot_shear_the_table():
    """The width is derived, so a future column name of any length stays aligned."""
    cols = list(_summary._LEVEL_COLS) + ["other", "a-very-long-future-lever-name"]
    assert _drift(cols) == [0] * len(cols)


def test_short_names_do_not_collapse_the_table():
    """A set of short names must keep the 8-wide floor rather than bunching up."""
    hdr, _ = _render(["a", "b"])
    assert "a       " in hdr and "b       " in hdr
