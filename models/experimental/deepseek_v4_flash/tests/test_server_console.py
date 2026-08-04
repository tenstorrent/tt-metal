# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the server's live console (``demo/tui.py``).

The console draws over the alternate screen, so its own key handling is the only way to
read back through the log. These cover that scrolling without a terminal: the painter's
rendering pass is called directly and the resulting window is compared against the log.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rich", reason="the live console needs rich")

from models.experimental.deepseek_v4_flash.demo import tui  # noqa: E402

HEIGHT = 10


def make_console(lines: int = 100) -> tui.ServerConsole:
    view = tui.ServerConsole(lambda: {})
    for i in range(lines):
        view._append(f"t{i}", "INFO", f"line {i}")
    view._log_lines(HEIGHT)  # the painter has always drawn a frame before a key arrives,
    return view  # which is what tells a page key how tall a page is


def window(view: tui.ServerConsole, height: int = HEIGHT) -> list[str]:
    """The log lines the next frame would show, oldest first."""
    view._log_lines(height)  # the render pass fixes the height and clamps the scroll
    lines = list(view._display)
    end = len(lines) - view._scroll
    return [text for _, _, text in lines[max(end - height, 0) : end]]


def test_the_log_follows_the_newest_line_by_default() -> None:
    assert window(make_console())[-1] == "line 99"


@pytest.mark.parametrize(
    "keys, newest",
    [
        (["up"], "line 98"),
        (["up", "up"], "line 97"),
        (["k", "k", "k"], "line 96"),
        (["pageup"], "line 90"),  # a screen back, less the overlapping line
        (["pageup", "pagedown"], "line 99"),
        (["up", "down"], "line 99"),
        (["pageup", "end"], "line 99"),
    ],
)
def test_keys_move_the_window(keys: list[str], newest: str) -> None:
    view = make_console()
    for key in keys:
        view._on_key(key)
    assert window(view)[-1] == newest


def test_home_reaches_the_oldest_line_and_end_returns() -> None:
    view = make_console()
    view._on_key("home")
    assert window(view)[0] == "line 0"
    view._on_key("end")
    assert window(view)[-1] == "line 99"


def test_arrivals_do_not_slide_the_view_while_scrolled_back() -> None:
    """Reading back is useless if incoming lines keep pushing the text off the screen."""
    view = make_console()
    view._on_key("pageup")
    before = window(view)

    for i in range(5):
        view._incoming.put(("t", "INFO", f"new {i}"))
    added = view._drain()
    view._scroll += added  # what the painter does when scrolled

    assert added == 5
    assert window(view) == before


def test_a_console_note_holds_the_view_too() -> None:
    view = make_console()
    view._on_key("pageup")
    before = window(view)
    view._on_key("d")  # toggling debug logs a line of its own
    assert window(view) == before


def test_the_tail_still_picks_up_arrivals() -> None:
    view = make_console()
    view._incoming.put(("t", "INFO", "newest"))
    view._drain()
    assert window(view)[-1] == "newest"


def test_scrolling_stops_at_the_oldest_line() -> None:
    view = make_console()
    for _ in range(500):
        view._on_key("up")
    assert window(view)[0] == "line 0"
    assert view._scroll == 100 - HEIGHT


def test_a_log_shorter_than_the_screen_shows_whole() -> None:
    view = make_console(5)
    view._on_key("home")
    assert window(view) == [f"line {i}" for i in range(5)]
    assert view._scroll == 0


def test_clearing_returns_to_following() -> None:
    view = make_console()
    view._on_key("pageup")
    view._on_key("c")
    assert view._scroll == 0
    assert window(view) == []


def test_one_long_record_fills_the_pane_instead_of_overflowing_it() -> None:
    """A logged prompt is thousands of characters: as one row it would outgrow the frame.

    The pane is budgeted in screen lines, so such a record folds into many of them and
    the window still shows exactly as many as fit.
    """
    view = tui.ServerConsole(lambda: {})
    width = view._message_width()
    view._append("t0", "INFO", "x" * (width * 40))

    assert len(view._display) == 40, "the record was not folded to the column width"
    assert len(window(view)) == HEIGHT, "the pane showed more lines than it has room for"


def test_a_long_record_can_be_scrolled_through() -> None:
    """Scrolling has to reach inside one record, not just between records."""
    view = tui.ServerConsole(lambda: {})
    width = view._message_width()
    view._append("t0", "INFO", "".join(f"{i:<{width}}" for i in range(40)))  # 40 numbered lines
    view._log_lines(HEIGHT)

    assert window(view)[-1].startswith("39")
    view._on_key("home")
    assert window(view)[0].startswith("0")


def test_the_frame_never_grows_past_the_terminal() -> None:
    """Whatever is logged, the status header must not be pushed off the screen."""
    view = tui.ServerConsole(lambda: {"model_id": "m", "users": [], "active": []})
    view._append("t0", "INFO", "x" * 20000)

    rendered = view.console.render_lines(view._frame(), view.console.options, pad=False)
    assert len(rendered) <= view.console.size.height, f"frame is {len(rendered)} lines"


def test_a_resize_refolds_the_log() -> None:
    view = tui.ServerConsole(lambda: {})
    view._append("t0", "INFO", "y" * 400)
    before = len(view._display)

    # Both dimensions: rich ignores a width set on its own.
    view.console.size = (max(view.console.width // 2, 40), view.console.size.height)
    view._log_lines(HEIGHT)

    assert len(view._display) > before, "narrower console should need more lines"
    assert "".join(text for _, _, text in view._display) == "y" * 400, "refolding lost text"


@pytest.mark.parametrize(
    "sequence, name",
    [("[A", "up"), ("OA", "up"), ("[B", "down"), ("[5~", "pageup"), ("[6~", "pagedown"), ("[H", "home"), ("[F", "end")],
)
def test_navigation_escape_sequences_are_recognised(sequence: str, name: str) -> None:
    """Terminals send the arrow and page keys as escape sequences, in either cursor mode."""
    assert tui._KEY_SEQUENCES[sequence] == name
