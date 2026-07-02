# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Console banner / table formatting helpers for the training entry point."""

from __future__ import annotations

import os
import re
from pathlib import Path

HEADER_WIDTH = 70
# Content wraps within a 2-space margin on each side of the table.
_CONTENT_WIDTH = HEADER_WIDTH - 4


def _print_banner(title: str) -> None:
    print("=" * HEADER_WIDTH)
    print(f"  {title}")
    print("=" * HEADER_WIDTH)


# Break a long value after whitespace or a path-like separator (/ . _ -), keeping the
# separator on the preceding line so paths and config names wrap at natural boundaries.
_WRAP_PIECE = re.compile(r"[^\s/._-]*[\s/._-]+|[^\s/._-]+")


def _wrap(text: str, width: int) -> list[str]:
    """Wrap `text` to `width`, preferring separator boundaries; hard-breaks tokens too long to fit. Always ≥1 line."""
    if len(text) <= width:
        return [text]
    lines, cur = [], ""
    for piece in _WRAP_PIECE.findall(text):
        if cur and len(cur) + len(piece) > width:
            lines.append(cur)
            cur = piece
        else:
            cur += piece
        while len(cur) > width:  # oversized spaceless token: hard-break at the edge
            lines.append(cur[:width])
            cur = cur[width:]
    if cur:
        lines.append(cur)
    return [line.rstrip() for line in lines]


def _print_section(name: str, fields: list[tuple[str, str]] | str, label_width: int | None = None) -> None:
    """Emit `[NAME] ----` divider, then 2-space-indented `key  value` rows (or bare lines if `fields` is a string).

    Content wraps within a 2-space margin on both sides: string fields wrap back to the left indent,
    kv values wrap back to the column where the value starts. When `label_width` is provided, the key
    column is right-justified to that width — used to align value columns across multiple sections.
    """
    prefix = f"[{name.upper()}] "
    print()
    print(prefix + "-" * (HEADER_WIDTH - len(prefix)))
    if isinstance(fields, str):
        for line in fields.split("\n"):
            for chunk in _wrap(line, _CONTENT_WIDTH):
                print(f"  {chunk}")
        return
    if not fields:
        return
    width = label_width if label_width is not None else max(len(k) for k, _ in fields)
    value_width = max(1, _CONTENT_WIDTH - width - 2)  # minus label column + 2-space gap
    for label, value in fields:
        chunks = _wrap(value, value_width)
        print(f"  {label.rjust(width)}  {chunks[0]}")
        for chunk in chunks[1:]:
            print(f"  {' ' * width}  {chunk}")


def _print_header_close() -> None:
    print()
    print("=" * HEADER_WIDTH)
    print()


def print_header(title: str, sections: list[tuple[str, list[tuple[str, str]] | str]]) -> None:
    """Print the full opening banner: title, then each section sharing one label width so values align vertically."""
    kv_widths = [max(len(k) for k, _ in fields) for _, fields in sections if isinstance(fields, list) and fields]
    width = max(kv_widths) if kv_widths else 0
    _print_banner(title)
    for name, fields in sections:
        _print_section(name, fields, label_width=width)
    _print_header_close()


def print_footer(title: str, fields: list[tuple[str, str]]) -> None:
    """Closing banner mirroring `_print_banner`: title rule, aligned `key  value` rows, closing rule."""
    width = max((len(k) for k, _ in fields), default=0)
    print()
    print("=" * HEADER_WIDTH)
    print(f"  {title}")
    for label, value in fields:
        print(f"  {label.rjust(width)}  {value}")
    print("=" * HEADER_WIDTH)
    print()


def shorten_home(path: str) -> str:
    """Replace the user's home directory prefix in `path` with `~` for display."""
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path
