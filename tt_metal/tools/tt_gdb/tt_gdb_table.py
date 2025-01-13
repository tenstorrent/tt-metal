# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
from collections import defaultdict
import json
import yaml
from typing import Union
import re

from rich.align import Align
from rich.console import Console
from rich.table import Table
from rich.highlighter import RegexHighlighter
from rich.theme import Theme
from rich.text import Text, Span
from rich.live import Live
from rich.style import Style
from curtsies import Input

from debugger_screen import DebugLayout, update_split_screen_layout


def get_functional_workers(path="tt_metal/soc_descriptors/grayskull_120_arch.yaml"):
    with open(path, "r") as f:
        functional_workers = yaml.safe_load(f)["functional_workers"]

    last_col_idx = None
    cols = []
    col = []
    # functional worker is of the form {col}-{row}, which is confusing, so we
    # are transposing the array to be of form {row}-{col}
    for functional_worker in functional_workers:
        col_idx, row_idx = functional_worker.split("-")

        if last_col_idx != col_idx:
            cols.append([])
            last_col_idx = col_idx

        cols[-1].append(f"{row_idx}-{col_idx}")

    functional_worker_arr = [[None for _ in range(len(cols))] for _ in range(len(cols[0]))]

    for col_idx, col in enumerate(cols):
        for row_idx, data in enumerate(cols[col_idx]):
            functional_worker_arr[row_idx][col_idx] = data

    return functional_worker_arr


def highlight_exact_core(
    text,
    core: str,
    style: Union[str, Style],
) -> int:
    def get_matches():
        for match in re.finditer(core, text.plain):
            start, end = match.span()
            end = end - 1  # To make it inclusive
            actual_match = True
            if start != 0:
                actual_match &= not text.plain[start - 1].isnumeric()
            if end != len(text.plain) - 1:
                actual_match &= not text.plain[end + 1].isnumeric()

            if actual_match:
                yield match

            yield None

    matches = filter(lambda x: x is not None, get_matches())
    add_span = text._spans.append
    count = 0
    _Span = Span
    for match in matches:
        start, end = match.span(0)
        add_span(_Span(start, end, style))
        count += 1
    return count


class ChipGrid:
    MIN_WIDTH = 150
    DEFAULT_COLOR = "gray"
    HIGHLIGHT_COLOR = "bold blue"
    BREAKPOINT_HIGHLIGHT_COLOR = "bold red"
    BREAKPOINT_HIGHLIGHT_COLOR_ON_TOP_OF_HIGHLIGHT_COLOR = "grey0"
    ARCH_YAML = "tt_metal/soc_descriptors/grayskull_120_arch.yaml"

    def __init__(self, cores_with_breakpoint: list = [], start_index: tuple = (0, 0)):
        self.functional_workers = get_functional_workers()

        # Where our cursor starts at within the chip grid
        self.start_index = start_index

        # We highlight these cores differently
        self.cores_with_breakpoint = defaultdict(list)
        for row_col in cores_with_breakpoint:
            row, col = row_col.split("-")
            row = int(row)
            col = int(col)
            self.cores_with_breakpoint[row].append(col)

        s = [[str(e) for e in row] for row in self.functional_workers]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = "\t".join("{{:{}}}".format(x) for x in lens)
        t = [fmt.format(*row) for row in s]
        t = [Text(text=text) for text in t]

        self.text_rows = t

        table = Table.grid()
        table.min_width = ChipGrid.MIN_WIDTH
        for i in range(len(self.functional_workers[0])):
            table.add_column()

        for row in t:
            table.add_row(row)

        self.table = table
        self.reset()

    def get_physical_coord_from_row_and_col(self, row, col):
        return [int(x) for x in self.functional_workers[row][col].split("-")]

    def reset_row(self, row):
        self.text_rows[row].spans = []

        physical_coord = self.get_physical_coord_from_row_and_col(row, 0)
        physical_row, _ = physical_coord

        for physical_col in self.cores_with_breakpoint.get(physical_row, []):
            self.highlight_index(row, physical_row, physical_col, ChipGrid.BREAKPOINT_HIGHLIGHT_COLOR)

    def reset_rows(self):
        for row in range(len(self.functional_workers)):
            self.reset_row(row)

    def reset(self):
        self.highlight_row, self.highlight_col = self.start_index
        self.reset_rows()
        self.highlight_core_at_current_index()

    def highlight_index(self, row, physical_row, physical_col, color):
        highlight_exact_core(self.text_rows[row], f"{physical_row}-{physical_col}", color)
        assert len(self.text_rows[row].spans) > 0, "Highlight index did not correctly apply"

    def highlight_core_at_current_index(self):
        row = self.highlight_row
        col = self.highlight_col

        physical_row, physical_col = self.get_physical_coord_from_row_and_col(row, col)

        if (
            physical_row in self.cores_with_breakpoint.keys()
            and physical_col in self.cores_with_breakpoint[physical_row]
        ):
            color = ChipGrid.BREAKPOINT_HIGHLIGHT_COLOR_ON_TOP_OF_HIGHLIGHT_COLOR
        else:
            color = ChipGrid.HIGHLIGHT_COLOR

        self.highlight_index(row, physical_row, physical_col, color)

    def render(self, key):
        # Reset old row back to default color
        old_row = self.highlight_row

        if key == "'KEY_UP'":
            self.highlight_row = max(0, self.highlight_row - 1)
        elif key == "'KEY_RIGHT'":
            self.highlight_col = min(len(self.functional_workers[0]) - 1, self.highlight_col + 1)
        elif key == "'KEY_DOWN'":
            self.highlight_row = min(len(self.functional_workers) - 1, self.highlight_row + 1)
        elif key == "'KEY_LEFT'":
            self.highlight_col = max(0, self.highlight_col - 1)
        else:
            return

        self.reset_row(old_row)

        # Highlight new position
        self.highlight_core_at_current_index()

    @property
    def table_view(self):
        return Align.center(self.table)


def enter_core(op: str = "", text: list = [], current_risc: str = "trisc0"):
    layout = DebugLayout(op, text, current_risc)
    return update_split_screen_layout(layout)


def core_grid(c, console, input_generator):
    with Live(screen=True, auto_refresh=False) as live:
        while True:
            live.update(c.table_view, refresh=True)
            key = repr(next(input_generator))

            # Handle special keys
            if key == repr("q"):
                return repr("q")
            elif key == repr("\n"):
                return repr("\n")

            c.render(key)


def prepare_core_text(c: ChipGrid, cores_with_breakpoint: list = [], breakpoint_lines: list = [], ops: list = []):
    """
    Docs:
    """
    text = defaultdict(str)
    physical_row, physical_col = c.get_physical_coord_from_row_and_col(c.highlight_row, c.highlight_col)
    if physical_row in c.cores_with_breakpoint.keys() and physical_col in c.cores_with_breakpoint[physical_row]:
        current_core = f"{physical_row}-{physical_col}"
        index_of_core_in_list = cores_with_breakpoint.index(current_core)
        breakpoint_lines_for_core = breakpoint_lines[index_of_core_in_list]

        for key, line in breakpoint_lines_for_core.items():
            text[key] += f"breakpoint hit on line: {line}\n"

    return text


def write_core_debug_info(core_debug_info):
    with open("core_debug_info.json", "w") as j:
        j.write(json.dumps(core_debug_info, indent=4))


def debugger(
    cores_with_breakpoint: list = [],
    ops: list = [],
    breakpoint_lines: list = [],
    start_index: tuple = (0, 0),
    current_risc: str = "trisc0",
    reenter: bool = False,
):
    assert (
        len(cores_with_breakpoint) == len(breakpoint_lines) == len(ops)
    ), "The lengths of all arguments to 'debugger' must be equal"

    c = ChipGrid(cores_with_breakpoint=cores_with_breakpoint, start_index=start_index)
    console = Console()

    with Input(keynames="curses") as input_generator:
        while True:
            if not reenter:
                special_char = core_grid(c, console, input_generator)

            if reenter or special_char == repr("\n"):
                # If we are leaving the tt_gdb debugger and we are going back into the python context, we update the start state
                # of the grid, and then enter the core that we left off at
                reenter = False

                text = prepare_core_text(c, cores_with_breakpoint, breakpoint_lines, ops)

                op = ""
                physical_row, physical_col = c.get_physical_coord_from_row_and_col(c.highlight_row, c.highlight_col)
                if (
                    physical_row in c.cores_with_breakpoint.keys()
                    and physical_col in c.cores_with_breakpoint[physical_row]
                ):
                    current_core = f"{physical_row}-{physical_col}"
                    index_of_core_in_list = cores_with_breakpoint.index(current_core)
                    op = ops[index_of_core_in_list]

                ret_vals = enter_core(op, text, current_risc=current_risc)

                # User wants to enter the debugger for a particular core
                if ret_vals is not None:
                    write_core_debug_info(
                        {
                            "op": op,
                            "current_core_x": physical_row,
                            "current_core_y": physical_col,
                            "current_risc": ret_vals.current_risc,
                            "reenter": True,
                        }
                    )
                    return

            elif special_char == repr("q"):
                write_core_debug_info({"exit": True})
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tt-gdb-table.py",
        description="Creates a terminal grid of cores with information on which cores hit breakpoints, and allows users to enter the debugger",
    )

    parser.add_argument("--cores_with_breakpoint", required=True, type=str, nargs="+")
    parser.add_argument("--breakpoint_lines", required=True, type=json.loads, nargs="+")
    parser.add_argument("--ops", required=True, type=str, nargs="+")
    parser.add_argument("--reenter", required=False, action="store_true")
    parser.add_argument("--current_risc", required=False, type=str)
    parser.add_argument("--start_index", required=False, type=int, nargs=2)
    args = parser.parse_args()

    debugger(
        cores_with_breakpoint=args.cores_with_breakpoint,
        breakpoint_lines=args.breakpoint_lines,
        ops=args.ops,
        # Optional arguments
        start_index=args.start_index if args.start_index is not None else (0, 0),
        current_risc=args.current_risc if args.current_risc is not None else "trisc0",
        reenter=args.reenter if args.reenter is not None else False,
    )
