# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from collections import defaultdict, namedtuple

from rich import print as rich_print
from rich.layout import Layout
from rich.panel import Panel
from rich.style import Style
from rich.live import Live
from rich.align import Align
from curtsies import Input


risc_to_color = {"trisc0": "yellow", "trisc1": "red", "trisc2": "purple", "ncrisc": "green", "brisc": "blue"}


risc_to_risc = {
    "trisc0": {"'KEY_DOWN'": "trisc1", "'KEY_RIGHT'": "ncrisc"},
    "trisc1": {"'KEY_UP'": "trisc0", "'KEY_DOWN'": "trisc2", "'KEY_RIGHT'": "ncrisc"},
    "trisc2": {"'KEY_UP'": "trisc1", "'KEY_RIGHT'": "brisc"},
    "ncrisc": {"'KEY_LEFT'": "trisc0", "'KEY_DOWN'": "brisc"},
    "brisc": {"'KEY_UP'": "ncrisc", "'KEY_LEFT'": "trisc2"},
}

risc_to_layout_side = {"trisc0": "left", "trisc1": "left", "trisc2": "left", "ncrisc": "right", "brisc": "right"}


def create_layout_with_panel(risc_name, highlight_risc=False, text=""):
    if highlight_risc:
        title = f"[grey0]{risc_name}"
    else:
        title = risc_name

    return Panel(text, title=title, border_style=Style(color=risc_to_color[risc_name]))


def get_split_screen_layout(text, current_risc):
    layout = Layout()

    layout.split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    layout["right"].split(
        Layout(
            create_layout_with_panel("ncrisc", highlight_risc="ncrisc" == current_risc, text=text.get("ncrisc", "")),
            name="ncrisc",
        ),
        Layout(
            create_layout_with_panel("brisc", highlight_risc="brisc" == current_risc, text=text.get("brisc", "")),
            name="brisc",
        ),
    )

    layout["left"].split(
        Layout(
            create_layout_with_panel("trisc0", highlight_risc="trisc0" == current_risc, text=text.get("trisc0", "")),
            name="trisc0",
        ),
        Layout(
            create_layout_with_panel("trisc1", highlight_risc="trisc1" == current_risc, text=text.get("trisc1", "")),
            name="trisc1",
        ),
        Layout(
            create_layout_with_panel("trisc2", highlight_risc="trisc2" == current_risc, text=text.get("trisc2", "")),
            name="trisc2",
        ),
    )

    return layout


class DebugLayout:
    def __init__(self, op: str = "", text: dict = {}, current_risc: str = "trisc0"):
        self.text = text
        self.layout = get_split_screen_layout(self.text, current_risc)
        self.op = op
        self.current_risc = current_risc

    def render(self, key):
        old_risc = self.current_risc
        new_risc = risc_to_risc[self.current_risc].get(key, old_risc)

        self.current_risc = new_risc

        old_layout = self.layout[risc_to_layout_side[old_risc]]
        old_layout[old_risc].update(
            create_layout_with_panel(risc_name=old_risc, highlight_risc=False, text=self.text[old_risc])
        )

        new_layout = self.layout[risc_to_layout_side[new_risc]]
        new_layout[new_risc].update(
            create_layout_with_panel(risc_name=new_risc, highlight_risc=True, text=self.text[new_risc])
        )


def update_split_screen_layout(layout):
    with Input(keynames="curses") as input_generator:
        with Live(layout.layout, screen=True) as live:
            while True:
                key = repr(next(input_generator))
                if key == repr("q"):
                    return
                elif key == repr("\n"):
                    # Go back to cpp context... need to provide core, risc, and op
                    ret_vals = namedtuple("ret_vals", "current_risc")
                    return ret_vals(layout.current_risc)
                layout.render(key)
                live.update(layout.layout)


if __name__ == "__main__":
    layout = DebugLayout()
    update_split_screen_layout(layout)
