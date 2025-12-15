# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from rich.theme import Theme
from rich.style import Style


def should_use_color() -> bool:
    """
    Determine if color output should be enabled based on environment.

    Returns:
        bool: True if colors should be used, False otherwise.

    Checks:
        - TT_TRIAGE_COLOR environment variable (0 = disabled, other values = enabled)
        - Whether stdout is connected to a TTY (terminal)
    """
    # Respect TT_TRIAGE_COLOR environment variable
    color_env = os.environ.get("TT_TRIAGE_COLOR")
    if color_env is not None:
        return color_env != "0"

    # Check if output is going to a terminal
    return sys.stdout.isatty()


def create_console_theme(disable_colors: bool) -> Theme:
    """Create a Rich theme for console output based on color support."""
    blue = Style(color="blue")
    red = Style(color="red")
    green = Style(color="green")
    grey = Style(color="grey85")
    yellow = Style(color="yellow")
    styles: dict[str, str | Style] = {
        "command": green,  # Command that user should execute
        "debug": green,  # Debug messages
        "info": blue,  # Informational messages
        "error": red,  # Error messages
        "status": blue,  # Status messages
        "warning": yellow,  # Warning messages
        "verbose": grey,  # Verbose output messages
        "progress.tasks": "gray50",  # Progress task numbers
        "progress.description": "cyan",  # Progress description
        "blue": blue,
        "red": red,
        "green": green,
        "grey": grey,
        "yellow": yellow,
    }
    if disable_colors or not should_use_color():
        for key in styles.keys():
            styles[key] = ""
    return Theme(styles)


# Tabulate format for displaying tables
from tabulate import TableFormat, Line, DataRow

DEFAULT_TABLE_FORMAT = TableFormat(
    lineabove=Line("╭", "─", "┬", "╮"),
    linebelowheader=Line("├", "─", "┼", "┤"),
    linebetweenrows=None,
    linebelow=Line("╰", "─", "┴", "╯"),
    headerrow=DataRow("│", "│", "│"),
    datarow=DataRow("│", "│", "│"),
    padding=1,
    with_header_hide=None,
)


# Verbosity and logging methods
from enum import Enum


class Verbosity(Enum):
    NONE = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    VERBOSE = 4
    DEBUG = 5

    @staticmethod
    def set(verbosity: "Verbosity | int") -> None:
        """Set the verbosity level of messages shown.

        Args:
            verbosity (int): Verbosity level.
                1: ERROR
                2: WARN
                3: INFO
                4: VERBOSE
                5: DEBUG
        """
        global VERBOSITY_VALUE

        VERBOSITY_VALUE = Verbosity(verbosity)

    @staticmethod
    def get() -> "Verbosity":
        """Get the verbosity level of messages shown.

        Returns:
            int: Verbosity level.
                1: ERROR
                2: WARN
                3: INFO
                4: VERBOSE
                5: DEBUG
        """
        global VERBOSITY_VALUE

        return VERBOSITY_VALUE

    @staticmethod
    def supports(verbosity: "Verbosity") -> bool:
        """Check if the verbosity level is supported and should be printed.

        Returns:
            bool: True if supported, False otherwise.
        """
        global VERBOSITY_VALUE

        return VERBOSITY_VALUE.value >= verbosity.value


VERBOSITY_VALUE: Verbosity = Verbosity.INFO


def ERROR(s, **kwargs):
    if Verbosity.supports(Verbosity.ERROR):
        from triage import console

        console.print(f"[error]{s}[/]", **kwargs)


def WARN(s, **kwargs):
    if Verbosity.supports(Verbosity.WARN):
        from triage import console

        console.print(f"[warning]{s}[/]", **kwargs)


def DEBUG(s, **kwargs):
    if Verbosity.supports(Verbosity.DEBUG):
        from triage import console

        console.print(f"[debug]{s}[/]", **kwargs)


def INFO(s, **kwargs):
    if Verbosity.supports(Verbosity.INFO):
        from triage import console

        console.print(f"[info]{s}[/]", **kwargs)


def VERBOSE(s, **kwargs):
    if Verbosity.supports(Verbosity.VERBOSE):
        from triage import console

        console.print(f"[verbose]{s}[/]", **kwargs)
