# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


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


# Cache the result of should_use_color
_USE_COLOR = should_use_color()

# Colors for terminal output
RST = "\033[0m" if _USE_COLOR else ""
BLUE = "\033[34m" if _USE_COLOR else ""  # For good values
RED = "\033[31m" if _USE_COLOR else ""  # For bad values
GREEN = "\033[32m" if _USE_COLOR else ""  # For instructions
GREY = "\033[37m" if _USE_COLOR else ""  # For general information
ORANGE = "\033[33m" if _USE_COLOR else ""  # For warnings
VERBOSE_CLR = "\033[94m" if _USE_COLOR else ""  # For verbose output


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
        print(f"{RED}{s}{RST}", **kwargs)


def WARN(s, **kwargs):
    if Verbosity.supports(Verbosity.WARN):
        print(f"{ORANGE}{s}{RST}", **kwargs)


def DEBUG(s, **kwargs):
    if Verbosity.supports(Verbosity.DEBUG):
        print(f"{GREEN}{s}{RST}", **kwargs)


def INFO(s, **kwargs):
    if Verbosity.supports(Verbosity.INFO):
        print(f"{BLUE}{s}{RST}", **kwargs)


def VERBOSE(s, **kwargs):
    if Verbosity.supports(Verbosity.VERBOSE):
        print(f"{GREY}{s}{RST}", **kwargs)
