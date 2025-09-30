# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Colors for terminal output
RST = "\033[0m"
BLUE = "\033[34m"  # For good values
RED = "\033[31m"  # For bad values
GREEN = "\033[32m"  # For instructions
GREY = "\033[37m"  # For general information
ORANGE = "\033[33m"  # For warnings
VERBOSE_CLR = "\033[94m"  # For verbose output


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
