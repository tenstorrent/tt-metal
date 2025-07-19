# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Colors for terminal output
RST = "\033[0m"
BLUE = "\033[34m"  # For good values
RED = "\033[31m"  # For bad values
GREEN = "\033[32m"  # For instructions
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
