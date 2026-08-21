# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LLK perf infrastructure.

Canonical modules for perf comparison, baselines, and gate operations.
"""

from .regression_compare import (
    DEFAULT_MIN_CYCLES,
    DEFAULT_THRESHOLD,
    compare_runs,
    render_report,
)

__all__ = [
    "compare_runs",
    "render_report",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_CYCLES",
]
