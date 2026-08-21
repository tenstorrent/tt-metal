# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in gate for the perf reports the examples in this tree write.

Every example measures on device and can render its findings as a report.md that is
checked into the repo next to the example. Those numbers are box- and arch-specific,
so an ordinary test run must not touch them: otherwise each CI run and each local run
leaves a spurious diff, and the committed numbers silently become whoever ran last.

Refreshing a report is therefore opt-in, three equivalent ways:

    pytest --write-reports tests/ttnn/unit_tests/operations/examples
    EXAMPLES_WRITE_REPORTS=1 pytest tests/ttnn/unit_tests/operations/examples
    MCT_REPORT=/tmp/mine.md  pytest tests/ttnn/unit_tests/operations/examples

The first two refresh every checked-in report; the last redirects one example somewhere
harmless. Naming a per-example path is itself the opt-in, so it needs no extra flag.

Reports are off by default, and `report_target()` returns None then -- callers skip the
write and just log the table, which is all a pass/fail run needs.
"""

import os
from pathlib import Path

ENABLE_VAR = "EXAMPLES_WRITE_REPORTS"

# Treat an explicit falsey value as off, so EXAMPLES_WRITE_REPORTS=0 does what it reads like.
_OFF = {"", "0", "false", "no", "off"}


def reports_enabled() -> bool:
    """True when the caller asked for the checked-in reports to be refreshed."""
    return os.environ.get(ENABLE_VAR, "").strip().lower() not in _OFF


def report_target(env_var: str, default: "str | Path") -> "Path | None":
    """Where this example should write its report, or None to skip writing entirely.

    env_var is the example's own path override (MCT_REPORT, RAF_REPORT, ...); default is
    the checked-in report it maintains.
    """
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        return Path(explicit)
    return Path(default) if reports_enabled() else None
