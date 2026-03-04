# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the log-start plugin only when output capture is disabled."""
    if config.option.capture == "no":
        config.pluginmanager.register(_LogStartPlugin(), "tt_dit_logstart")


class _LogStartPlugin:
    @staticmethod
    def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:  # noqa: ARG004
        parts = nodeid.split("::")
        filename = parts[0].rsplit("/", 1)[-1]
        rest = "::".join(parts[1:])

        # split off params from last part
        params = []
        if "[" in rest:
            rest, param_str = rest.split("[", 1)
            params = param_str.rstrip("]").split("-")

        dim = "\033[2m"
        bold = "\033[1m"
        reset = "\033[0m"

        label = f"{dim}{filename}{reset}  {bold}{rest}{reset}"
        if params:
            label += f"  {dim}({', '.join(params)}){reset}"

        raw = f"{filename}  {rest}"
        if params:
            raw += f"  ({', '.join(params)})"

        width = shutil.get_terminal_size()[0]
        print(f"\n\n{'━' * width}\n{label}\n")  # noqa: T201
