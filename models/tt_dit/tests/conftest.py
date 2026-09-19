# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil

import pytest
import torch
from loguru import logger

_DEFAULT_PROMPT_IMAGE = "./prompt_image.png"


def pytest_addoption(parser: pytest.Parser) -> None:
    """tt_dit options. Registered whenever the run's target path is under this directory."""
    parser.addoption(
        "--prompt-image",
        action="store",
        default=None,
        metavar="PATH_OR_URL",
        help="Conditioning image for image-to-video tests: a local path or an http(s) URL. "
        f"Falls back to $WAN_I2V_IMAGE, then {_DEFAULT_PROMPT_IMAGE}.",
    )


@pytest.fixture
def prompt_image(request: pytest.FixtureRequest) -> str:
    """Conditioning image for I2V: --prompt-image, else $WAN_I2V_IMAGE, else ./prompt_image.png."""
    return request.config.getoption("--prompt-image") or os.environ.get("WAN_I2V_IMAGE") or _DEFAULT_PROMPT_IMAGE


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


num_torch_threads = max(1, os.cpu_count())
logger.info(f"Setting torch num_threads to {num_torch_threads}")
torch.set_num_threads(num_torch_threads)
