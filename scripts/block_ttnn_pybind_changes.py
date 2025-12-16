#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Pre-commit hook that blocks changes to pybind11 sources under ttnn/.
Used while nanobind bindings fully replace the legacy pybind11 layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence
from loguru import logger


FREEZE_MESSAGE = "Legacy pybind11 bindings are deprecated. Use nanobind bindings."
PYBIND_TOKEN = "pybind"


def _normalize_paths(paths: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw in paths:
        path = Path(raw)
        normalized.append(str(path.as_posix()))
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Abort commits that modify or add files under ttnn/cpp/ttnn-pybind."))
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths supplied by pre-commit (relative to repository root).",
    )
    return parser.parse_args(argv)


def _offending_paths(paths: Sequence[str]) -> list[str]:
    frozen: list[str] = []
    for path in _normalize_paths(paths):
        if PYBIND_TOKEN in path.lower():
            frozen.append(path)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    offending_paths = _offending_paths(args.paths)

    if not offending_paths:
        return 0

    logger.error(FREEZE_MESSAGE)
    logger.error("Blocked files:")
    for rel_path in offending_paths:
        logger.error(f"  - {rel_path}")

    logger.error("Please move any new binding work to the nanobind implementation.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
