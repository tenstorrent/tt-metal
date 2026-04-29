# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dots OCR demo/benchmark console logging (does not modify shared TTNN code)."""

from __future__ import annotations

import os
import sys

from loguru import logger


def configure_dots_ocr_console_logging() -> None:
    """
    Point loguru at stderr with a sane default level so noisy TTNN DEBUG lines
    (e.g. tensor ``.tensorbin`` cache load/generate in ``ttnn.from_torch``) stay hidden.

    Override with ``DOTS_LOG_LEVEL`` (e.g. ``DEBUG`` to see those messages again).
    """
    raw = (os.environ.get("DOTS_LOG_LEVEL") or "INFO").strip().upper()
    allowed = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR")
    level = raw if raw in allowed else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level)
