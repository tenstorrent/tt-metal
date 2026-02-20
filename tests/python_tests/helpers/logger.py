# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Centralized logging configuration for LLK test infrastructure.

Uses loguru for structured, leveled logging with color support.
The log level can be controlled via:
  1. The --loguru-level pytest CLI option (highest priority)
  2. The LOGURU_LEVEL environment variable
  3. Defaults to INFO

Usage:
    from helpers.logger import logger

    logger.debug("Detailed diagnostic info")
    logger.info("General test progress")
    logger.warning("Something unexpected but non-fatal")
    logger.error("Test failure details")

Log levels (from most to least verbose):
    TRACE -> DEBUG -> INFO -> SUCCESS -> WARNING -> ERROR -> CRITICAL

Output behavior:
    - Logs are written to test_run.log (overwritten each session).
    - Errors (ERROR+) are appended to test_errors.log (persists across runs).
    - Under pytest-xdist, each worker writes to its own log files
      (e.g. test_run_gw0.log, test_errors_gw0.log) to avoid clobbering.
    - Terminal output uses pytest's live logging (log_cli) which integrates
      with pytest-sugar. Use --loguru-level=INFO to see logs in the terminal.
"""

import logging
import os
import re

from loguru import logger

# Remove loguru's default handler so we can configure our own
logger.remove()

# Compact format for file output (no colors)
_PLAIN_FORMAT = "{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

# Error log format with date for the persistent error log
_ERROR_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Regex to strip ANSI escape codes from messages written to files
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(message: str) -> str:
    return _ANSI_RE.sub("", message)


def _file_format(record):
    """Format function that strips ANSI codes from the message for file sinks."""
    record["message"] = _strip_ansi(record["message"])
    return _PLAIN_FORMAT + "\n"


def _error_file_format(record):
    """Format function that strips ANSI codes for the persistent error log."""
    record["message"] = _strip_ansi(record["message"])
    return _ERROR_FORMAT + "\n"


class _PropagateHandler(logging.Handler):
    """Bridge loguru messages into Python's stdlib logging.

    This lets pytest's live logging (--log-cli-level) display loguru messages,
    which integrates cleanly with pytest-sugar's progress bar.
    """

    def emit(self, record):
        logging.getLogger(record.name).handle(record)


def _xdist_worker_suffix() -> str:
    """Return a filename suffix like '_gw0' when running under pytest-xdist."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return f"_{worker}" if worker else ""


def configure_logger(level: str = None):
    """Configure the loguru logger for the test session.

    Under pytest-xdist each worker gets its own log files (e.g.
    test_run_gw0.log) so parallel workers never clobber each other.

    Args:
        level: Log level string (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from LOGURU_LEVEL env var, defaulting to INFO.
    """
    logger.remove()

    level = (level or os.getenv("LOGURU_LEVEL", "INFO")).upper()

    suffix = _xdist_worker_suffix()

    # Propagate to stdlib logging so pytest's --log-cli-level can display them.
    # pytest's live logging integrates properly with pytest-sugar.
    logger.add(_PropagateHandler(), format="{message}", level=level)

    # Session log - full log for this run (overwritten each session)
    logger.add(
        f"test_run{suffix}.log",
        format=_file_format,
        level=level,
        mode="w",
        colorize=False,
    )

    # Persistent error log - appends ERROR+ across runs so failures aren't lost
    logger.add(
        f"test_errors{suffix}.log",
        format=_error_file_format,
        level="ERROR",
        mode="a",
        colorize=False,
    )

    logger.debug(
        "Logger configured with level={}, worker={}", level, suffix or "controller"
    )


# Apply initial configuration (can be reconfigured later from conftest.py)
configure_logger()
