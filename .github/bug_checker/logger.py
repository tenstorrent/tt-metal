"""Loguru-based logger for bug_checker — colorful, structured output."""

import os
import sys

from loguru import logger

bug_checker_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<bold><yellow>BUG-CHECK</yellow></bold> - "
    "<level>{message}</level>"
)

LOGURU_ENV_VAR = "LOGURU_LEVEL"
LOGURU_DEFAULT_LEVEL = "INFO"

level = os.environ.get(LOGURU_ENV_VAR, LOGURU_DEFAULT_LEVEL)

logger.remove()
logger.add(sys.stderr, level=level, format=bug_checker_format)


def set_verbose():
    """Reconfigure logger to DEBUG level for --verbose mode."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format=bug_checker_format)
