# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from loguru import logger

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    " - <level>{message}</level>"
)

sweeps_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<bold><magenta>SWEEPS</magenta></bold> - "
    "<level>{message}</level>"
)

logger.remove()
logger.add(
    sys.stderr, level="INFO", format=logger_format, filter=lambda record: record["extra"].get("name") != "sweeps"
)
logger.add(
    sys.stderr, level="INFO", format=sweeps_format, filter=lambda record: record["extra"].get("name") == "sweeps"
)
sweeps_logger = logger.bind(name="sweeps")
