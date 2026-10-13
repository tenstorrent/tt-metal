# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Startup logging policy for MoE forward-path diagnostics.

Read LOGURU_LEVEL once, matching Loguru's default DEBUG level. Guarding at the
call site skips eager tensor formatting and logger bookkeeping when debug is
disabled. Runtime sink/level changes do not update this startup policy.
"""

import os

from loguru import logger

DEBUG_LOGGING_ENABLED = logger.level(os.environ.get("LOGURU_LEVEL", "DEBUG")).no <= logger.level("DEBUG").no
