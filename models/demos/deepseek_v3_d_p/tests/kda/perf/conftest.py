# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the KDA perf tests in the CI environment, so local numbers compare with CI's.

Untraced (eager) time is host dispatch, and per-PID shared-memory tracking for tt-smi adds ~0.2 ms to each untraced
KDA layer call on Galaxy (~12%). CI's perf jobs export TT_METAL_SHM_TRACKING_DISABLED=1 and LOGURU_LEVEL=ERROR, so the
references assume both. They are set here before any device opens; an explicit value in the environment wins.
"""

import os
import sys

from loguru import logger

_CI_PERF_ENV = {"TT_METAL_SHM_TRACKING_DISABLED": "1", "LOGURU_LEVEL": "ERROR"}

if "LOGURU_LEVEL" not in os.environ:
    # loguru reads LOGURU_LEVEL once, at import, which has already happened: apply the level to its sink too.
    logger.remove()
    logger.add(sys.stderr, level=_CI_PERF_ENV["LOGURU_LEVEL"])
for _name, _value in _CI_PERF_ENV.items():
    os.environ.setdefault(_name, _value)
