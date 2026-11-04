# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""otherlayers: keeps a chain hold alive for OLS_SLEEP seconds (no device): lets the next jobs of the chain be appended after a code change."""

import os
import time


def test_sleep():
    time.sleep(int(os.environ.get("OLS_SLEEP", "240")))
