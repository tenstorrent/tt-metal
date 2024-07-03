# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TestStatus(Enum):
    PASS = 0
    FAIL_ASSERT_EXCEPTION = 1
    FAIL_CRASH_HANG = 2
    NOT_RUN = 3
