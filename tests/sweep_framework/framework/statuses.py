# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TestStatus(Enum):
    PASS = 0
    FAIL_ASSERT_EXCEPTION = 1
    FAIL_CRASH_HANG = 2
    NOT_RUN = 3
    FAIL_L1_OUT_OF_MEM = 4
    FAIL_WATCHER = 5
    FAIL_UNSUPPORTED_DEVICE_PERF = 6
    XFAIL = 7  # Expected failure - test failed as expected
    XPASS = 8  # Unexpected pass - test passed when it was expected to fail
    # Every run passed its own PCC check, but repeated executions of the same vector did not
    # produce bit-identical outputs (--determinism-runs N). PCC alone cannot see this: for
    # eltwise it is ~1.0 on every run, so this is a distinct status rather than a message.
    FAIL_NON_DETERMINISTIC = 9


class VectorValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


class VectorStatus(Enum):
    CURRENT = 0
    ARCHIVED = 1
