# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Negative control for ``test_time_layer_masked_long`` (D5 item 2): the same test with the masked-SDPA chunk cap
forced to 256. The test is real if this run fails.

    ttq run --model t0-gamma --part 3 --kind pcc -- python models/experimental/t0_forecast/tests/perf/neg_chunk256.py
"""

import sys

import pytest

from models.experimental.t0_forecast.tt import attention


class _Cap256:
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        attention._MAX_MASKED_CHUNK = 256


if __name__ == "__main__":
    test = "models/experimental/t0_forecast/tests/pcc/test_modules.py::test_time_layer_masked_long"
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", test], plugins=[_Cap256()]))
