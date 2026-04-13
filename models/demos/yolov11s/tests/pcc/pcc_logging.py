# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


def log_assert_with_pcc(component: str, expected, actual, pcc: float = 0.99) -> tuple:
    """Same as ``assert_with_pcc``, and log measured PCC so ``pytest`` output shows values."""
    passed, pcc_value = assert_with_pcc(expected, actual, pcc)
    logger.info("PCC | {} | value={} threshold={}", component, pcc_value, pcc)
    return passed, pcc_value
