# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA nightly CI test - runs PCC verification for 1-step inference.

This test validates that TT implementation matches CPU within expected precision tolerances.
"""

import pytest
from loguru import logger


def test_smolvla_pcc():
    """Run SmolVLA PCC verification test."""
    logger.info("Running SmolVLA PCC test")

    exit_code = pytest.main(
        [
            "models/experimental/smolvla/tests/test_smol_vla_pcc.py::TestSmolVLAPCC::test_end_to_end_pcc_1_step",
            "-v",
            "-x",  # Fail fast
        ]
    )

    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            "SmolVLA PCC test failed. TT implementation may not match CPU within expected tolerances.",
            pytrace=False,
        )
