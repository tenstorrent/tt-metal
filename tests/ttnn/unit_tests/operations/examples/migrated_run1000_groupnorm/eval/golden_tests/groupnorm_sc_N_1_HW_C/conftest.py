# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for groupnorm_sc_N_1_HW_C golden tests.

Registry-model golden tests carry no marker vocabulary — cells are
identified by parametrize ids, not pytest markers. Only `numerics` needs
registration (used by test_regression.py) so pytest doesn't warn about
unknown marks.
"""


import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numerics: numerical-stability / data-distribution regression tests — "
        "not registry-driven, run unconditionally in the full suite",
    )


def pytest_collection_modifyitems(items):
    # Open the device once per module for this single-device op's tests.
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
