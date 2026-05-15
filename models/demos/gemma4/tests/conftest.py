# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest"""
    parser.addoption(
        "--test-modules",
        action="store",
        default="all",
        help="Comma-separated list of modules to test. Options: all, attention, rms_norm, router, experts, shared_mlp, moe, layer, model. Example: --test-modules=attention,shared_mlp",
    )


@pytest.fixture
def test_modules(request):
    """Fixture to get the test_modules value from command line or use default 'all'"""
    return request.config.getoption("--test-modules")


@pytest.fixture(autouse=True)
def _enforce_max_prefill(request):
    """Auto-skip parametrized tests whose seq_len exceeds --max-prefill.

    Looks up the seq_len param on the test's callspec; tests without a seq_len
    parametrization are unaffected. Decode (seq_len=1) always runs.
    """
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    seq_len = callspec.params.get("seq_len")
    if not isinstance(seq_len, int):
        return
    max_prefill = request.config.getoption("--max-prefill")
    if seq_len > max_prefill:
        pytest.skip(f"seq_len={seq_len} > --max-prefill={max_prefill}")
