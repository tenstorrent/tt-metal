# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest plugin for the sparse-MLA (DSA) tests.

This is a real plugin module (NOT a conftest), registered via ``pytest_plugins`` in this package's
conftest (``tests/sparse_mla/conftest.py``). Living under the sparse_mla/ subtree scopes its options:
they register only when these sparse tests are collected, so dense runs (tests/test_mla.py)
never see them. Because it is referenced by its dotted name and never auto-loaded as a conftest, pytest
registers it exactly once.

It carries what the sparse suites must share: the ``--ds-*`` MLA/indexer knobs with their ``ds_*``
fixtures.
"""

import pytest


def pytest_addoption(parser):
    """DSA (GLM) MLA/indexer test knobs: weights source + input data."""
    g = parser.getgroup("deepseek_dsa")
    g.addoption(
        "--ds-layer",
        type=int,
        default=None,
        help="Pretrained transformer layer to load into MLA/indexer (default: random weights).",
    )
    g.addoption(
        "--ds-checkpoint",
        default=None,
        help="Local safetensors shard path(s), comma-separated; overrides HF resolution for --ds-layer.",
    )
    g.addoption(
        "--ds-repo",
        default=None,
        help="HF repo for pretrained weights.",
    )
    g.addoption(
        "--ds-input",
        default=None,
        help="Path to a .pt file with the MLA/indexer input hidden states [..., seq, hidden]; "
        "default is deterministic randn(seed).",
    )


@pytest.fixture
def ds_layer(request):
    return request.config.getoption("--ds-layer")


@pytest.fixture
def ds_checkpoint(request):
    val = request.config.getoption("--ds-checkpoint")
    if not val:
        return None
    paths = [p for p in val.split(",") if p]
    return paths[0] if len(paths) == 1 else paths


@pytest.fixture
def ds_repo(request):
    return request.config.getoption("--ds-repo")


@pytest.fixture
def ds_input(request):
    return request.config.getoption("--ds-input")
