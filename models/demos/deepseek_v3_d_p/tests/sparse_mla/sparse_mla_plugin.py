# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest plugin for the sparse-MLA (DSA) tests — DeepSeek V3.2 & GLM-5.1.

This is a real plugin module (NOT a conftest), registered via ``pytest_plugins`` in this package's
conftest (``tests/sparse_mla/conftest.py``). Living under the sparse_mla/ subtree scopes its options
and markers: they register only when these sparse tests are collected, so dense runs (tests/test_mla.py)
never see them. Because it is referenced by its dotted name and never auto-loaded as a conftest, pytest
registers it exactly once.

It carries what the sparse suites must share: the ``perf`` / ``trace`` markers (for the out-of-band
suites that run separately from the fast correctness matrix) and the ``--ds-*`` MLA/indexer knobs with
their ``ds_*`` fixtures.
"""

import re

import pytest


def is_marker_explicitly_selected(config, marker: str) -> bool:
    markexpr = getattr(config.option, "markexpr", "") or ""
    token = re.escape(marker)
    selected = re.search(rf"(^|[\s()]){token}($|[\s()])", markexpr)
    negated = re.search(rf"(^|[\s()])not\s+{token}($|[\s()])", markexpr)
    return bool(selected and not negated)


def pytest_configure(config):
    """Register the markers for tests that run SEPARATELY from the default correctness matrix.

    The sparse-MLA correctness suite (test_sparse_mla.py) is small/fast enough (~2 min) to run
    wholesale, so it carries no tier/intent markers. Only the out-of-band suites are marked, so CI can
    exclude them with e.g. -m "not perf and not trace".
    """
    for line in (
        "perf: per-op device-kernel timing (run under tracy, separate from correctness)",
        "trace: official trace-bundle parity tests (run separately from the correctness matrix)",
    ):
        config.addinivalue_line("markers", line)


def pytest_addoption(parser):
    """DSA (V3.2 / GLM) MLA/indexer test knobs: weights source + input data."""
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
        help="HF repo for pretrained weights (default: deepseek-ai/DeepSeek-V3.2-Exp).",
    )
    g.addoption(
        "--ds-input",
        default=None,
        help="Path to a .pt file with the MLA/indexer input hidden states [..., seq, hidden]; "
        "default is deterministic randn(seed).",
    )
    g.addoption(
        "--ds-kpe-layout",
        choices=["interleaved", "vllm"],
        default="interleaved",
        help="k_pe RoPE layout for the KV-cache reference comparison (test_sparse_mla_vs_trace.py). "
        "'interleaved' (default): our/official layout — assert latent PCC + frame-invariant k_pe L2. "
        "'vllm': reindex our k_pe to vLLM's half-split layout and assert element-wise PCC (cross-stack).",
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


@pytest.fixture
def ds_kpe_layout(request):
    return request.config.getoption("--ds-kpe-layout")
