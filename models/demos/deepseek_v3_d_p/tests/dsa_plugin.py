# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest plugin for the DeepSeek family MLA/DSA tests (v3.1 / V3.2 / GLM).

This is a real plugin module (NOT a conftest), loaded by both the v3.1 and the V3.2 tests/ conftests
via ``pytest_plugins = ["models.demos.deepseek_v3_d_p.tests.dsa_plugin"]``. Because it is referenced
by its dotted name and never auto-loaded as a conftest, pytest registers it exactly once even when
both sibling suites are collected together — so the options/markers are registered a single time
(listing it in two conftests would otherwise double-register and error).

It carries everything the two suites must share: the tier markers (dev ⊆ gate ⊆ nightly) and the
``--ds-*`` MLA/indexer knobs with their ``ds_*`` fixtures.
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
    """Register the shared tier markers (dev ⊆ gate ⊆ nightly) and the test-group markers."""
    for line in (
        # tier
        "dev: fast inner-loop tests (~1 min, no cold CPU truth) — run per-edit",
        "gate: full correctness matrix (CPU truths must be cached; ~10-15 min) — pre-commit/CI",
        "nightly: cold-truth builds + scale gate (big-box only; excluded by default)",
        # group (intent) — orthogonal to tier; select with e.g. -m "accuracy and gate"
        "accuracy: output / KV PCC vs the reference",
        "determinism: same input -> same output across repeated runs",
        "feature_chunking: chunked prefill == single-shot",
        "feature_cache: KV / index cache correctness",
        "mesh: SP x TP distribution coverage",
        "perf: per-op device-kernel timing",
        "trace: official trace-bundle parity tests",
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
