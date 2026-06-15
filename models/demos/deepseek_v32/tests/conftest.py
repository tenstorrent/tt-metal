# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Reuse all v3 fixtures/hooks (variant, weights, model download, collection rules).
import models.demos.deepseek_v3_d_p.tests.conftest as _v3_conftest
from models.demos.deepseek_v3_d_p.tests.conftest import *  # noqa: F401,F403


def pytest_configure(config):
    """Register v32 test-group markers (compose with v3's marker registration)."""
    _v3_conftest.pytest_configure(config)
    for line in (
        "dev: fast inner-loop tests (~1 min, no cold CPU truth) — run per-edit",
        "gate: full correctness matrix (CPU truths must be cached; ~10-15 min) — pre-commit/CI",
        "nightly: cold-truth builds + scale gate (big-box only; excluded by default)",
    ):
        config.addinivalue_line("markers", line)


def pytest_addoption(parser):
    """v32 MLA/indexer test knobs (weights source + input data)."""
    g = parser.getgroup("deepseek_v32")
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
        help="k_pe RoPE layout for the KV-cache reference comparison (test_vs_gpu_ref.py). "
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
