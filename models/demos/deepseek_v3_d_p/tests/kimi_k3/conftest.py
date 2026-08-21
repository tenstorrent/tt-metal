# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for Kimi-K3 tests gated against the captured vLLM trace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.tests.kimi_k3.trace import KimiK3Trace

KIMI_K3_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="session")
def kimi_k3_checkpoint_dir() -> Path:
    """The Hugging Face checkpoint the trace was captured from."""
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint")
    return Path(value)


@pytest.fixture(scope="session")
def kimi_k3_trace() -> KimiK3Trace:
    """The captured trace, or a skip when it is absent or unreadable.

    Payload readability is worth probing up front: the trace directories are only group-readable
    if whoever published them made them so, and safetensors surfaces EACCES as a missing file,
    which reads as a corrupt trace rather than as a permission problem.
    """
    value = os.getenv("KIMI_K3_TRACE")
    if value is None:
        pytest.skip("set KIMI_K3_TRACE to the Kimi-K3 vLLM trace directory")
    root = Path(value)
    if not (root / "tensor_mapping.json").is_file():
        raise FileNotFoundError(f"{root} is not a trace directory: no tensor_mapping.json")
    trace = KimiK3Trace(root)
    unreadable = sorted(stream.name for stream in trace.streams.values() if not os.access(stream.path, os.R_OK))
    if unreadable:
        pytest.skip(f"{root} holds unreadable streams: {unreadable}")
    return trace


@pytest.fixture(scope="session")
def kimi_k3_tt_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the ttnn weight cache goes.

    Layer 0 alone caches ~1.4 GB of dense-FFN weights, so point ``KIMI_K3_TT_CACHE`` at a
    persistent directory to pay that once; the tmp fallback rebuilds it every run.
    """
    value = os.getenv("KIMI_K3_TT_CACHE")
    return Path(value) if value else tmp_path_factory.mktemp("kimi_k3_tt_cache")
