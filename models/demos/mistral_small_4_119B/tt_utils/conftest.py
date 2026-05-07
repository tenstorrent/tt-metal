# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures/hooks for Mistral Small 4 demo tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tme_out",
        action="store",
        type=int,
        default=120,
        help=(
            "Timeout in seconds for mistral_small_4_119B MoE experts tests. "
            "Use --tme_out=0 to run without practical timeout."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Apply test-specific timeout policy for long-running MoE experts tests."""
    tme_out = int(config.getoption("--tme_out"))
    # pytest-timeout does not expose a portable 'disable for one test' marker override.
    # Treat 0 as "run till end" by setting an effectively unbounded timeout.
    effective_timeout = 10**9 if tme_out == 0 else tme_out
    target = "models/demos/mistral_small_4_119B/tests/test_moe_experts.py"
    for item in items:
        if target in item.nodeid:
            item.add_marker(pytest.mark.timeout(effective_timeout))


@pytest.fixture(scope="session")
def mistral_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"


@pytest.fixture(scope="session")
def _mistral_text_config_store(mistral_snapshot_dir: Path):
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    cfg_path = mistral_snapshot_dir / "config.json"
    if cfg_path.is_file():
        from models.demos.mistral_small_4_119B.tt.moe.moe import mistral4_text_config_from_snapshot

        return mistral4_text_config_from_snapshot(mistral_snapshot_dir)

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
    )


@pytest.fixture
def mistral_text_config(_mistral_text_config_store):
    """Fresh ``Mistral4Config`` per test — avoids cross-test mutation on a shared session object.

    ``Mistral4Config`` can retain aliased nested structures after a single ``deepcopy`` such that
    some layers (MoE + grouped MM) produce non-finite activations; double-copy matches fully
    independent configs from ``Mistral4Config(**text_cfg)``.
    """
    return deepcopy(deepcopy(_mistral_text_config_store))


@pytest.fixture(scope="session")
def mistral_sharded_checkpoint(mistral_snapshot_dir: Path) -> Path:
    """Path to snapshot dir if index + at least one shard exist; else skip checkpoint tests."""
    idx = mistral_snapshot_dir / "model.safetensors.index.json"
    if not idx.is_file():
        pytest.skip("No model.safetensors.index.json under mistral_snapshot_dir")

    weight_map = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
    probe = "language_model.model.layers.0.self_attn.q_a_proj.weight"
    if probe not in weight_map:
        pytest.skip("Checkpoint index missing expected Mistral3 multimodal attention key")

    shard_name = weight_map[probe]
    shard_path = mistral_snapshot_dir / shard_name
    if not shard_path.is_file():
        pytest.skip(f"Shard not present locally: {shard_name} (download full weights)")

    return mistral_snapshot_dir
