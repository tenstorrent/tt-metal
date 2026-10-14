# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the GPT-OSS DFlash context-KV tail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_to_halfsplit_perm
from models.demos.gpt_oss_d_p.tt.dflash_kv import DFlashKVConfig, load_dflash_kv_weights, reference_dflash_kv
from models.demos.gpt_oss_d_p.tt.rope import build_yarn_cos_sin
from models.demos.gpt_oss_d_p.tt.runners.adapters.gpt_oss import GptOssPrefillAdapter
from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, resolve_dflash_execution

HIDDEN = 64
LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 32


def _config(path: Path) -> DFlashKVConfig:
    return DFlashKVConfig(
        checkpoint_path=path,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-5,
        rope_theta=150000.0,
        yarn_factor=32.0,
        yarn_orig_max_pos=4096,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
    )


def _weights(config: DFlashKVConfig, seed=7):
    generator = torch.Generator().manual_seed(seed)
    weights = {"hidden_norm.weight": torch.randn(HIDDEN, generator=generator)}
    for layer_idx in range(config.num_hidden_layers):
        weights[f"layers.{layer_idx}.self_attn.k_proj.weight"] = torch.randn(config.kv_dim, HIDDEN, generator=generator)
        weights[f"layers.{layer_idx}.self_attn.v_proj.weight"] = torch.randn(config.kv_dim, HIDDEN, generator=generator)
        weights[f"layers.{layer_idx}.self_attn.k_norm.weight"] = torch.randn(HEAD_DIM, generator=generator)
    return weights


def _rms(value, weight, epsilon):
    return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + epsilon) * weight


def test_reference_dflash_kv_matches_independent_context_path():
    config = _config(Path("/synthetic/not-read"))
    weights = _weights(config)
    hidden = torch.randn(1, 1, 37, HIDDEN, generator=torch.Generator().manual_seed(11))
    start_pos = 96

    actual_k, actual_v = reference_dflash_kv(hidden, weights, config, start_pos=start_pos)

    normalized = _rms(hidden, weights["hidden_norm.weight"], config.rms_norm_eps)
    src = torch.argsort(interleaved_to_halfsplit_perm(HEAD_DIM))
    cos, sin = build_yarn_cos_sin(
        start_pos + hidden.shape[-2],
        HEAD_DIM,
        rope_theta=config.rope_theta,
        yarn_factor=config.yarn_factor,
        yarn_orig_max_pos=config.yarn_orig_max_pos,
        yarn_beta_fast=config.yarn_beta_fast,
        yarn_beta_slow=config.yarn_beta_slow,
    )
    cos = cos[:, :, start_pos:]
    sin = sin[:, :, start_pos:]
    for layer_idx in range(LAYERS):
        k_weight = weights[f"layers.{layer_idx}.self_attn.k_proj.weight"]
        k = F.linear(normalized, k_weight).reshape(1, 1, 37, KV_HEADS, HEAD_DIM)[..., src]
        k = k.movedim(-2, -3)
        k = _rms(k, weights[f"layers.{layer_idx}.self_attn.k_norm.weight"][src], config.rms_norm_eps)
        rotate_half = torch.stack((-k[..., 1::2], k[..., 0::2]), dim=-1).flatten(-2)
        expected_k = k * cos + rotate_half * sin
        expected_v = (
            F.linear(normalized, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])
            .reshape(1, 1, 37, KV_HEADS, HEAD_DIM)
            .movedim(-2, -3)
        )
        torch.testing.assert_close(actual_k[layer_idx], expected_k)
        torch.testing.assert_close(actual_v[layer_idx], expected_v)


def _write_checkpoint(path: Path, *, malformed_key: str | None = None):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": HIDDEN,
                "num_hidden_layers": LAYERS,
                "num_key_value_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "rms_norm_eps": 1e-5,
                "rope_theta": 150000,
                "rope_scaling": {
                    "factor": 32,
                    "original_max_position_embeddings": 4096,
                    "beta_fast": 32,
                    "beta_slow": 1,
                },
            }
        )
    )
    weights = _weights(_config(path))
    if malformed_key is not None:
        weights[malformed_key] = torch.zeros(1)
    save_file(weights, path / "model.safetensors")


def test_checkpoint_tail_contract_validates_every_key_and_shape(tmp_path, expect_error):
    checkpoint = tmp_path / "valid"
    _write_checkpoint(checkpoint)
    config = DFlashKVConfig.from_checkpoint(
        checkpoint,
        expected_hidden_size=HIDDEN,
        expected_num_hidden_layers=LAYERS,
        expected_num_key_value_heads=KV_HEADS,
        expected_head_dim=HEAD_DIM,
    )
    assert set(load_dflash_kv_weights(config)) == set(_weights(config))

    malformed = tmp_path / "malformed"
    bad_key = "layers.1.self_attn.k_proj.weight"
    _write_checkpoint(malformed, malformed_key=bad_key)
    with expect_error(ValueError, bad_key):
        DFlashKVConfig.from_checkpoint(
            malformed,
            expected_hidden_size=HIDDEN,
            expected_num_hidden_layers=LAYERS,
            expected_num_key_value_heads=KV_HEADS,
            expected_head_dim=HEAD_DIM,
        )


@pytest.mark.parametrize(
    ("checkpoint_enabled", "handoff_requested", "kv_tail", "expected"),
    [
        (False, False, None, (False, False)),
        (True, False, None, (True, True)),
        (True, True, False, (True, False)),
        (True, False, False, (False, False)),
        (True, False, True, (True, True)),
    ],
)
def test_dflash_execution_modes_keep_feature_only_path_separate(
    checkpoint_enabled, handoff_requested, kv_tail, expected
):
    assert (
        resolve_dflash_execution(
            checkpoint_enabled=checkpoint_enabled,
            handoff_requested=handoff_requested,
            kv_tail=kv_tail,
        )
        == expected
    )


def test_dflash_tail_extends_dense_layer_ack_space():
    runtime = object.__new__(TtPrefillRuntime)
    runtime.dflash_kv_builder = None
    assert runtime.layer_ack_layers(36, 36) == (36, 36)

    runtime.dflash_kv_builder = object()
    runtime.dflash_kv_config = _config(Path("/synthetic/not-read"))
    assert runtime.layer_ack_layers(36, 36) == (36 + LAYERS, 36 + LAYERS)


def test_migration_adapter_maps_sparse_target_and_dflash_rows():
    adapter = GptOssPrefillAdapter()
    target_configs = 2 * adapter.model_config.NUM_KEY_VALUE_HEADS
    assert adapter.cache_kind(0) == "gqa"
    assert adapter.cache_kind(target_configs - 1) == "gqa"
    assert adapter.cache_kind(target_configs) == "dflash"
    assert adapter.cache_layer_rows(0, 36) == {layer: layer for layer in range(36)}
    assert adapter.cache_layer_rows(target_configs, 36) == {
        36 + layer: 36 + layer for layer in range(adapter.dflash_num_layers)
    }
    assert adapter.cache_head_dim(target_configs) == adapter.model_config.HEAD_DIM
