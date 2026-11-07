# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1's real checkpoint loads into the state dicts the K3 stack reads. Host only, no device.

Reads two layers' non-expert weights (one KDA, one DSA attention layer), a few hundred MB. The indexer
must land under `TtIndexer.WEIGHT_NAMES` at the host shapes `TtIndexer` expects; otherwise `ttMLA` would
take the layer as cache-only and load placeholders.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import MLA_KEYS, load_layer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

_KDA_LAYER = 0
_DSA_LAYER = KimiK31Config.mla_layer_ids()[0]


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = Path(os.getenv(KimiK31Adapter.env_var) or KimiK31Adapter.hf_model_default)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"K3.1 checkpoint not on this box ({path}); set ${KimiK31Adapter.env_var}")
    return path


def test_dsa_layer_loads_mla_and_indexer(checkpoint):
    state = load_layer_state_dict(checkpoint, _DSA_LAYER, model_cfg=KimiK31Config)
    assert "kda_weights" not in state
    mla = state["mla_weights"]
    assert set(MLA_KEYS) <= set(mla)
    assert TtIndexer.has_host_weights(mla)

    cfg = kimi_k3_1_hf_config()
    expected = {
        "indexer.wq_b.weight": (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
        "indexer.wk.weight": (cfg.index_head_dim, cfg.hidden_size),
        "indexer.weights_proj.weight": (cfg.index_n_heads, cfg.hidden_size),
        "indexer.k_norm.weight": (cfg.index_head_dim,),
        "indexer.k_norm_bias.weight": (cfg.index_head_dim,),
    }
    assert {name: tuple(mla[name].shape) for name in expected} == expected
    # LayerNorm weight and bias are distinct tensors, not one read twice.
    assert not mla["indexer.k_norm.weight"].equal(mla["indexer.k_norm_bias.weight"])


def test_kda_layer_loads_kda_weights(checkpoint):
    state = load_layer_state_dict(checkpoint, _KDA_LAYER, model_cfg=KimiK31Config)
    assert "mla_weights" not in state
    assert state["kda_weights"]
    assert set(state["ffn_weights"]) == {"gate_proj", "up_proj", "down_proj"}
