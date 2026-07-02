# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config tests + reconciliation against the real in-repo gemma-4 configs (#47461).

These read the gemma4 ``configs/*/config.json`` shipped in the repo (always
present, no checkpoint/HW needed) and assert ``config.py`` stays in sync — the
guard the #47461 weight-mapping pass relies on to catch renamed/missing keys.
"""

import json
import os

import pytest

from models.experimental.diffusion_gemma.config import DiffusionGemmaConfig, TextConfig

# Repo root: honor TT_METAL_HOME, else derive from this file's location so the
# config-drift guard below actually runs wherever the repo is checked out (a
# personal-home fallback silently skips the guard when TT_METAL_HOME is unset).
REPO = os.environ.get("TT_METAL_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
CFG_26B = os.path.join(REPO, "models/demos/gemma4/configs/gemma-4-26B-A4B-it/config.json")


def test_max_context_is_256k():
    assert DiffusionGemmaConfig().max_context == 262144  # 256 canvas * 1024 blocks


def test_full_attention_layer_count_matches_pattern():
    tc = TextConfig()
    full_layers = [i for i in range(tc.num_hidden_layers) if (i + 1) % tc.sliding_window_pattern == 0]
    assert full_layers == [5, 11, 17, 23, 29]  # matches 26B-A4B layer_types


@pytest.mark.skipif(not os.path.exists(CFG_26B), reason="in-repo 26B-A4B config not found")
def test_from_hf_config_matches_target_26b():
    hf = json.load(open(CFG_26B))
    tc = TextConfig.from_hf_config(hf)

    # parsed from the real target config.json
    assert tc.num_hidden_layers == 30
    assert tc.hidden_size == 2816
    assert (tc.num_attention_heads, tc.num_key_value_heads) == (16, 8)
    assert tc.head_dim == 256
    assert tc.vocab_size == 262144
    assert tc.intermediate_size == 2112
    assert tc.sliding_window == 1024
    assert tc.sliding_window_pattern == 6  # derived from layer_types
    assert tc.rms_norm_eps == 1e-6
    assert tc.final_logit_softcapping == 30.0
    assert tc.num_experts == 128
    assert tc.moe_intermediate_size == 704
    assert (tc.num_global_key_value_heads, tc.global_head_dim) == (2, 512)
    # NB: this reads the gemma-4-26B-A4B *base* config (the backbone we reuse), which
    # carries attention_k_eq_v=True. The DiffusionGemma config itself omits the key
    # (DG derives K=V tying from layer geometry); we validate the backbone value here.
    assert tc.attention_k_eq_v is True
    assert tc.hidden_activation == "gelu_pytorch_tanh"


@pytest.mark.skipif(not os.path.exists(CFG_26B), reason="in-repo 26B-A4B config not found")
def test_hand_written_defaults_stay_in_sync_with_config_json():
    # config.py defaults must equal what we parse from the shipped config.json,
    # so the two never silently diverge.
    hf = json.load(open(CFG_26B))
    parsed = TextConfig.from_hf_config(hf)
    defaults = TextConfig()
    for field in [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "intermediate_size",
        "sliding_window",
        "sliding_window_pattern",
        "rms_norm_eps",
        "final_logit_softcapping",
        "num_experts",
        "moe_intermediate_size",
        "num_global_key_value_heads",
        "global_head_dim",
        "attention_k_eq_v",
        "hidden_activation",
    ]:
        assert getattr(parsed, field) == getattr(defaults, field), f"{field} drift vs config.json"
