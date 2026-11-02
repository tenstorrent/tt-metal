# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only: how DFlashDrafterConfig.from_hf_config reads a drafter's rope config."""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

GLM53_DFLASH2_ROPE = {"rope_theta": 1000000, "rope_type": "default"}
KIMI_K27_DFLASH_ROPE = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 64.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "rope_theta": 50000.0,
    "rope_type": "deepseek_yarn",
    "type": "deepseek_yarn",
}


def _hf_config(rope_parameters):
    # transformers 4.53 keeps rope_parameters as given and leaves rope_theta at Qwen3's 1e4 default.
    return SimpleNamespace(
        hidden_size=6144,
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_hidden_layers=6,
        num_target_layers=78,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=rope_parameters,
        dflash_config={"target_layer_ids": [5, 19, 33, 47, 61, 75]},
        architectures=["DFlash2DraftModel"],
    )


@pytest.mark.parametrize(
    "rope_parameters, theta, factor",
    [(GLM53_DFLASH2_ROPE, 1e6, 1.0), (KIMI_K27_DFLASH_ROPE, 5e4, 64.0)],
    ids=["glm53-unscaled", "kimi-k27-yarn"],
)
def test_rope_parse(rope_parameters, theta, factor):
    cfg = DFlashDrafterConfig.from_hf_config(_hf_config(rope_parameters))
    assert (cfg.rope_theta, cfg.rope_factor) == (theta, factor)
