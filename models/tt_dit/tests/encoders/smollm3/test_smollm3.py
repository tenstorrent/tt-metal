# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.tt_dit.encoders.smollm3.config import SmolLM3Config


def test_smollm3_config_defaults():
    c = SmolLM3Config()
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 4
    assert c.head_dim == 128
    assert c.num_hidden_layers == 36
    assert c.intermediate_size == 11008
    assert c.rope_theta == 5000000.0
    assert c.rms_norm_eps == 1e-6
    assert c.vocab_size == 128256
    assert c.attention_bias is False
    # NoPE on every 4th layer (0-indexed 3,7,...,35); 1 = apply rope, 0 = NoPE
    assert len(c.no_rope_layers) == 36
    assert c.no_rope_layers[0] == 1 and c.no_rope_layers[3] == 0 and c.no_rope_layers[7] == 0
    assert sum(c.no_rope_layers) == 27  # 36 - 9 NoPE layers
