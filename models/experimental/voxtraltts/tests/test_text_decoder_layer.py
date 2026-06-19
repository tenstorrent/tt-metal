# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.voxtraltts.tt.text_decoder_layer import remap_voxtral_text_state_dict


def test_voxtral_decoder_layer_state_dict_remap():
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": 1,
        "model.layers.0.self_attn.k_proj.weight": 2,
        "model.layers.0.self_attn.v_proj.weight": 3,
        "model.layers.0.self_attn.o_proj.weight": 4,
        "model.layers.0.mlp.gate_proj.weight": 5,
        "model.layers.0.mlp.down_proj.weight": 6,
        "model.layers.0.mlp.up_proj.weight": 7,
    }

    remapped = remap_voxtral_text_state_dict(state_dict)
    assert "layers.0.attention.wq.weight" in remapped
    assert "layers.0.attention.wk.weight" in remapped
    assert "layers.0.attention.wv.weight" in remapped
    assert "layers.0.attention.wo.weight" in remapped
    assert "layers.0.feed_forward.w1.weight" in remapped
    assert "layers.0.feed_forward.w2.weight" in remapped
    assert "layers.0.feed_forward.w3.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.weight" not in remapped
