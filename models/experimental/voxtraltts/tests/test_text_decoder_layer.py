# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.experimental.voxtraltts.tt.text_decoder_layer import (
    VoxtralTTTextDecoderLayer,
    remap_voxtral_text_state_dict,
)
from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip
from models.tt_transformers.tt.ccl import TT_CCL


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


@pytest.mark.timeout(3600)
def test_text_decoder_layer_create_with_real_checkpoint(device, reset_seeds):
    text_model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)
    args = text_model.inner.args
    state_dict = args.load_state_dict()

    layer = VoxtralTTTextDecoderLayer.create(
        args=args,
        mesh_device=device,
        tt_ccl=TT_CCL(device),
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
        transformation_mats=text_model.inner.rope_setup.get_both_trans_mats(),
    )

    assert layer.inner is not None
