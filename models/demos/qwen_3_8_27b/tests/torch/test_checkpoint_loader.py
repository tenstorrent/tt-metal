# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 (host only): the real-checkpoint loader — prefix filtering, key set per layer type, shapes and
dtype against config.json, loud failure on a missing key. The checkpoint is plain bf16, so there is
no dequantization to check (the recipe's mxfp4 loader row does not apply)."""

import pytest
import torch

from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader


@pytest.fixture(scope="module")
def reader():
    try:
        return CheckpointReader()
    except (RuntimeError, FileNotFoundError) as e:
        pytest.skip(str(e))


def test_text_model_key_set(reader):
    text = [k for k in reader.weight_map if k.startswith("model.language_model.")]
    layers = {int(k.split(".")[3]) for k in text if ".layers." in k}
    assert layers == set(range(QWEN38.num_hidden_layers))
    assert "lm_head.weight" in reader.weight_map


@pytest.mark.parametrize("i", [0, 3])
def test_layer_shapes(reader, i):
    c = QWEN38
    sd = reader.layer(i)
    H = c.hidden_size
    want = {
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
        "mlp.gate_proj.weight": (c.intermediate_size, H),
        "mlp.up_proj.weight": (c.intermediate_size, H),
        "mlp.down_proj.weight": (H, c.intermediate_size),
    }
    if c.is_full_attention(i):
        want.update(
            {
                "self_attn.q_proj.weight": (2 * c.num_attention_heads * c.head_dim, H),
                "self_attn.k_proj.weight": (c.num_key_value_heads * c.head_dim, H),
                "self_attn.v_proj.weight": (c.num_key_value_heads * c.head_dim, H),
                "self_attn.o_proj.weight": (H, c.num_attention_heads * c.head_dim),
                "self_attn.q_norm.weight": (c.head_dim,),
                "self_attn.k_norm.weight": (c.head_dim,),
            }
        )
    else:
        nv = c.linear_num_value_heads
        want.update(
            {
                "linear_attn.in_proj_qkv.weight": (c.conv_dim, H),
                "linear_attn.in_proj_z.weight": (c.linear_value_dim, H),
                "linear_attn.in_proj_a.weight": (nv, H),
                "linear_attn.in_proj_b.weight": (nv, H),
                "linear_attn.conv1d.weight": (c.conv_dim, 1, c.linear_conv_kernel_dim),
                "linear_attn.A_log": (nv,),
                "linear_attn.dt_bias": (nv,),
                "linear_attn.norm.weight": (c.linear_value_head_dim,),
                "linear_attn.out_proj.weight": (H, c.linear_value_dim),
            }
        )
    assert set(sd) == set(want), set(sd) ^ set(want)
    for k, shape in want.items():
        assert tuple(sd[k].shape) == shape, (k, sd[k].shape, shape)
        assert sd[k].dtype in (torch.bfloat16, torch.float32), (k, sd[k].dtype)


def test_missing_key_fails_loudly(reader, expect_error):
    with expect_error(KeyError, "not in checkpoint"):
        reader.text("layers.999.mlp.up_proj.weight")
