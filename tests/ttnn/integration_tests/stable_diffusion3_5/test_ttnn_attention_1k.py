# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_attention_1k import (
    ttnn_Attention as tt_module,
    ttnn_JointAttnProcessor2_0,
)
from models.experimental.functional_stable_diffusion3_5.reference.attention import Attention, JointAttnProcessor2_0
from models.experimental.functional_stable_diffusion3_5.reference.rms_norm import RMSNorm
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, Attention):
            parameters["norm_q"] = {}
            parameters["norm_q"]["weight"] = ttnn.from_torch(
                model.norm_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            parameters["norm_k"] = {}
            parameters["norm_k"]["weight"] = ttnn.from_torch(
                model.norm_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            parameters["to_q"] = {}
            parameters["to_q"]["weight"] = preprocess_linear_weight(model.to_q.weight, dtype=ttnn.bfloat8_b)
            parameters["to_q"]["bias"] = preprocess_linear_bias(model.to_q.bias, dtype=ttnn.bfloat8_b)
            parameters["to_k"] = {}
            parameters["to_k"]["weight"] = preprocess_linear_weight(model.to_k.weight, dtype=ttnn.bfloat8_b)
            parameters["to_k"]["bias"] = preprocess_linear_bias(model.to_k.bias, dtype=ttnn.bfloat8_b)
            parameters["to_v"] = {}
            parameters["to_v"]["weight"] = preprocess_linear_weight(model.to_v.weight, dtype=ttnn.bfloat8_b)
            parameters["to_v"]["bias"] = preprocess_linear_bias(model.to_v.bias, dtype=ttnn.bfloat8_b)
            if hasattr(model, "add_k_proj"):
                parameters["add_k_proj"] = {}
                parameters["add_k_proj"]["weight"] = preprocess_linear_weight(
                    model.add_k_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["add_k_proj"]["bias"] = preprocess_linear_bias(model.add_k_proj.bias, dtype=ttnn.bfloat8_b)
            if hasattr(model, "add_v_proj"):
                parameters["add_v_proj"] = {}
                parameters["add_v_proj"]["weight"] = preprocess_linear_weight(
                    model.add_v_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["add_v_proj"]["bias"] = preprocess_linear_bias(model.add_v_proj.bias, dtype=ttnn.bfloat8_b)
            if hasattr(model, "add_q_proj"):
                parameters["add_q_proj"] = {}
                parameters["add_q_proj"]["weight"] = preprocess_linear_weight(
                    model.add_q_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["add_q_proj"]["bias"] = preprocess_linear_bias(model.add_q_proj.bias, dtype=ttnn.bfloat8_b)
            parameters["to_out"] = {}
            parameters["to_out"][0] = {}
            parameters["to_out"][0]["weight"] = preprocess_linear_weight(model.to_out[0].weight, dtype=ttnn.bfloat8_b)
            parameters["to_out"][0]["bias"] = preprocess_linear_bias(model.to_out[0].bias, dtype=ttnn.bfloat8_b)
            if hasattr(model, "to_add_out"):
                parameters["to_add_out"] = {}
                parameters["to_add_out"]["weight"] = preprocess_linear_weight(
                    model.to_add_out.weight, dtype=ttnn.bfloat8_b
                )
                parameters["to_add_out"]["bias"] = preprocess_linear_bias(model.to_add_out.bias, dtype=ttnn.bfloat8_b)
            if model.norm_added_q != None:
                parameters["norm_added_q"] = {}
                parameters["norm_added_q"]["weight"] = ttnn.from_torch(
                    model.norm_added_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
            if model.norm_added_k != None:
                parameters["norm_added_k"] = {}
                parameters["norm_added_k"]["weight"] = ttnn.from_torch(
                    model.norm_added_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "attn_inputs, hidden_states, attention_mask, encoder_hidden_states",
    [
        # (
        #     {  # 512x512
        #         "query_dim": 1536,
        #         "cross_attention_dim": None,
        #         "heads": 24,
        #         "kv_heads": None,
        #         "dim_head": 64,
        #         "dropout": 0.0,
        #         "bias": True,
        #         "upcast_attention": False,
        #         "upcast_softmax": False,
        #         "cross_attention_norm": None,
        #         "cross_attention_norm_num_groups": 32,
        #         "qk_norm": "rms_norm",
        #         "added_kv_proj_dim": 1536,
        #         "added_proj_bias": True,
        #         "norm_num_groups": None,
        #         "spatial_norm_dim": None,
        #         "out_bias": True,
        #         "scale_qk": True,
        #         "only_cross_attention": False,
        #         "eps": 1e-06,
        #         "rescale_output_factor": 1.0,
        #         "residual_connection": False,
        #         "_from_deprecated_attn_block": False,
        #         "out_dim": 1536,
        #         "context_pre_only": False,
        #         "pre_only": False,
        #         "elementwise_affine": True,
        #     },
        #     torch.randn([2, 1, 1024, 1536], dtype=torch.bfloat16),
        #     None,
        #     torch.randn([2, 1, 160, 1536], dtype=torch.bfloat16),
        # ),
        # (
        #     {  # 512x512
        #         "query_dim": 1536,
        #         "cross_attention_dim": None,
        #         "heads": 24,
        #         "kv_heads": None,
        #         "dim_head": 64,
        #         "dropout": 0.0,
        #         "bias": True,
        #         "upcast_attention": False,
        #         "upcast_softmax": False,
        #         "cross_attention_norm": None,
        #         "cross_attention_norm_num_groups": 32,
        #         "qk_norm": "rms_norm",
        #         "added_kv_proj_dim": None,
        #         "added_proj_bias": True,
        #         "norm_num_groups": None,
        #         "spatial_norm_dim": None,
        #         "out_bias": True,
        #         "scale_qk": True,
        #         "only_cross_attention": False,
        #         "eps": 1e-06,
        #         "rescale_output_factor": 1.0,
        #         "residual_connection": False,
        #         "_from_deprecated_attn_block": False,
        #         "out_dim": 1536,
        #         "context_pre_only": None,
        #         "pre_only": False,
        #         "elementwise_affine": True,
        #     },
        #     torch.randn([2, 1, 1024, 1536], dtype=torch.bfloat16),
        #     None,
        #     None,
        # ),
        (
            {  # 1024x1024
                "query_dim": 1536,
                "cross_attention_dim": None,
                "heads": 24,
                "kv_heads": None,
                "dim_head": 64,
                "dropout": 0.0,
                "bias": True,
                "upcast_attention": False,
                "upcast_softmax": False,
                "cross_attention_norm": None,
                "cross_attention_norm_num_groups": 32,
                "qk_norm": "rms_norm",
                "added_kv_proj_dim": 1536,
                "added_proj_bias": True,
                "norm_num_groups": None,
                "spatial_norm_dim": None,
                "out_bias": True,
                "scale_qk": True,
                "only_cross_attention": False,
                "eps": 1e-06,
                "rescale_output_factor": 1.0,
                "residual_connection": False,
                "_from_deprecated_attn_block": False,
                "out_dim": 1536,
                "context_pre_only": False,
                "pre_only": False,
                "elementwise_affine": True,
            },
            torch.randn([2, 1, 4096, 1536], dtype=torch.bfloat16),
            None,
            torch.randn([2, 1, 352, 1536], dtype=torch.bfloat16),
        ),
        # (
        #     {  # 1024x1024
        #         "query_dim": 1536,
        #         "cross_attention_dim": None,
        #         "heads": 24,
        #         "kv_heads": None,
        #         "dim_head": 64,
        #         "dropout": 0.0,
        #         "bias": True,
        #         "upcast_attention": False,
        #         "upcast_softmax": False,
        #         "cross_attention_norm": None,
        #         "cross_attention_norm_num_groups": 32,
        #         "qk_norm": "rms_norm",
        #         "added_kv_proj_dim": None,
        #         "added_proj_bias": True,
        #         "norm_num_groups": None,
        #         "spatial_norm_dim": None,
        #         "out_bias": True,
        #         "scale_qk": True,
        #         "only_cross_attention": False,
        #         "eps": 1e-06,
        #         "rescale_output_factor": 1.0,
        #         "residual_connection": False,
        #         "_from_deprecated_attn_block": False,
        #         "out_dim": 1536,
        #         "context_pre_only": None,
        #         "pre_only": False,
        #         "elementwise_affine": True,
        #     },
        #     torch.randn([2, 1, 4096, 1536], dtype=torch.bfloat16),
        #     None,
        #     None,
        # ),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_attention(attn_inputs, device, hidden_states, attention_mask, encoder_hidden_states, reset_seeds):
    torch_sub_module = Attention(**attn_inputs, processor=JointAttnProcessor2_0()).to(dtype=torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )

    tt_input_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if encoder_hidden_states is not None:
        tt_input_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    else:
        tt_input_encoder_hidden_states = None

    tt_sub_module = tt_module(**attn_inputs, processor=ttnn_JointAttnProcessor2_0(), parameters=parameters)

    torch_hidden_states = hidden_states.squeeze(1)

    if encoder_hidden_states is not None:
        torch_encoder_hidden_states = encoder_hidden_states.squeeze(1)
    else:
        torch_encoder_hidden_states = None

    if encoder_hidden_states is not None:
        torch_out_1, torch_out_2 = torch_sub_module(torch_hidden_states, torch_encoder_hidden_states)
        tt_out_1, tt_out_2 = tt_sub_module(
            tt_input_hidden_states, tt_input_encoder_hidden_states, attention_mask, device
        )
    else:
        torch_out_1 = torch_sub_module(torch_hidden_states, torch_encoder_hidden_states)
        tt_out_1 = tt_sub_module(tt_input_hidden_states, tt_input_encoder_hidden_states, attention_mask, device)

    tt_out_in_torch_1 = ttnn.to_torch(tt_out_1).squeeze(1)

    if encoder_hidden_states is not None:
        tt_out_2_shape = tt_out_2.shape
        tt_out_in_torch_2 = ttnn.to_torch(tt_out_2).squeeze(1)
        assert_with_pcc(torch_out_2[:, :154, :], tt_out_in_torch_2[:, :154, :], 0.988)
    assert_with_pcc(torch_out_1, tt_out_in_torch_1, 0.987)
