# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time
import numpy as np

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import (
    skip_for_grayskull,
    is_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.joint_transformer_block import (
    JointTransformerBlock,
    Attention,
    SD35AdaLayerNormZeroX,
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    FeedForward,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_joint_transformer_block import (
    ttnn_JointTransformerBlock,
)

from models.experimental.functional_stable_diffusion3_5.ttnn_unopt.ttnn_joint_transformer_block import (
    ttnn_JointTransformerBlock as ttnn_JointTransformerBlock_unopt,
)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, JointTransformerBlock):
            if isinstance(model.norm1, SD35AdaLayerNormZeroX):
                parameters["norm1"] = {}
                parameters["norm1"]["linear"] = {}
                parameters["norm1"]["linear"]["weight"] = preprocess_linear_weight(
                    model.norm1.linear.weight, dtype=ttnn.bfloat8_b
                )
                parameters["norm1"]["linear"]["bias"] = preprocess_linear_bias(
                    model.norm1.linear.bias, dtype=ttnn.bfloat8_b
                )

                # Its none as elementwise_affine=False
                parameters["norm1"]["norm"] = {}
            elif isinstance(model.norm1, AdaLayerNormZero):
                parameters["norm1"] = {}
                parameters["norm1"]["linear"] = {}
                parameters["norm1"]["linear"]["weight"] = preprocess_linear_weight(
                    model.norm1.linear.weight, dtype=ttnn.bfloat8_b
                )
                parameters["norm1"]["linear"]["bias"] = preprocess_linear_bias(
                    model.norm1.linear.bias, dtype=ttnn.bfloat8_b
                )

                # Its none as elementwise_affine=False
                parameters["norm1"]["norm"] = {}

            if isinstance(model.norm1_context, AdaLayerNormZero):
                parameters["norm1_context"] = {}
                parameters["norm1_context"]["linear"] = {}
                parameters["norm1_context"]["linear"]["weight"] = preprocess_linear_weight(
                    model.norm1_context.linear.weight, dtype=ttnn.bfloat8_b
                )
                parameters["norm1_context"]["linear"]["bias"] = preprocess_linear_bias(
                    model.norm1_context.linear.bias, dtype=ttnn.bfloat8_b
                )

                # Its none as elementwise_affine=False
                parameters["norm1_context"]["norm"] = {}
            elif isinstance(model.norm1_context, AdaLayerNormContinuous):
                parameters["norm1_context"] = {}
                parameters["norm1_context"]["linear"] = {}
                parameters["norm1_context"]["linear"]["weight"] = preprocess_linear_weight(
                    model.norm1_context.linear.weight, dtype=ttnn.bfloat8_b
                )
                parameters["norm1_context"]["linear"]["bias"] = preprocess_linear_bias(
                    model.norm1_context.linear.bias, dtype=ttnn.bfloat8_b
                )

                # Its none as elementwise_affine=False
                parameters["norm"] = {}

            parameters["attn"] = {}
            parameters["attn"]["norm_q"] = {}
            parameters["attn"]["norm_q"]["weight"] = ttnn.from_torch(
                model.attn.norm_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            parameters["attn"]["norm_k"] = {}
            parameters["attn"]["norm_k"]["weight"] = ttnn.from_torch(
                model.attn.norm_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            parameters["attn"]["to_q"] = {}
            parameters["attn"]["to_q"]["weight"] = preprocess_linear_weight(
                model.attn.to_q.weight, dtype=ttnn.bfloat8_b
            )
            parameters["attn"]["to_q"]["bias"] = preprocess_linear_bias(model.attn.to_q.bias, dtype=ttnn.bfloat8_b)
            parameters["attn"]["to_k"] = {}
            parameters["attn"]["to_k"]["weight"] = preprocess_linear_weight(
                model.attn.to_k.weight, dtype=ttnn.bfloat8_b
            )
            parameters["attn"]["to_k"]["bias"] = preprocess_linear_bias(model.attn.to_k.bias, dtype=ttnn.bfloat8_b)
            parameters["attn"]["to_v"] = {}
            parameters["attn"]["to_v"]["weight"] = preprocess_linear_weight(
                model.attn.to_v.weight, dtype=ttnn.bfloat8_b
            )
            parameters["attn"]["to_v"]["bias"] = preprocess_linear_bias(model.attn.to_v.bias, dtype=ttnn.bfloat8_b)
            if hasattr(model.attn, "add_k_proj"):
                parameters["attn"]["add_k_proj"] = {}
                parameters["attn"]["add_k_proj"]["weight"] = preprocess_linear_weight(
                    model.attn.add_k_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn"]["add_k_proj"]["bias"] = preprocess_linear_bias(
                    model.attn.add_k_proj.bias, dtype=ttnn.bfloat8_b
                )
            if hasattr(model.attn, "add_v_proj"):
                parameters["attn"]["add_v_proj"] = {}
                parameters["attn"]["add_v_proj"]["weight"] = preprocess_linear_weight(
                    model.attn.add_v_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn"]["add_v_proj"]["bias"] = preprocess_linear_bias(
                    model.attn.add_v_proj.bias, dtype=ttnn.bfloat8_b
                )
            if hasattr(model.attn, "add_q_proj"):
                parameters["attn"]["add_q_proj"] = {}
                parameters["attn"]["add_q_proj"]["weight"] = preprocess_linear_weight(
                    model.attn.add_q_proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn"]["add_q_proj"]["bias"] = preprocess_linear_bias(
                    model.attn.add_q_proj.bias, dtype=ttnn.bfloat8_b
                )
            parameters["attn"]["to_out"] = {}
            parameters["attn"]["to_out"][0] = {}
            parameters["attn"]["to_out"][0]["weight"] = preprocess_linear_weight(
                model.attn.to_out[0].weight, dtype=ttnn.bfloat8_b
            )
            parameters["attn"]["to_out"][0]["bias"] = preprocess_linear_bias(
                model.attn.to_out[0].bias, dtype=ttnn.bfloat8_b
            )
            if hasattr(model.attn, "to_add_out"):
                parameters["attn"]["to_add_out"] = {}
                parameters["attn"]["to_add_out"]["weight"] = preprocess_linear_weight(
                    model.attn.to_add_out.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn"]["to_add_out"]["bias"] = preprocess_linear_bias(
                    model.attn.to_add_out.bias, dtype=ttnn.bfloat8_b
                )
            if model.attn.norm_added_q != None:
                parameters["attn"]["norm_added_q"] = {}
                parameters["attn"]["norm_added_q"]["weight"] = ttnn.from_torch(
                    model.attn.norm_added_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
            if model.attn.norm_added_k != None:
                parameters["attn"]["norm_added_k"] = {}
                parameters["attn"]["norm_added_k"]["weight"] = ttnn.from_torch(
                    model.attn.norm_added_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )

            if model.attn2 != None:
                parameters["attn2"] = {}
                parameters["attn2"]["norm_q"] = {}
                parameters["attn2"]["norm_q"]["weight"] = ttnn.from_torch(
                    model.attn2.norm_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                parameters["attn2"]["norm_k"] = {}
                parameters["attn2"]["norm_k"]["weight"] = ttnn.from_torch(
                    model.attn2.norm_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                parameters["attn2"]["to_q"] = {}
                parameters["attn2"]["to_q"]["weight"] = preprocess_linear_weight(
                    model.attn2.to_q.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_q"]["bias"] = preprocess_linear_bias(
                    model.attn2.to_q.bias, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_k"] = {}
                parameters["attn2"]["to_k"]["weight"] = preprocess_linear_weight(
                    model.attn2.to_k.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_k"]["bias"] = preprocess_linear_bias(
                    model.attn2.to_k.bias, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_v"] = {}
                parameters["attn2"]["to_v"]["weight"] = preprocess_linear_weight(
                    model.attn2.to_v.weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_v"]["bias"] = preprocess_linear_bias(
                    model.attn2.to_v.bias, dtype=ttnn.bfloat8_b
                )
                if hasattr(model.attn2, "add_k_proj"):
                    parameters["attn2"]["add_k_proj"] = {}
                    parameters["attn2"]["add_k_proj"]["weight"] = preprocess_linear_weight(
                        model.attn2.add_k_proj.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["attn2"]["add_k_proj"]["bias"] = preprocess_linear_bias(
                        model.attn2.add_k_proj.bias, dtype=ttnn.bfloat8_b
                    )
                if hasattr(model.attn2, "add_v_proj"):
                    parameters["attn2"]["add_v_proj"] = {}
                    parameters["attn2"]["add_v_proj"]["weight"] = preprocess_linear_weight(
                        model.attn2.add_v_proj.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["attn2"]["add_v_proj"]["bias"] = preprocess_linear_bias(
                        model.attn2.add_v_proj.bias, dtype=ttnn.bfloat8_b
                    )
                if hasattr(model.attn2, "add_q_proj"):
                    parameters["attn2"]["add_q_proj"] = {}
                    parameters["attn2"]["add_q_proj"]["weight"] = preprocess_linear_weight(
                        model.attn2.add_q_proj.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["attn2"]["add_q_proj"]["bias"] = preprocess_linear_bias(
                        model.attn2.add_q_proj.bias, dtype=ttnn.bfloat8_b
                    )
                parameters["attn2"]["to_out"] = {}
                parameters["attn2"]["to_out"][0] = {}
                parameters["attn2"]["to_out"][0]["weight"] = preprocess_linear_weight(
                    model.attn2.to_out[0].weight, dtype=ttnn.bfloat8_b
                )
                parameters["attn2"]["to_out"][0]["bias"] = preprocess_linear_bias(
                    model.attn2.to_out[0].bias, dtype=ttnn.bfloat8_b
                )
                if hasattr(model.attn2, "to_add_out"):
                    parameters["attn2"]["to_add_out"] = {}
                    parameters["attn2"]["to_add_out"]["weight"] = preprocess_linear_weight(
                        model.attn2.to_add_out.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["attn2"]["to_add_out"]["bias"] = preprocess_linear_bias(
                        model.attn2.to_add_out.bias, dtype=ttnn.bfloat8_b
                    )
                if model.attn2.norm_added_q != None:
                    parameters["attn2"]["norm_added_q"] = {}
                    parameters["attn2"]["norm_added_q"]["weight"] = ttnn.from_torch(
                        model.attn2.norm_added_q.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                    )
                if model.attn2.norm_added_k != None:
                    parameters["attn2"]["norm_added_k"] = {}
                    parameters["attn2"]["norm_added_k"]["weight"] = ttnn.from_torch(
                        model.attn2.norm_added_k.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                    )

            parameters["norm2"] = {}  # Its none as elementwise_affine=False

            # FeedForward
            parameters["ff"] = {}
            parameters["ff"]["net"] = {}
            parameters["ff"]["net"][0] = {}
            parameters["ff"]["net"][0] = {}
            parameters["ff"]["net"][0]["proj"] = {}
            parameters["ff"]["net"][0]["proj"]["weight"] = preprocess_linear_weight(
                model.ff.net[0].proj.weight, dtype=ttnn.bfloat8_b
            )
            parameters["ff"]["net"][0]["proj"]["bias"] = preprocess_linear_bias(
                model.ff.net[0].proj.bias, dtype=ttnn.bfloat8_b
            )
            parameters["ff"]["net"][1] = {}
            parameters["ff"]["net"][2] = {}
            parameters["ff"]["net"][2]["weight"] = preprocess_linear_weight(
                model.ff.net[2].weight, dtype=ttnn.bfloat8_b
            )
            parameters["ff"]["net"][2]["bias"] = preprocess_linear_bias(model.ff.net[2].bias, dtype=ttnn.bfloat8_b)

            if model.norm2_context != None:
                parameters["norm2_context"] = {}  # Its none as elementwise_affine=False

            if model.ff_context != None:
                parameters["ff_context"] = {}
                parameters["ff_context"]["net"] = {}
                parameters["ff_context"]["net"][0] = {}
                parameters["ff_context"]["net"][0] = {}
                parameters["ff_context"]["net"][0]["proj"] = {}
                parameters["ff_context"]["net"][0]["proj"]["weight"] = preprocess_linear_weight(
                    model.ff_context.net[0].proj.weight, dtype=ttnn.bfloat8_b
                )
                parameters["ff_context"]["net"][0]["proj"]["bias"] = preprocess_linear_bias(
                    model.ff_context.net[0].proj.bias, dtype=ttnn.bfloat8_b
                )
                parameters["ff_context"]["net"][1] = {}
                parameters["ff_context"]["net"][2] = {}
                parameters["ff_context"]["net"][2]["weight"] = preprocess_linear_weight(
                    model.ff_context.net[2].weight, dtype=ttnn.bfloat8_b
                )
                parameters["ff_context"]["net"][2]["bias"] = preprocess_linear_bias(
                    model.ff_context.net[2].bias, dtype=ttnn.bfloat8_b
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "context_pre_only,use_dual_attention",
    [
        # (False, True),
        # (False, False),
        (True, False)
    ],
)
def test_joint_transformer_block(device, reset_seeds, context_pre_only, use_dual_attention):
    reference_model = JointTransformerBlock(
        dim=1536,
        num_attention_heads=24,
        attention_head_dim=64,
        context_pre_only=context_pre_only,
        qk_norm="rms_norm",
        use_dual_attention=use_dual_attention,
    ).to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_input_hidden_states = torch.randn(2, 1024, 1536, dtype=torch.bfloat16)
    torch_input_encoder_hidden_states = torch.randn(2, 154, 1536, dtype=torch.bfloat16)
    torch_input_temb = torch.randn(2, 1536, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    torch_output = reference_model(torch_input_hidden_states, torch_input_encoder_hidden_states, torch_input_temb)

    #########
    i = 23
    torch_input_hidden_states = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__TR_in0_layer_"
            + str(i)
            + ".npy"
        )
    )  # .to(torch.bfloat16)
    torch_input_encoder_hidden_states = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__TR_in1_layer_"
            + str(i)
            + ".npy"
        )
    )  # .to(torch.bfloat16)
    torch_input_temb = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__TR_in2_layer_"
            + str(i)
            + ".npy"
        )
    )  # .to(torch.bfloat16)

    #####
    torch_input_x_unsqueezed = torch_input_hidden_states.unsqueeze(1)

    ttnn_model = ttnn_JointTransformerBlock(
        dim=1536,
        num_attention_heads=24,
        attention_head_dim=64,
        context_pre_only=context_pre_only,
        qk_norm="rms_norm",
        use_dual_attention=use_dual_attention,
        parameters=parameters,
    )

    for i in range(1):
        ttnn_input_temb = ttnn.from_torch(
            torch_input_temb.unsqueeze(1).unsqueeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_encoder_hidden_states = ttnn.from_torch(
            torch_input_encoder_hidden_states.unsqueeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_hidden_states = ttnn.from_torch(
            torch_input_x_unsqueezed,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # input_memory_config
        )

        t0 = time.time()
        ttnn_output = ttnn_model(
            ttnn_input_hidden_states, ttnn_input_encoder_hidden_states, ttnn_input_temb, parameters=parameters
        )
        # enable_persistent_kernel_cache()
        t1 = time.time()
        print("Time (sec):", t1 - t0)

    if context_pre_only != True:
        ttnn_output_0 = ttnn.to_torch(ttnn_output[0])
        ttnn_output_1 = ttnn.to_torch(ttnn_output[1])
        # assert_with_pcc(torch_output[0].unsqueeze(1), ttnn_output_0, pcc=0.999)
        # assert_with_pcc(torch_output[1].unsqueeze(1), ttnn_output_1, pcc=0.999)
        ttnn.deallocate(ttnn_output[0])
        ttnn.deallocate(ttnn_output[1])
    else:
        ttnn_output_1 = ttnn.to_torch(ttnn_output[1])
        # assert_with_pcc(torch_output[1].unsqueeze(1), ttnn_output_1, pcc=0.999)
        ttnn.deallocate(ttnn_output[1])

    ttnn.deallocate(ttnn_input_temb)
    ttnn.deallocate(ttnn_input_encoder_hidden_states)
    ttnn.deallocate(ttnn_input_hidden_states)

    print("+++++")
    ################
    ttnn_model_unopt = ttnn_JointTransformerBlock_unopt(
        dim=1536,
        num_attention_heads=24,
        attention_head_dim=64,
        context_pre_only=context_pre_only,
        qk_norm="rms_norm",
        use_dual_attention=use_dual_attention,
        parameters=parameters,
    )

    for i in range(1):
        ttnn_input_temb = ttnn.from_torch(
            torch_input_temb,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_encoder_hidden_states = ttnn.from_torch(
            torch_input_encoder_hidden_states,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_hidden_states = ttnn.from_torch(
            torch_input_hidden_states,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # input_memory_config
        )

        t0 = time.time()
        ttnn_output_unopt = ttnn_model_unopt(
            ttnn_input_hidden_states, ttnn_input_encoder_hidden_states, ttnn_input_temb, parameters=parameters
        )
        # enable_persistent_kernel_cache()
        t1 = time.time()
        print("Time (sec):", t1 - t0)

    if context_pre_only != True:
        ttnn_output_0_unopt = ttnn.to_torch(ttnn_output_unopt[0])
        ttnn_output_1_unopt = ttnn.to_torch(ttnn_output_unopt[1])
        print(check_with_pcc(ttnn_output_0_unopt.unsqueeze(1), ttnn_output_0, pcc=0.99999999))
        print(check_with_pcc(ttnn_output_1_unopt.unsqueeze(1), ttnn_output_1, pcc=0.99999999))
    else:
        ttnn_output_1_unopt = ttnn.to_torch(ttnn_output_unopt[1])
        print(check_with_pcc(ttnn_output_1_unopt.unsqueeze(1), ttnn_output_1, pcc=0.99999999))
