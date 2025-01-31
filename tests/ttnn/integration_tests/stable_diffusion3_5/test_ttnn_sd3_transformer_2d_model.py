# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.sd3_transformer_2d_model import SD3Transformer2DModel
from models.experimental.functional_stable_diffusion3_5.reference.joint_transformer_block import (
    JointTransformerBlock,
    SD35AdaLayerNormZeroX,
    AdaLayerNormContinuous,
    AdaLayerNormZero,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_sd3_transformer_2d_model import (
    ttnn_SD3Transformer2DModel,
)

from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login, logout


def create_custom_preprocessor_transformer_block(device):
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


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, SD3Transformer2DModel):
            parameters["pos_embed"] = {}
            parameters["pos_embed"]["proj"] = {}
            parameters["pos_embed"]["proj"]["weight"] = ttnn.from_torch(
                model.pos_embed.proj.weight, dtype=ttnn.bfloat16
            )
            parameters["pos_embed"]["proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.pos_embed.proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )
            parameters["pos_embed"]["pos_embed"] = ttnn.from_torch(model.pos_embed.pos_embed, dtype=ttnn.bfloat16)

            # CombinedTimestepTextProjEmbeddings
            parameters["time_text_embed"] = {}
            # TimestepEmbedding
            parameters["time_text_embed"]["timestep_embedder"] = {}
            parameters["time_text_embed"]["timestep_embedder"]["linear_1"] = {}
            parameters["time_text_embed"]["timestep_embedder"]["linear_1"]["weight"] = preprocess_linear_weight(
                model.time_text_embed.timestep_embedder.linear_1.weight, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["timestep_embedder"]["linear_1"]["bias"] = preprocess_linear_bias(
                model.time_text_embed.timestep_embedder.linear_1.bias, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["timestep_embedder"]["linear_2"] = {}
            parameters["time_text_embed"]["timestep_embedder"]["linear_2"]["weight"] = preprocess_linear_weight(
                model.time_text_embed.timestep_embedder.linear_2.weight, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["timestep_embedder"]["linear_2"]["bias"] = preprocess_linear_bias(
                model.time_text_embed.timestep_embedder.linear_2.bias, dtype=ttnn.bfloat16
            )
            # PixArtAlphaTextProjection
            parameters["time_text_embed"]["text_embedder"] = {}
            parameters["time_text_embed"]["text_embedder"]["linear_1"] = {}
            parameters["time_text_embed"]["text_embedder"]["linear_1"]["weight"] = preprocess_linear_weight(
                model.time_text_embed.text_embedder.linear_1.weight, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["text_embedder"]["linear_1"]["bias"] = preprocess_linear_bias(
                model.time_text_embed.text_embedder.linear_1.bias, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["text_embedder"]["linear_2"] = {}
            parameters["time_text_embed"]["text_embedder"]["linear_2"]["weight"] = preprocess_linear_weight(
                model.time_text_embed.text_embedder.linear_2.weight, dtype=ttnn.bfloat16
            )
            parameters["time_text_embed"]["text_embedder"]["linear_2"]["bias"] = preprocess_linear_bias(
                model.time_text_embed.text_embedder.linear_2.bias, dtype=ttnn.bfloat16
            )

            parameters["context_embedder"] = {}
            parameters["context_embedder"]["weight"] = preprocess_linear_weight(
                model.context_embedder.weight, dtype=ttnn.bfloat16
            )
            parameters["context_embedder"]["bias"] = preprocess_linear_bias(
                model.context_embedder.bias, dtype=ttnn.bfloat16
            )

            # transformers
            parameters["transformer_blocks"] = {}
            for index, child in enumerate(model.transformer_blocks):
                join_transformer_preprocessor = create_custom_preprocessor_transformer_block(device)
                parameters["transformer_blocks"][index] = join_transformer_preprocessor(child, None, None)

            parameters["norm_out"] = {}
            parameters["norm_out"]["linear"] = {}
            parameters["norm_out"]["linear"]["weight"] = preprocess_linear_weight(
                model.norm_out.linear.weight, dtype=ttnn.bfloat8_b
            )
            parameters["norm_out"]["linear"]["bias"] = preprocess_linear_bias(
                model.norm_out.linear.bias, dtype=ttnn.bfloat8_b
            )

            parameters["proj_out"] = {}
            parameters["proj_out"]["weight"] = preprocess_linear_weight(model.proj_out.weight, dtype=ttnn.bfloat8_b)
            parameters["proj_out"]["bias"] = preprocess_linear_bias(model.proj_out.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_sd3_transformer_2d_model(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )

    config = pipe.transformer.config
    reference_model = SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qk_norm="rms_norm",
        config=config,
    )
    reference_model.load_state_dict(pipe.transformer.state_dict())

    reference_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    parameters["pos_embed"]["proj"]["weight"] = ttnn.from_device(parameters["pos_embed"]["proj"]["weight"])
    parameters["pos_embed"]["proj"]["bias"] = ttnn.from_device(parameters["pos_embed"]["proj"]["bias"])

    ttnn_model = ttnn_SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qk_norm="rms_norm",
        config=config,
        parameters=parameters,
    )

    # hidden_states = torch.load(
    #     "models/experimental/functional_stable_diffusion3_5/demo/inputs_512_latest/hidden_states_512.pt",
    #     map_location=torch.device("cpu"),
    # )
    # encoder_hidden_states = torch.load(
    #     "models/experimental/functional_stable_diffusion3_5/demo/inputs_512_latest/encoder_hidden_states_512_paddedto160.pt",
    #     map_location=torch.device("cpu"),
    # )  # the input is padded from 2,154,4096 to 2,160,4096 while passing to the torch model.
    # pooled_projections = torch.load(
    #     "models/experimental/functional_stable_diffusion3_5/demo/inputs_512_latest/pooled_projections_512.pt",
    #     map_location=torch.device("cpu"),
    # )
    # timestep = torch.load(
    #     "models/experimental/functional_stable_diffusion3_5/demo/inputs_512_latest/timestep_512.pt",
    #     map_location=torch.device("cpu"),
    # )
    import time

    for i in range(38):
        # i = 0
        # if (1):
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__hidden_states_"
            + str(i)
            + ".npy"
        )
        hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__encoder_hidden_"
            + str(i)
            + ".npy"
        )
        encoder_hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__pooled_proj_"
            + str(i)
            + ".npy"
        )
        pooled_projections = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512___timesteps_proj_"
            + str(i)
            + ".npy"
        )
        timesteps_proj = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

        hidden_states = hidden_states.permute(0, 2, 3, 1)  # NCHW to NHWC

        ttnn_hidden_states = ttnn.from_torch(
            hidden_states,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states.unsqueeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn_pooled_projections = ttnn.from_torch(
            pooled_projections.unsqueeze(0).unsqueeze(0),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn_timestep = ttnn.from_torch(
            timesteps_proj.unsqueeze(0).unsqueeze(0),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # ttnn_timestep = ttnn.from_torch(
        #     timestep, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        # )
        t0 = time.time()

        ttnn_output = ttnn_model(
            ttnn_hidden_states,
            ttnn_encoder_hidden_states,
            ttnn_pooled_projections,
            ttnn_timestep,
            None,
            None,
            None,
            parameters=parameters,
        )

        t1 = time.time()
        print("Time:", t1 - t0)

        torch_output = torch.from_numpy(
            np.load(
                "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__noise_pred_"
                + str(i)
                + ".npy"
            )
        )
        print(i, " --", assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output[0]), pcc=-100))  # 0.94
        # print(torch_output)
        # print(ttnn.to_torch(ttnn_output[0]))
