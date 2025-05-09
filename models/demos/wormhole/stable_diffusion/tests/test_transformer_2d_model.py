# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import TRANSFORMER_PARAMETERIZATIONS
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_transformer_2d_new_conv import transformer_2d_model
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    post_process_output_and_move_to_host,
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shape, shard_layout, shard_end_core, shard_shape, attention_head_dim, block, block_index, attention_index",
    TRANSFORMER_PARAMETERIZATIONS,
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_transformer_2d_model_512x512(
    device,
    model_name,
    input_shape,
    shard_layout,
    shard_end_core,
    shard_shape,
    attention_head_dim,
    block,
    block_index,
    attention_index,
    reset_seeds,
    use_program_cache,
):
    torch.manual_seed(0)
    encoder_hidden_states = [1, 2, 77, 768]
    timestep = (None,)
    class_labels = (None,)
    cross_attention_kwargs = (None,)
    return_dict = True

    num_layers = 1
    num_attention_heads = 8
    norm_num_groups = 32
    norm_type = "layer_norm"
    cross_attention_dim = 768
    upcast_attention = False

    _, in_channels, _, _ = input_shape

    input = torch.randn(input_shape) * 0.01
    encoder_hidden_states = torch.rand(encoder_hidden_states)

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config

    parameters = preprocess_model_parameters(
        model_name=model_name, initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )

    if block == "up":
        parameters = parameters.up_blocks[block_index].attentions[attention_index]
        transformer = pipe.unet.up_blocks[block_index].attentions[attention_index]
    elif block == "down":
        parameters = parameters.down_blocks[block_index].attentions[attention_index]
        transformer = pipe.unet.down_blocks[block_index].attentions[attention_index]
    elif block == "mid":
        parameters = parameters.mid_block.attentions[0]
        transformer = pipe.unet.mid_block.attentions[0]

    torch_output = transformer(input, encoder_hidden_states.squeeze(0)).sample

    ttnn_hidden_state = ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    model = transformer_2d_model(
        device, parameters, input_shape[0], input_shape[2], input_shape[3], compute_kernel_config
    )

    ttnn_hidden_state = preprocess_and_push_input_to_device(
        device,
        input,
        memory_config=ttnn.MemoryConfig(
            shard_layout,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(shard_end_core[0], shard_end_core[1]),
                        ),
                    }
                ),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    output = model(
        hidden_states=ttnn_hidden_state,
        config=config,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        in_channels=in_channels,
        out_channels=in_channels,
        num_layers=num_layers,
        norm_num_groups=norm_num_groups,
        norm_type=norm_type,
        cross_attention_dim=cross_attention_dim,
        upcast_attention=upcast_attention,
    )

    ttnn_output_torch = post_process_output_and_move_to_host(
        output,
        model.batch_size,
        model.input_height,
        model.input_width,
        model.proj_out_out_channels,
    )

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
