# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from diffusers import AutoPipelineForText2Image

from models.demos.stable_diffusion_xl_turbo.tt.sd_cross_attention_downblock_2d import (
    sd_geglu,
    sd_feed_forward,
    sd_attention,
    sd_basic_transformer_block,
    sd_transformer_2d,
    ResnetBlock2D,
    sd_downsample_2,
    sd_cross_attention_down_blocks2d,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.stable_diffusion_xl_turbo.tt.utils import custom_preprocessor, custom_preprocessor_resnet
from models.demos.stable_diffusion_xl_turbo.tt.resnetblock2d_utils import update_params


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_geglu(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    geglu_model = model.down_blocks[1].attentions[0].transformer_blocks[0].ff.net[0]

    hidden_states = torch.randn((1, 4096, 640), dtype=torch.float16)

    torch_output = geglu_model(hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: geglu_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = sd_geglu(
        ttnn_input_tensor,
        parameters,
        device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99947)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_feed_forward(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    feed_forward_model = model.down_blocks[1].attentions[0].transformer_blocks[0].ff
    hidden_states = torch.randn((1, 4096, 640), dtype=torch.float16)

    torch_output = feed_forward_model(hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: feed_forward_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = sd_feed_forward(
        ttnn_input_tensor,
        parameters,
        device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.9994)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 4096, 640, False, 1),
        (1, 2, 4096, 640, True, 1),
        (1, 2, 1024, 1280, False, 2),
        (1, 2, 1024, 1280, True, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_down_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.down_blocks[index].attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.down_blocks[index].attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.9987)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 4096, 640, False, 1),
        (1, 2, 4096, 640, True, 1),
        (1, 2, 1024, 1280, False, 0),
        (1, 2, 1024, 1280, True, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_up_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.up_blocks[index].attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.up_blocks[index].attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "N, C, H, W, has_encoder_hidden_states, index",
    [
        (1, 2, 1024, 1280, False, 0),
        (1, 2, 1024, 1280, True, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_attention_mid_blocks(device, N, C, H, W, has_encoder_hidden_states, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()

    encoder_hidden_states = None
    ttnn_encoder_hidden_states = None
    attention_model = model.mid_block.attentions[0].transformer_blocks[0].attn1
    heads = attention_model.heads

    if has_encoder_hidden_states:
        attention_model = model.mid_block.attentions[0].transformer_blocks[0].attn2
        encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float16)
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    hidden_states = torch.randn((N, H, W), dtype=torch.float16)

    torch_output = attention_model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: attention_model, device=device)

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_attention(
        ttnn_input_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        heads=heads,
        parameters=parameters,
        device=device,
        query_dim=hidden_states.shape[-1] // 8,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "N, C, H, W, attention_head_dim, index",
    [
        (1, 2, 4096, 640, 40, 1),
        (1, 2, 1024, 1280, 40, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_basic_transformer_block(device, N, C, H, W, attention_head_dim, index, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    basic_transformer = model.down_blocks[index].attentions[0].transformer_blocks[0]

    hidden_states = torch.randn((N, H, W), dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = basic_transformer(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None

    parameters = preprocess_model_parameters(initialize_model=lambda: basic_transformer, device=device)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_input_tensor = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = sd_basic_transformer_block(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.996)


@pytest.mark.parametrize(
    "input_shape, index1, index2, attention_head_dim, block, transformer_layers_per_block",
    [
        ((1, 640, 64, 64), 1, 0, 10, "down", 2),
        ((1, 1280, 32, 32), 2, 0, 20, "down", 10),
        ((1, 640, 64, 64), 1, 0, 10, "up", 2),
        ((1, 1280, 32, 32), 0, 0, 20, "up", 10),
        ((1, 1280, 32, 32), 0, 0, 20, "mid", 10),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_transformer_2d_model(
    input_shape, index1, index2, block, attention_head_dim, transformer_layers_per_block, device, reset_seeds
):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config

    num_attention_heads = 8
    norm_num_groups = 32

    if block == "down":
        transformer_2d_model = model.down_blocks[index1].attentions[index2]
    elif block == "up":
        transformer_2d_model = model.up_blocks[index1].attentions[index2]
    else:
        transformer_2d_model = model.mid_block.attentions[index1]

    hidden_states = torch.randn(input_shape, dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = transformer_2d_model(hidden_states, encoder_hidden_states=encoder_hidden_states)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None
    return_dict = False

    parameters = preprocess_model_parameters(initialize_model=lambda: transformer_2d_model, device=device)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_output = sd_transformer_2d(
        hidden_states=ttnn_hidden_states,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        parameters=parameters,
        device=device,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        attention_head_dim=attention_head_dim,
        transformer_layers_per_block=transformer_layers_per_block,
        norm_num_groups=norm_num_groups,
        attention_mask=attention_mask,
        config=config,
        eps=1e-06,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.sample, ttnn_output, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1, index2, block_name",
    [
        (1, 320, 128, 128, 0, 0, "down"),
        (1, 320, 128, 128, 0, 1, "down"),
        (1, 320, 64, 64, 1, 0, "down"),  #  0.9790
        (1, 640, 64, 64, 1, 1, "down"),  #  0.982
        (1, 640, 32, 32, 2, 0, "down"),  #  0.95
        (1, 1280, 32, 32, 2, 1, "down"),
        (1, 1280, 32, 32, 0, 0, "mid"),  # 0.9858
        (1, 1280, 32, 32, 1, 0, "mid"),  # 0.9746
        (1, 2560, 32, 32, 0, 0, "up"),  #  0.9509
        (1, 2560, 32, 32, 0, 1, "up"),  #  0.9595
        (1, 1920, 32, 32, 0, 2, "up"),  #  0.7976363194085021
        (1, 1920, 64, 64, 1, 0, "up"),  #  0.9596
        (1, 1280, 64, 64, 1, 1, "up"),  #  0.9555
        (1, 960, 64, 64, 1, 2, "up"),  # 0.9764
        (1, 960, 128, 128, 2, 0, "up"),  #  OOM
        (1, 640, 128, 128, 2, 1, "up"),  #  0.93
        (1, 640, 128, 128, 2, 2, "up"),  # 0.9760
    ],
)
def test_resnet_block_2d_1024x1024(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    index1,
    index2,
    block_name,
    reset_seeds,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    parameters = update_params(parameters)
    if block_name == "up":
        parameters = parameters.up_blocks[index1].resnets[index2]
        resnet = model.up_blocks[index1].resnets[index2]
    elif block_name == "down":
        parameters = parameters.down_blocks[index1].resnets[index2]
        resnet = model.down_blocks[index1].resnets[index2]
    else:
        parameters = parameters.mid_block.resnets[index1]
        resnet = model.mid_block.resnets[index1]

    temb_channels = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, temb_channels]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    torch_output = resnet(input, temb)

    input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    resnet_block = ResnetBlock2D(
        config,
        input,
        temb,
        parameters,
        device,
    )
    resnet_block = ttnn.to_torch(resnet_block)
    assert_with_pcc(torch_output, resnet_block, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W",
    [
        (1, 640, 64, 64),
    ],
)
def test_downsample_2(device, N, C, H, W, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    model = model.down_blocks[1].downsamplers[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    input_tensor = torch.randn((N, C, H, W), dtype=torch.float32)
    torch_output = model(input_tensor)

    ttnn_hidden_state = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
    )
    output = sd_downsample_2(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.999572)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, add_downsample, index, transformer_layers_per_block , use_torch_conv",
    [
        (1, 320, 64, 64, True, 1, 2, True),  # 0.993296883554152 with torch resnet
        (1, 640, 32, 32, False, 2, 10, False),  # 0.9054295719095694 with ttnn resnet
    ],
)
def test_cross_attention_downblock2d(
    device, N, C, H, W, add_downsample, use_torch_conv, index, transformer_layers_per_block, reset_seeds
):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    cross_attention_down_blocks2d = model.down_blocks[index]
    print(cross_attention_down_blocks2d.num_attention_heads)
    config = model.config
    attention_head_dim = cross_attention_down_blocks2d.num_attention_heads
    num_layers = 2

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    parameters = update_params(parameters)
    parameters = parameters.down_blocks[index]
    conv_shortcut = True
    temb_channels = 1280

    hidden_states = torch.randn((N, C, H, W), dtype=torch.float32)
    temb = torch.randn((1, temb_channels), dtype=torch.float32)
    encoder_hidden_states = torch.randn((1, 77, 2048), dtype=torch.float32)

    torch_output = cross_attention_down_blocks2d(hidden_states, temb=temb, encoder_hidden_states=encoder_hidden_states)

    ttnn_hidden_states = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16)
    ttnn_temb = ttnn.from_torch(temb, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_output = sd_cross_attention_down_blocks2d(
        hidden_states=ttnn_hidden_states,
        temb=ttnn_temb,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        config=config,
        conv_shortcut=conv_shortcut,
        use_torch_conv=use_torch_conv,
        device=device,
        parameters=parameters,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        transformer_layers_per_block=transformer_layers_per_block,
        add_downsample=add_downsample,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output[0], ttnn_output, 0.90)
