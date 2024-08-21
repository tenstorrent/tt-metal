# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import math
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.tt import ttnn_optimized_vit_highres
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, patch_size, num_patches, height: int, width: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.

    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, 0]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = position_embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("image_size_h", [1024])
@pytest.mark.parametrize("image_size_w", [1024])
@pytest.mark.parametrize("image_channels", [3])
def test_vit_patch_embeddings(device, model_name, image_size_h, image_size_w, image_channels):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    torch_pixel_values = torch_random((1, image_channels, image_size_h, image_size_w), -1, 1, dtype=torch.float32)
    torch_output, *_ = model.vit.embeddings.patch_embeddings(torch_pixel_values, interpolate_pos_encoding=True)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
    )

    # pixel_values = ttnn.from_torch(torch_pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    patch_size = 16
    pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    print(pixel_values.shape)
    # pixel_values = pixel_values.reshape(batch_size, image_size, image_size // patch_size, 4 * patch_size) # run it on device
    pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_patch_embeddings(
        config,
        pixel_values,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0], 0.9999)


# padding
# 1024 / 16 = 64
# 64*64 + 32 = 4128 (from cls_token concat)
# 4352 = (4128 + 224)
# 4352 / 8 = 136


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("image_size_h", [1024])
@pytest.mark.parametrize("image_size_w", [1024])
def test_vit_embeddings(device, model_name, image_size_h, image_size_w):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0].resize((image_size_h, image_size_w))
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt", do_resize=False, do_center_crop=False).pixel_values
    torch_output, *_ = model.vit.embeddings(torch_pixel_values, interpolate_pos_encoding=True)

    # High resolution patch_parameters interpolation
    model_state_dict = model.state_dict()
    init_position_embeddings = torch.nn.Parameter(model_state_dict["vit.embeddings.position_embeddings"])
    patch_size = 16
    tot_patch_count = (image_size_h // patch_size) * (image_size_w // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size_h, image_size_w)
    )
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
    )

    # pixel_values = ttnn.from_torch(torch_pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    patch_size = 16
    pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    print(pixel_values.shape)
    # pixel_values = pixel_values.reshape(batch_size, image_size, image_size // patch_size, 4 * patch_size) # run it on device
    pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_embeddings(
        config,
        pixel_values,
        position_embeddings,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)
    print(output.shape)

    # pad_size = (0, 0, 0, 255)  # 224 + 31
    # torch_output = torch.nn.functional.pad(torch_output, pad_size, "constant", 0)

    assert_with_pcc(torch_output, output[0], 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [640])
def test_vit_attention_experimental(device, model_name, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    torch_hidden_states = torch_random((1, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, 1, 1, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_attention_experimental(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [640])
def test_vit_attention(device, model_name, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    torch_hidden_states = torch_random((1, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, 1, 1, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [4032])
def test_vit_intermediate(device, model_name, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

    torch_hidden_states = torch_random((1, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_intermediate(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [4032])
def test_vit_output(device, model_name, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTOutput(config).eval()

    torch_intermediate = torch_random((1, sequence_size, config.intermediate_size), -1, 1, dtype=torch.float32)
    torch_residual = torch_random((1, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    intermediate = ttnn.from_torch(torch_intermediate, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_vit_highres.vit_output(
        config,
        intermediate,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [4032])
def test_vit_layer(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").vit.encoder.layer[0]
    # print(model)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(batch_size, 1, 1, sequence_size, dtype=torch.float32)

    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_optimized_vit_highres.vit_layer(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("sequence_size", [640])  #
def test_vit_encoder(device, model_name, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    ).vit.encoder

    torch_hidden_states = torch_random((1, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output = model(torch_hidden_states, torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(1, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    output = ttnn_optimized_vit_highres.vit_encoder(
        config,
        hidden_states,
        head_masks,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


# padding
# 1024 / 16 = 64
# 64*64 + 32 = 4128 (from cls_token concat)
# 4352 = (4128 + 224)
# 4352 / 8 = 136


@pytest.mark.skip(reason="#7527: Test and PCC threshold needs review")
@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("image_size_h", [1024])
@pytest.mark.parametrize("image_size_w", [1024])
@pytest.mark.parametrize("sequence_size", [4032])
def test_vit(device, model_name, image_size_h, image_size_w, sequence_size):
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.num_hidden_layers = 12
    model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0].resize((image_size_h, image_size_w))
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt", do_resize=False, do_center_crop=False).pixel_values
    torch_attention_mask = torch.ones(config.num_hidden_layers, sequence_size, dtype=torch.float32)
    torch_output, *_ = model(torch_pixel_values, interpolate_pos_encoding=True).logits

    # High resolution patch_parameters interpolation
    model_state_dict = model.state_dict()
    init_position_embeddings = torch.nn.Parameter(model_state_dict["vit.embeddings.position_embeddings"])
    patch_size = 16
    tot_patch_count = (image_size_h // patch_size) * (image_size_w // patch_size)
    torch_position_embeddings = torch.nn.Parameter(
        interpolate_pos_encoding(init_position_embeddings, patch_size, tot_patch_count, image_size, image_size)
    )
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_vit_highres.custom_preprocessor,
    )

    torch_pixel_values = torch_pixel_values.to(torch.bfloat16)
    pixel_values = ttnn.from_torch(torch_pixel_values, layout=ttnn.TILE_LAYOUT, device=device)

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    output = ttnn_optimized_vit_highres.vit(
        config,
        pixel_values,
        head_masks,
        position_embeddings,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output[0][0], 0.9999)
