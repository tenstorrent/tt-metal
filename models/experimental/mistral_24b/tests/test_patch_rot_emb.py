from loguru import logger

import torch
import pytest
import os
import ttnn

# models/tt_transformers/tt/common.py
from models.experimental.mistral_24b.tt.vision_rope import VisionRotarySetup as RotarySetup

from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.tt_transformers.tt.model_config import ModelArgs


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("device"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_rot_emb(seq_len, batch_size, use_program_cache, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()
    partial_state_dict = {}

    reference_model = tt_model_args.reference_vision_rot_emb()
    reference_model.load_state_dict(partial_state_dict)

    ##### Create input tensor for the all gather #####
    B, NCH, H, W = (1, 3, tt_model_args.vision_chunk_size, tt_model_args.vision_chunk_size)
    in_channels, out_channels, kernel_size, stride, bias = (
        3,
        tt_model_args.vision_dim,
        tt_model_args.vision_patch_size,
        tt_model_args.vision_patch_size,
        False,
    )

    patch_size = tt_model_args.vision_patch_size
    image_size = tt_model_args.vision_image_size
    dim = tt_model_args.vision_dim
    num_patches_per_dim = image_size // patch_size
    num_patches = num_patches_per_dim * num_patches_per_dim

    input_val = torch.randn(batch_size, num_patches, dim)
    ##### Prepare inputs #####
    input_tensor = torch.randn((B, NCH, H, W))
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    first_layer_prefix = "vision_tower.patch_conv."
    conv_partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    conv_reference_model = tt_model_args.reference_conv2d_patch()
    conv_reference_model.load_state_dict(conv_partial_state_dict)
    patch_embeds = conv_reference_model(input_tensor)

    image_sizes = [(224, 224)]
    patch_embeds_list = [
        embed[..., : (size[0] // patch_size), : (size[1] // patch_size)]
        for embed, size in zip(patch_embeds, image_sizes)
    ]

    print("patch_embeds_list:", patch_embeds_list)

    position_ids = position_ids_in_meshgrid(
        patch_embeds_list, max_width=tt_model_args.vision_image_size // tt_model_args.vision_patch_size
    )
    print("position_ids:", position_ids.shape)

    reference_output = reference_model(input_val, position_ids)[0]
    print("ref output:", reference_output.shape)
    print("ref output:", reference_output)

    tt_model = RotarySetup(
        device,
        batch_size=batch_size,
        head_dim=tt_model_args.vision_dim,
        image_size=tt_model_args.vision_image_size,
        patch_size=tt_model_args.vision_patch_size,
        max_seq_len=tt_model_args.max_seq_len,
        rope_theta=tt_model_args.vision_rope_theta,
        scale_factor=tt_model_args.vision_image_size // tt_model_args.vision_patch_size,
        orig_context_len=tt_model_args.max_seq_len,
        datatype=dtype,
    )

    tt_output = tt_model.get_rot_mats(position_ids)[0]
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    print("tt output:", tt_output)
    print("tt output:", tt_output.shape)

    passing, pcc_message = comp_pcc(reference_output, tt_output)

    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
