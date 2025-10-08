# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.gpt import GPT
from models.experimental.transfuser.tt.gpt import TTGpt

from models.experimental.transfuser.tests.test_gpt_block import create_gpt_block_preprocessor

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight
from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def generate_token_embeddings(image_tensor, lidar_tensor, seq_len, n_embd):
    print(f"{image_tensor.shape,lidar_tensor.shape=}")
    bz = lidar_tensor.shape[0]
    lidar_h, lidar_w = lidar_tensor.shape[2:4]
    img_h, img_w = image_tensor.shape[2:4]

    assert seq_len == 1
    image_tensor = (
        image_tensor.view(bz, seq_len, -1, img_h, img_w).permute(0, 1, 3, 4, 2).contiguous().view(bz, -1, n_embd)
    )
    lidar_tensor = (
        lidar_tensor.view(bz, seq_len, -1, lidar_h, lidar_w).permute(0, 1, 3, 4, 2).contiguous().view(bz, -1, n_embd)
    )

    token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

    return token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w


def post_process_output(
    x,
    bz,
    seq_len,
    img_vert_anchors,
    img_horz_anchors,
    lidar_vert_anchors,
    lidar_horz_anchors,
    n_embed,
    img_h,
    img_w,
    lidar_h,
    lidar_w,
):
    x = x.view(
        bz,
        seq_len * img_vert_anchors * img_horz_anchors + seq_len * lidar_vert_anchors * lidar_horz_anchors,
        n_embed,
    )

    image_tensor_out = (
        x[:, : seq_len * img_vert_anchors * img_horz_anchors, :].contiguous().view(bz * seq_len, -1, img_h, img_w)
    )
    lidar_tensor_out = (
        x[:, seq_len * img_vert_anchors * img_horz_anchors :, :].contiguous().view(bz * seq_len, -1, lidar_h, lidar_w)
    )
    return image_tensor_out, lidar_tensor_out


def create_gpt_preprocessor(device, n_layer, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if hasattr(torch_model, "pos_emb"):
            parameters["pos_emb"] = ttnn.from_torch(
                torch_model.pos_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        if hasattr(torch_model, "vel_emb"):
            parameters["vel_emb_weight"] = preprocess_linear_weight(torch_model.vel_emb.weight, dtype=weight_dtype)
            parameters["vel_emb_bias"] = preprocess_linear_weight(torch_model.vel_emb.bias, dtype=weight_dtype)
        if hasattr(torch_model, "ln_f"):
            parameters["ln_f_weight"] = preprocess_linear_weight(torch_model.ln_f.weight, dtype=weight_dtype)
            parameters["ln_f_bias"] = preprocess_linear_weight(torch_model.ln_f.bias, dtype=weight_dtype)
        if hasattr(torch_model, "blocks"):
            for i in range(n_layer):
                parameters[f"blocks_{i}"] = preprocess_model_parameters(
                    initialize_model=lambda: torch_model.blocks[i],
                    custom_preprocessor=create_gpt_block_preprocessor(device, weight_dtype),
                    device=device,
                )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, block_exp, n_layer, img_vert_anchors, img_horz_anchors, lidar_vert_anchors, lidar_horz_anchors, seq_len, embd_pdrop, attn_pdrop, resid_pdrop, use_velocity, img_input_shape, lidar_input_shape",
    ((72, 4, 4, 4, 5, 22, 8, 8, 1, 0.1, 0.1, 0.1, False, (1, 72, 5, 22), (1, 72, 8, 8)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_gpt(
    device,
    n_embed,
    n_head,
    block_exp,
    n_layer,
    img_vert_anchors,
    img_horz_anchors,
    lidar_vert_anchors,
    lidar_horz_anchors,
    seq_len,
    embd_pdrop,
    attn_pdrop,
    resid_pdrop,
    use_velocity,
    img_input_shape,
    lidar_input_shape,
    input_dtype,
    weight_dtype,
):
    image_input = torch.randn(img_input_shape)
    lidar_input = torch.randn(lidar_input_shape)
    velocity_input = torch.randn(1, 1)

    # setting machine to avoid loading files
    config = GlobalConfig(setting="eval")

    ref_layer = GPT(
        n_embd=n_embed,
        n_head=n_head,
        block_exp=block_exp,
        n_layer=n_layer,
        img_vert_anchors=img_vert_anchors,
        img_horz_anchors=img_horz_anchors,
        lidar_vert_anchors=lidar_vert_anchors,
        lidar_horz_anchors=lidar_horz_anchors,
        seq_len=seq_len,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        config=config,
        use_velocity=use_velocity,
    ).eval()

    ref_image_output, ref_lidar_output = ref_layer(image_input, lidar_input, velocity_input)

    token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w = generate_token_embeddings(
        image_input, lidar_input, seq_len, n_embed
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, weight_dtype),
        device=device,
    )
    tt_layer = TTGpt(
        device,
        parameters,
        n_head,
        n_layer,
        use_velocity=use_velocity,
        img_vert_anchors=img_vert_anchors,
        img_horz_anchors=img_horz_anchors,
        lidar_vert_anchors=lidar_vert_anchors,
        lidar_horz_anchors=lidar_horz_anchors,
        seq_len=seq_len,
        n_embd=n_embed,
        dtype=weight_dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # tt_token_embeddings = ttnn.from_torch(
    #     token_embeddings, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    # )
    tt_velocity_input = ttnn.from_torch(
        velocity_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output = tt_layer(token_embeddings, tt_velocity_input)
    tt_torch_output = tt2torch_tensor(tt_output)
    tt_image_output, tt_lidar_output = post_process_output(
        tt_torch_output,
        bz,
        seq_len,
        img_vert_anchors,
        img_horz_anchors,
        lidar_vert_anchors,
        lidar_horz_anchors,
        n_embed,
        img_h,
        img_w,
        lidar_h,
        lidar_w,
    )
    does_pass, image_out_pcc_message = check_with_pcc(ref_image_output, tt_image_output, 0.95)

    logger.info(f"Image Output PCC: {image_out_pcc_message}")
    assert does_pass, f"PCC check failed: {image_out_pcc_message}"

    does_pass, lidar_out_pcc_message = check_with_pcc(ref_lidar_output, tt_lidar_output, 0.95)
    assert does_pass, f"PCC check failed: {lidar_out_pcc_message}"

    logger.info(f"Lidar Output PCC: {lidar_out_pcc_message}")

    if does_pass:
        logger.info("GPT Passed!")
    else:
        logger.warning("GPT Failed!")
