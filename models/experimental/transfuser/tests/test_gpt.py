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
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def generate_token_embeddings(image_tensor, lidar_tensor, seq_len, n_embd):
    print(f"{image_tensor.shape,lidar_tensor.shape, seq_len, n_embd =}")
    """
    Generate token embeddings from NCHW format tensors.

    Args:
        image_tensor: (batch, channels, height, width) - e.g., (1, 72, 5, 22)
        lidar_tensor: (batch, channels, height, width) - e.g., (1, 72, 8, 8)
        seq_len: sequence length (should be 1)
        n_embd: embedding dimension (should be 72)

    Returns:
        token_embeddings: (batch, total_tokens, n_embd)
        Additional metadata for post-processing
    """
    print(f"{image_tensor.shape, lidar_tensor.shape, seq_len, n_embd =}")

    bz = image_tensor.shape[0]
    img_c = image_tensor.shape[1]  # Should be 72
    img_h, img_w = image_tensor.shape[2:4]  # 5, 22

    lidar_c = lidar_tensor.shape[1]  # Should be 72
    lidar_h, lidar_w = lidar_tensor.shape[2:4]  # 8, 8

    assert seq_len == 1, f"seq_len must be 1, got {seq_len}"
    assert img_c == n_embd, f"Image channels {img_c} must match n_embd {n_embd}"
    assert lidar_c == n_embd, f"LiDAR channels {lidar_c} must match n_embd {n_embd}"

    # Reshape from NCHW to token sequence format
    # (batch, channels, height, width) -> (batch, height*width, channels)
    image_tokens = image_tensor.permute(0, 2, 3, 1).contiguous()  # (1, 5, 22, 72)
    image_tokens = image_tokens.view(bz, img_h * img_w, n_embd)  # (1, 110, 72)

    lidar_tokens = lidar_tensor.permute(0, 2, 3, 1).contiguous()  # (1, 8, 8, 72)
    lidar_tokens = lidar_tokens.view(bz, lidar_h * lidar_w, n_embd)  # (1, 64, 72)

    # Concatenate image and lidar tokens along sequence dimension
    token_embeddings = torch.cat((image_tokens, lidar_tokens), dim=1)  # (1, 174, 72)

    print(f"Generated token_embeddings shape: {token_embeddings.shape}")
    print(f"  Image tokens: {img_h * img_w}, LiDAR tokens: {lidar_h * lidar_w}")

    return token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w


def generate_token_embeddings_tt(image_tensor, lidar_tensor, seq_len, n_embd):
    """
    Generate token embeddings from NCHW format tensors.

    Args:
        image_tensor: (batch, channels, height, width) - e.g., (1, 72, 5, 22)
        lidar_tensor: (batch, channels, height, width) - e.g., (1, 72, 8, 8)
        seq_len: sequence length (should be 1)
        n_embd: embedding dimension (should be 72)

    Returns:
        token_embeddings: (batch, total_tokens, n_embd)
        Additional metadata for post-processing
    """
    bz = image_tensor.shape[0]
    img_c = image_tensor.shape[1]  # Should be 72
    img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]  # 5, 22

    lidar_c = lidar_tensor.shape[1]  # Should be 72
    lidar_h, lidar_w = lidar_tensor.shape[2], lidar_tensor.shape[3]  # 8, 8

    # Permute from NCHW to NHWC format
    # (batch, channels, height, width) -> (batch, height, width, channels)
    image_tokens = ttnn.permute(image_tensor, (0, 2, 3, 1))  # (1, 5, 22, 72)
    image_tokens = ttnn.reshape(image_tokens, (bz, img_h * img_w, n_embd))  # (1, 110, 72)

    lidar_tokens = ttnn.permute(lidar_tensor, (0, 2, 3, 1))  # (1, 8, 8, 72)
    lidar_tokens = ttnn.reshape(lidar_tokens, (bz, lidar_h * lidar_w, n_embd))  # (1, 64, 72)

    # Concatenate image and lidar tokens along sequence dimension
    token_embeddings = ttnn.concat([image_tokens, lidar_tokens], dim=1)  # (1, 174, 72)

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


def post_process_output_tt(
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
    # Reshape to [bz, total_seq, n_embed]
    total_seq = seq_len * img_vert_anchors * img_horz_anchors + seq_len * lidar_vert_anchors * lidar_horz_anchors
    x = ttnn.reshape(x, (bz, total_seq, n_embed))

    # Split image and lidar tensors
    img_seq_len = seq_len * img_vert_anchors * img_horz_anchors

    # Slice image tensor
    image_tensor = x[:, :img_seq_len, :]
    image_tensor_out = ttnn.reshape(image_tensor, (bz * seq_len, -1, img_h, img_w))

    # Slice lidar tensor
    lidar_tensor = x[:, img_seq_len:, :]
    lidar_tensor_out = ttnn.reshape(lidar_tensor, (bz * seq_len, -1, lidar_h, lidar_w))

    return image_tensor_out, lidar_tensor_out


def create_gpt_preprocessor(device, n_layer, weight_dtype=ttnn.bfloat16, use_optimized_self_attn=False):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if hasattr(torch_model, "pos_emb"):
            parameters["pos_emb"] = ttnn.from_torch(
                torch_model.pos_emb, dtype=weight_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
        # Layer norm parameters
        if hasattr(torch_model, "ln_f"):
            parameters["ln_f_weight"] = preprocess_linear_weight(torch_model.ln_f.weight, dtype=weight_dtype)
            parameters["ln_f_bias"] = preprocess_linear_weight(torch_model.ln_f.bias, dtype=weight_dtype)

        # Velocity embedding parameters (if exists)
        if hasattr(torch_model, "vel_emb"):
            parameters["vel_emb_weight"] = preprocess_linear_weight(torch_model.vel_emb.weight, dtype=weight_dtype)
            parameters["vel_emb_bias"] = preprocess_linear_weight(torch_model.vel_emb.bias, dtype=weight_dtype)

        # Transformer blocks
        if hasattr(torch_model, "blocks"):
            for i in range(n_layer):
                parameters[f"blocks_{i}"] = preprocess_model_parameters(
                    initialize_model=lambda i=i: torch_model.blocks[i],  # Capture i in closure
                    custom_preprocessor=create_gpt_block_preprocessor(device, weight_dtype, use_optimized_self_attn),
                    device=device,
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n_embed, n_head, block_exp, n_layer, img_vert_anchors, img_horz_anchors, lidar_vert_anchors, lidar_horz_anchors, seq_len, embd_pdrop, attn_pdrop, resid_pdrop, use_velocity, img_input_shape, lidar_input_shape",
    # ((72, 4, 4, 4, 5, 22, 8, 8, 1, 0.1, 0.1, 0.1, False, (1, 72, 5, 22), (1, 72, 8, 8)),),  # GPT-SelfAttention 1
    ((216, 4, 4, 4, 5, 22, 8, 8, 1, 0.1, 0.1, 0.1, False, (1, 216, 5, 22), (1, 216, 8, 8)),),  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_optimized_self_attn", [False, True])
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
    use_optimized_self_attn,
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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_gpt_preprocessor(device, n_layer, weight_dtype, use_optimized_self_attn),
        device=device,
    )

    # High accuracy compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
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
        compute_kernel_config=compute_kernel_config,
    )

    img_h, img_w = img_input_shape[2], img_input_shape[3]
    image_input_tokens = image_input.permute(0, 2, 3, 1).reshape(1, 1, img_h * img_w, n_embed)

    # (1, 72, 8, 8) -> (1, 1, 64, 72)
    lidar_h, lidar_w = lidar_input_shape[2], lidar_input_shape[3]
    lidar_input_tokens = lidar_input.permute(0, 2, 3, 1).reshape(1, 1, lidar_h * lidar_w, n_embed)

    tt_image_input = ttnn.from_torch(
        image_input_tokens,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_lidar_input = ttnn.from_torch(
        lidar_input_tokens,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_velocity_input = ttnn.from_torch(
        velocity_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    tt_image_output, tt_lidar_output = tt_layer(tt_image_input, tt_lidar_input, tt_velocity_input, n_embed)

    tt_image_output = tt2torch_tensor(tt_image_output)
    tt_lidar_output = tt2torch_tensor(tt_lidar_output)
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
