# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.detr3d.reference.detr3d_model import (
    MaskedTransformerEncoder,
    PointnetSAModuleVotes,
    TransformerEncoderLayer,
)
from models.experimental.detr3d.ttnn.masked_transformer_encoder import (
    TTTransformerEncoderLayer,
    TtMaskedTransformerEncoder,
    EncoderLayerArgs,
)
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.common import load_torch_model_state


def compute_mask(device, xyz, radius, dist=None):
    with torch.no_grad():
        if dist is None or dist.shape[1] != xyz.shape[1]:
            dist = torch.cdist(xyz, xyz, p=2)
        # entries that are True in the mask do not contribute to self-attention
        # so points outside the radius are not considered
        mask = dist >= radius
    mask_ttnn = torch.zeros_like(mask, dtype=torch.float).masked_fill_(mask, float("-inf"))
    mask_ttnn = ttnn.from_torch(mask_ttnn, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return mask, mask_ttnn


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before, masking_radius, weight_key_prefix",
    [
        (1, 2048, 256, 4, True, 0.16000000000000003, "encoder.layers.0"),
        (1, 1024, 256, 4, True, 0.6400000000000001, "encoder.layers.1"),
        (1, 1024, 256, 4, True, 1.44, "encoder.layers.2"),
    ],
)
def test_transformer_encoder_layer_inference(
    batch_size,
    seq_len,
    d_model,
    nhead,
    normalize_before,
    masking_radius,
    weight_key_prefix,
    device,
):
    """Test TTTransformerEncoderLayer against PyTorch reference implementation"""

    torch.manual_seed(0)
    mesh_device = device
    dtype = ttnn.bfloat16

    # Initialize reference model
    reference_model = TransformerEncoderLayer(
        d_model,
        nhead,
        normalize_before=normalize_before,
    )
    load_torch_model_state(reference_model, weight_key_prefix)

    # Create test inputs
    src_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    xyz = torch.randn(batch_size, seq_len, 3, dtype=torch.float32)
    attn_mask, attn_mask_ttnn = compute_mask(mesh_device, xyz, masking_radius, None)
    # mask must be tiled to num_heads of the transformer
    bsz, n, n = attn_mask.shape
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat(1, nhead, 1, 1)
    attn_mask = attn_mask.view(bsz * nhead, n, n)
    attn_mask_ttnn = ttnn.unsqueeze(attn_mask_ttnn, 1)

    # Create positional embeddings
    pos = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)

    # Get reference output with explicit None masks
    with torch.no_grad():
        ref_output = reference_model(
            src_input,
            src_mask=attn_mask,
            pos=pos,
        )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=mesh_device,
    )

    # Initialize TTNN model with preprocessed parameters
    tt_model = TTTransformerEncoderLayer(
        mesh_device,
        d_model,
        nhead,
        normalize_before=normalize_before,
        parameters=parameters,  # Pass preprocessed weights
    )

    # Convert inputs to TTNN tensors
    tt_src = ttnn.from_torch(
        src_input.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Run TTNN model
    tt_output = tt_model(tt_src, src_mask=attn_mask_ttnn, pos=None, return_attn_weights=False)

    if isinstance(ref_output, tuple):
        ref_output = ref_output[0]  # Get the tensor, ignore attention weights

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = torch.permute(tt_output_torch, (1, 0, 2))

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.99)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}, Heads: {nhead}")
    logger.info(f"Normalize before: {normalize_before}")

    if passing:
        logger.info("TransformerEncoderLayer Test Passed!")
    else:
        logger.warning("TransformerEncoderLayer Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"

    ttnn.close_device(mesh_device)


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_name, use_ffn, ffn_use_bias",
    [
        (
            256,  # d_model
            4,  # nhead
            128,  # dim_feedforward
            0.0,  # dropout
            None,  # dropout_attn
            "relu",  # activation
            True,  # normalize_before
            "ln",  # norm_name
            True,  # use_ffn
            True,  # ffn_use_bias
        )
    ],
)
@pytest.mark.parametrize(
    "mlp, npoint, radius, nsample, bn, use_xyz, pooling, sigma, normalize_xyz, sample_uniformly, ret_unique_cnt",
    [
        (
            [256, 256, 256, 256],  # mlp
            1024,  # npoint
            0.4,  # radius
            32,  # nsample
            True,  # bn
            True,  # use_xyz
            "max",  # pooling
            None,  # sigma
            True,  # normalize_xyz
            False,  # sample_uniformly
            False,  # ret_unique_cnt
        )
    ],
)
@pytest.mark.parametrize(
    "num_layers,masking_radius,norm, weight_init_name,src_shape,mask,pos,xyz_shape,transpose_swap",
    [
        (
            3,
            [0.16000000000000003, 0.6400000000000001, 1.44],
            None,
            "xavier_uniform",
            (2048, 1, 256),
            None,
            None,
            (1, 2048, 3),
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_masked_transformer_encoder_inference(
    num_layers,
    masking_radius,
    norm,
    weight_init_name,
    src_shape,
    mask,
    pos,
    xyz_shape,
    transpose_swap,
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    dropout_attn,
    activation,
    normalize_before,
    norm_name,
    use_ffn,
    ffn_use_bias,
    mlp,
    npoint,
    radius,
    nsample,
    bn,
    use_xyz,
    pooling,
    sigma,
    normalize_xyz,
    sample_uniformly,
    ret_unique_cnt,
    device,
):
    torch.manual_seed(0)
    encoder_layer = TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        dropout_attn,
        activation,
        normalize_before,
        norm_name,
        use_ffn,
        ffn_use_bias,
    )
    interim_downsampling = PointnetSAModuleVotes(
        mlp=mlp[:],
        npoint=npoint,
        radius=radius,
        nsample=nsample,
        bn=bn,
        use_xyz=use_xyz,
        pooling=pooling,
        sigma=sigma,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
    )
    ref_module = MaskedTransformerEncoder(
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
    )
    load_torch_model_state(ref_module, "encoder")

    src = torch.randn(src_shape)
    xyz = torch.randn(xyz_shape)
    ref_out = ref_module(src, mask, None, pos, xyz, transpose_swap)

    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    tt_encoder_layer = TTTransformerEncoderLayer
    tt_interim_downsampling = TtnnPointnetSAModuleVotes(
        mlp=mlp[:],
        npoint=npoint,
        radius=radius,
        nsample=nsample,
        use_xyz=use_xyz,
        pooling=pooling,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
        module=interim_downsampling,
        parameters=ref_module_parameters.interim_downsampling.mlp_module,
        device=device,
    )
    tt_module = TtMaskedTransformerEncoder(
        tt_encoder_layer,
        num_layers,
        masking_radius,
        tt_interim_downsampling,
        encoder_args=EncoderLayerArgs(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            normalize_before=normalize_before,
            use_ffn=use_ffn,
        ),
        norm=None,
        parameters=ref_module_parameters,
        device=device,
    )

    tt_src = ttnn.from_torch(
        src,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_module(tt_src, mask, pos, xyz, transpose_swap)
    for tt_out, torch_out in zip(tt_output, ref_out):
        if not isinstance(tt_out, torch.Tensor):
            tt_out = ttnn.to_torch(tt_out)
            tt_out = torch.reshape(tt_out, torch_out.shape)
        assert_with_pcc(torch_out, tt_out, 0.99)
