# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.detr3d.reference.detr3d_model import (
    MaskedTransformerEncoder as ref_model,
    PointnetSAModuleVotes as ref_point_net_module_votes,
    TransformerEncoderLayer as ref_encoder_layer,
)

from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.detr3d.ttnn.transformer import TTTransformerEncoderLayer

from models.experimental.detr3d.ttnn.encoder import TtMaskedTransformerEncoder, EncoderArgs
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.ttnn.ttnn_pointnet_samodule_votes import TtnnPointnetSAModuleVotes


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
    "num_layers,masking_radius,norm, weight_init_name,src_shape,mask,src_key_padding_mask,pos,xyz_shape,transpose_swap",
    [
        (
            3,
            [0.16000000000000003, 0.6400000000000001, 1.44],
            None,
            "xavier_uniform",
            (2048, 1, 256),
            None,
            None,
            None,
            (1, 2048, 3),
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_masked_transformer_encoder(
    num_layers,
    masking_radius,
    norm,
    weight_init_name,
    src_shape,
    mask,
    src_key_padding_mask,
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
    encoder_layer = ref_encoder_layer(
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
    interim_downsampling = ref_point_net_module_votes(
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
    ref_module = ref_model(
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
        weight_init_name="xavier_uniform",
    )
    src = torch.randn(src_shape)
    xyz = torch.randn(xyz_shape)
    ref_module.eval()
    ref_out = ref_module(src, mask, src_key_padding_mask, pos, xyz, transpose_swap)

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
        bn=bn,
        use_xyz=use_xyz,
        pooling=pooling,
        sigma=sigma,
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
        encoder_args=EncoderArgs(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
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
    # tt_xyz = ttnn.from_torch(
    #     xyz,
    #     device=device,
    #     dtype=ttnn.bfloat16,
    # )

    tt_output = tt_module(tt_src, mask, src_key_padding_mask, pos, xyz, transpose_swap)
    ttnn_torch_out = []
    for tt_out, torch_out in zip(tt_output, ref_out):
        if not isinstance(tt_out, torch.Tensor):
            tt_out = ttnn.to_torch(tt_out)
            tt_out = torch.reshape(tt_out, torch_out.shape)
        ttnn_torch_out.append(tt_out)
        pcc_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)
        print(f"{pcc_message=}")
