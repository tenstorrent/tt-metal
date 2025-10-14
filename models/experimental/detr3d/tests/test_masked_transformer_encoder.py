# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import (
    MaskedTransformerEncoder as ref_model,
    PointnetSAModuleVotes as ref_point_net_module_votes,
    TransformerEncoderLayer as ref_encoder_layer,
)
from models.experimental.detr3d.source.detr3d.models.transformer import (
    MaskedTransformerEncoder as org_model,
    TransformerEncoderLayer as org_encoder_layer,
)
from models.experimental.detr3d.source.detr3d.third_party.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes as org_point_net_module_votes,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


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
):
    encoder_layer_1 = org_encoder_layer(
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
    ).to(torch.bfloat16)
    interim_downsampling_1 = org_point_net_module_votes(
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
    ).to(torch.bfloat16)
    org_module = org_model(
        encoder_layer_1,
        num_layers,
        masking_radius,
        interim_downsampling_1,
        norm=None,
        weight_init_name="xavier_uniform",
    ).to(torch.bfloat16)

    encoder_layer_2 = ref_encoder_layer(
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
    ).to(torch.bfloat16)
    interim_downsampling_2 = ref_point_net_module_votes(
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
    ).to(torch.bfloat16)
    ref_module = ref_model(
        encoder_layer_2,
        num_layers,
        masking_radius,
        interim_downsampling_2,
        norm=None,
        weight_init_name="xavier_uniform",
    ).to(torch.bfloat16)
    src = torch.randn(src_shape, dtype=torch.bfloat16)
    xyz = torch.randn(xyz_shape, dtype=torch.bfloat16)
    org_out = org_module(src, mask, src_key_padding_mask, pos, xyz, transpose_swap)
    ref_out = ref_module(src, mask, src_key_padding_mask, pos, xyz, transpose_swap)
    print("org out is", org_out[0].shape, org_out[1].shape, org_out[2].shape)
    print("org out is", org_out[0].shape, ref_out[1].shape, ref_out[2].shape)
    assert_with_pcc(org_out[0], ref_out[0], 1.0)
    assert_with_pcc(org_out[1], ref_out[1], 1.0)
    assert_with_pcc(org_out[2], ref_out[2], 1.0)
