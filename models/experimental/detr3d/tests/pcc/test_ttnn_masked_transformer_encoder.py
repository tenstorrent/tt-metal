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

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.detr3d.ttnn.transformer import TTTransformerEncoderLayer

# from models.experimental.detr3d.tests.pcc.test_ttnn_shared_mlp import (
#     custom_preprocessor_whole_model as custom_preprocessor_shared_mlp,
# )
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.encoder import TtMaskedTransformerEncoder, EncoderArgs
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


# def custom_preprocessor(model, name, ttnn_module_args):
#     parameters = {}
#     weight_dtype=ttnn.bfloat16
#     if isinstance(model, torch.nn.MultiheadAttention):
#         if hasattr(model, "self_attn"):
#             # Preprocess self-attention parameters
#             parameters["self_attn"] = {}

#             # Handle QKV weights for self-attention
#             if hasattr(model.self_attn, "in_proj_weight"):
#                 # Split combined QKV weight into separate Q, K, V
#                 qkv_weight = model.self_attn.in_proj_weight
#                 d_model = qkv_weight.shape[1]
#                 q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
#                 qkv_bias = model.self_attn.in_proj_bias
#                 q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

#                 parameters["self_attn"]["q_weight"] = ttnn.from_torch(
#                     q_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["k_weight"] = ttnn.from_torch(
#                     k_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["v_weight"] = ttnn.from_torch(
#                     v_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["q_bias"] = ttnn.from_torch(
#                     q_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["k_bias"] = ttnn.from_torch(
#                     k_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["v_bias"] = ttnn.from_torch(
#                     v_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )

#             if hasattr(model.self_attn, "out_proj"):
#                 parameters["self_attn"]["out_weight"] = ttnn.from_torch(
#                     model.self_attn.out_proj.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#                 )
#                 parameters["self_attn"]["out_bias"] = None
#                 if model.self_attn.out_proj.bias is not None:
#                     parameters["self_attn"]["out_bias"] = ttnn.from_torch(
#                         model.self_attn.out_proj.bias.reshape(1, -1),
#                         dtype=weight_dtype,
#                         layout=ttnn.TILE_LAYOUT,
#                         device=device,
#                     )

#     # if hasattr(model, "multihead_attn"):
#     #     # Preprocess cross-attention parameters
#     #     parameters["multihead_attn"] = {}

#     #     if hasattr(model.multihead_attn, "in_proj_weight"):
#     #         qkv_weight = model.multihead_attn.in_proj_weight
#     #         q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

#     #         parameters["multihead_attn"]["q_weight"] = ttnn.from_torch(
#     #             q_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )
#     #         parameters["multihead_attn"]["k_weight"] = ttnn.from_torch(
#     #             k_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )
#     #         parameters["multihead_attn"]["v_weight"] = ttnn.from_torch(
#     #             v_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )

#     #     if hasattr(model.multihead_attn, "out_proj"):
#     #         parameters["multihead_attn"]["out_weight"] = ttnn.from_torch(
#     #             model.multihead_attn.out_proj.weight.T,
#     #             dtype=weight_dtype,
#     #             layout=ttnn.TILE_LAYOUT,
#     #             device=device,
#     #         )

#     # # Preprocess layer normalization parameters
#     # for norm_name in ["norm1", "norm2", "norm3"]:
#     #     if hasattr(model, norm_name):
#     #         norm_layer = getattr(model, norm_name)
#     #         parameters[norm_name] = {}
#     #         parameters[norm_name]["weight"] = ttnn.from_torch(
#     #             norm_layer.weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )
#     #         if hasattr(norm_layer, "bias") and norm_layer.bias is not None:
#     #             parameters[norm_name]["bias"] = ttnn.from_torch(
#     #                 norm_layer.bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #             )

#     # # Preprocess feedforward parameters
#     # if hasattr(model, "linear1"):
#     #     parameters["linear1"] = {}
#     #     parameters["linear1"]["weight"] = ttnn.from_torch(
#     #         model.linear1.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #     )
#     #     if model.linear1.bias is not None:
#     #         parameters["linear1"]["bias"] = ttnn.from_torch(
#     #             model.linear1.bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )

#     # if hasattr(model, "linear2"):
#     #     parameters["linear2"] = {}
#     #     parameters["linear2"]["weight"] = ttnn.from_torch(
#     #         model.linear2.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #     )
#     #     if model.linear2.bias is not None:
#     #         parameters["linear2"]["bias"] = ttnn.from_torch(
#     #             model.linear2.bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
#     #         )

#     if isinstance(model, SharedMLP):
#         print("model.layer0.conv: ", model.layer0.conv)
#         print("model.layer0.bn: ", model.layer0.bn.bn)
#         weight, bias = fold_batch_norm2d_into_conv2d(model.layer0.conv, model.layer0.bn.bn)
#         parameters["layer0"] = {}
#         parameters["layer0"]["conv"] = {}
#         parameters["layer0"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
#         bias = bias.reshape((1, 1, 1, -1))
#         parameters["layer0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

#         weight, bias = fold_batch_norm2d_into_conv2d(model.layer1.conv, model.layer1.bn.bn)
#         parameters["layer1"] = {}
#         parameters["layer1"]["conv"] = {}
#         parameters["layer1"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
#         bias = bias.reshape((1, 1, 1, -1))
#         parameters["layer1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

#         weight, bias = fold_batch_norm2d_into_conv2d(model.layer2.conv, model.layer2.bn.bn)
#         parameters["layer2"] = {}
#         parameters["layer2"]["conv"] = {}
#         parameters["layer2"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
#         bias = bias.reshape((1, 1, 1, -1))
#         parameters["layer2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

#     return parameters


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
            # [0.16000000000000003],
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
        # False,
        interim_downsampling,
        norm=None,
        weight_init_name="xavier_uniform",
    )
    src = torch.randn(src_shape)
    xyz = torch.randn(xyz_shape)
    ref_module.eval()
    ref_out = ref_module(src, mask, src_key_padding_mask, pos, xyz, transpose_swap)

    print(f"{ref_out=}")

    # encoder_layer_parameters = preprocess_model_parameters(
    #     initialize_model=lambda: encoder_layer,
    #     custom_preprocessor=custom_preprocessor_shared_mlp,
    #     device=device,
    # )
    # interim_downsampling_parameters = preprocess_model_parameters(
    #     initialize_model=lambda: interim_downsampling.mlp_module,
    #     custom_preprocessor=custom_preprocessor_shared_mlp,
    #     device=device,
    # )
    ref_module_parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_module,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    tt_encoder_layer = TTTransformerEncoderLayer
    # tt_interim_downsampling = TtnnPointnetSAModuleVotes(
    #     mlp=mlp[:],
    #     npoint=npoint,
    #     radius=radius,
    #     nsample=nsample,
    #     bn=bn,
    #     use_xyz=use_xyz,
    #     pooling=pooling,
    #     sigma=sigma,
    #     normalize_xyz=normalize_xyz,
    #     sample_uniformly=sample_uniformly,
    #     ret_unique_cnt=ret_unique_cnt,
    #     module=interim_downsampling,
    #     parameters=ref_module_parameters.interim_downsampling.mlp_module,
    #     device=device,
    # )
    tt_interim_downsampling = ref_point_net_module_votes(
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
    tt_module = TtMaskedTransformerEncoder(
        tt_encoder_layer,
        num_layers,
        masking_radius,
        # None,
        tt_interim_downsampling,
        encoder_args=EncoderArgs(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            normalize_before,
            use_ffn,
            {},
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
    tt_xyz = ttnn.from_torch(
        xyz,
        device=device,
        dtype=ttnn.bfloat16,
    )

    tt_output = tt_module(tt_src, mask, src_key_padding_mask, pos, tt_xyz, transpose_swap)
    ttnn_torch_out = []
    for tt_out, torch_out in zip(tt_output, ref_out):
        print(
            f"//////////////////////////////////Starting the output torch convert //////////////////////////////////////"
        )
        print(f"{tt_out=}")
        print(f"layout={tt_out.layout}")
        print(f"tensor_on_device={ttnn.is_tensor_storage_on_device(tt_out)}")
        tt_out_ = ttnn.to_torch(tt_out)
        print(
            f"//////////////////////////////////Finished the output torch convert //////////////////////////////////////"
        )
        ttnn_torch_out.append(tt_out_)
        print(
            f"//////////////////////////////////Starting the output torch reshape //////////////////////////////////////"
        )
        ttnn_torch_out[-1] = torch.reshape(ttnn_torch_out[-1], torch_out.shape)
        print(
            f"//////////////////////////////////Finished the output torch reshape //////////////////////////////////////"
        )

    import pdb

    pdb.set_trace()

    pcc_pass, pcc_message = assert_with_pcc(ref_out[0], ttnn_torch_out[0], 0.1)
    print(f"{pcc_message=}")
    pcc_pass, pcc_message = assert_with_pcc(ref_out[1], ttnn_torch_out[1], 0.1)
    print(f"{pcc_message=}")
    # pcc_pass, pcc_message = assert_with_pcc(ref_out[2], ttnn_torch_out[2], 0.1)
    # print(f"{pcc_message=}")
