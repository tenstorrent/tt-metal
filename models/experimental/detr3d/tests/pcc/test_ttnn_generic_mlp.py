# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
<<<<<<< HEAD

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.ttnn.generic_mlp import TtnnGenericMLP
from models.experimental.detr3d.reference.model_3detr import GenericMLP
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "input_dim, hidden_dims, output_dim, norm_fn_name, hidden_use_bias, output_use_bias, output_use_activation,"
    "output_use_norm, dropout, x_shape, weight_key_prefix",
    [
        (
            256,
            [256],
            256,
            "bn1d",
            False,
            False,
            True,
            True,
            None,
            (1, 256, 1024),
            "encoder_to_decoder_projection",
        ),
        (
            256,
            [256],
            256,
            None,
            True,
            True,
            True,
            False,
            None,
            (1, 256, 128),
            "query_projection",
        ),
        (
            256,
            [256, 256],
            11,
            "bn1d",
            False,
            True,
            False,
            False,
            0.0,
            (8, 256, 128),
            "mlp_heads.sem_cls_head",
        ),
        (
            256,
            [256, 256],
            3,
            "bn1d",
            False,
            True,
            False,
            False,
            0.0,
            (8, 256, 128),
            "mlp_heads.center_head",
        ),
        (
            256,
            [256, 256],
            3,
            "bn1d",
            False,
            True,
            False,
            False,
            0.0,
            (8, 256, 128),
            "mlp_heads.size_head",
        ),
        (
            256,
            [256, 256],
            12,
            "bn1d",
            False,
            True,
            False,
            False,
            0.0,
            (8, 256, 128),
            "mlp_heads.angle_cls_head",
        ),
        (
            256,
            [256, 256],
            12,
            "bn1d",
            False,
            True,
            False,
            False,
            0.0,
            (8, 256, 128),
            "mlp_heads.angle_residual_head",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
=======
from models.experimental.detr3d.ttnn.ttnn_generic_mlp import TttnnGenericMLP
from models.experimental.detr3d.reference.detr3d_model import GenericMLP
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, torch.nn.Conv1d):
        weight = model.weight
        if model.bias is not None:
            bias = model.bias
            if bias.dim() < 4:
                bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.float32)

    return parameters


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,norm_fn_name,activation,use_conv,"
    "hidden_use_bias,output_use_bias,output_use_activation,output_use_norm,"
    "weight_init_name,dropout,x_shape",
    [
        (256, [256], 256, "bn1d", "relu", True, False, False, True, True, None, None, (1, 256, 1024)),
        (256, [256], 256, None, "relu", True, True, True, True, False, None, None, (1, 256, 128)),
        (256, [256, 256], 12, "bn1d", "relu", True, False, True, False, False, None, None, (8, 256, 128)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
def test_ttnn_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
<<<<<<< HEAD
=======
    activation,
    use_conv,
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
<<<<<<< HEAD
    dropout,
    x_shape,
    weight_key_prefix,
    device,
):
    torch_model = GenericMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        norm_fn_name=norm_fn_name,
        activation="relu",
        use_conv=True,
        dropout=dropout,
        hidden_use_bias=hidden_use_bias,
        output_use_bias=output_use_bias,
        output_use_activation=output_use_activation,
        output_use_norm=output_use_norm,
        weight_init_name=None,
    )
    load_torch_model_state(torch_model, weight_key_prefix)

    x = torch.randn(x_shape)
    torch_out = torch_model(x)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ttnn_model = TtnnGenericMLP(
        parameters,
        device,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        output_use_activation=output_use_activation,
    )
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1),
=======
    weight_init_name,
    dropout,
    x_shape,
    device,
):
    torch_model = GenericMLP(
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name,
        activation,
        use_conv,
        dropout,
        hidden_use_bias,
        output_use_bias,
        output_use_activation,
        output_use_norm,
        weight_init_name,
    ).to(torch.bfloat16)
    torch_model.eval()
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    torch_out = torch_model(x)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    print("param are", parameters)
    ttnn_model = TttnnGenericMLP(torch_model, parameters, device)
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1).unsqueeze(dim=0),
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
<<<<<<< HEAD
    ttnn_out = ttnn_model(ttnn_x)
    ttnn_out = ttnn.to_torch(ttnn_out)
    ttnn_out = ttnn_out.permute(0, 2, 1)

    passing, pcc_message = comp_pcc(torch_out, ttnn_out, 0.999)
    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(torch_out, ttnn_out))
    logger.info(f"Input shape: {x_shape}, Weight prefix: {weight_key_prefix}")
    logger.info(f"MLP config - Input: {input_dim}, Hidden: {hidden_dims}, Output: {output_dim}")

    if passing:
        logger.info("GenericMLP Test Passed!")
    else:
        logger.warning("GenericMLP Test Failed!")

    assert passing, f"PCC value is lower than 0.999. Check implementation! {pcc_message}"
=======
    p(ttnn_x, "ttnn input")
    ttnn_out = ttnn_model(ttnn_x)
    print("outputs are", ttnn_out.shape, torch_out.shape)
    ttnn_out = ttnn.to_torch(ttnn_out)
    if ttnn_out.shape[-1] == 12:
        ttnn_out = ttnn_out.reshape(1, 8, 128, 12).permute(0, 1, 3, 2).squeeze(0)
    else:
        ttnn_out = ttnn_out.squeeze(dim=0).permute(0, 2, 1)
    assert_with_pcc(torch_out, ttnn_out, 1.0)
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
