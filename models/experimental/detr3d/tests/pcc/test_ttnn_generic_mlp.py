# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.detr3d.ttnn.generic_mlp import TtnnGenericMLP
from models.experimental.detr3d.reference.model_3detr import GenericMLP
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.common import load_torch_model_state


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
def test_ttnn_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
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
    ttnn_model = TtnnGenericMLP(torch_model, parameters, device)
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
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
