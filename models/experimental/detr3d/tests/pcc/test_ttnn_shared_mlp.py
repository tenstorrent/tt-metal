# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.detr3d.ttnn.shared_mlp import TtnnSharedMLP
from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.reference.pointnet2_modules import SharedMLP
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "mlp, bn, features_shape, weight_key_prefix",
    [
        ([3, 64, 128, 256], True, (1, 3, 2048, 64), "pre_encoder.mlp_module"),  # mlp  # bn  # weight prefix
        (
            [259, 256, 256, 256],
            True,
            (1, 259, 1024, 32),
            "encoder.interim_downsampling.mlp_module",
        ),  # mlp  # bn  # weight prefix
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_shared_mlp(device, mlp, bn, features_shape, weight_key_prefix, reset_seeds):
    torch_model = SharedMLP(mlp, bn=bn).to(torch.bfloat16)
    load_torch_model_state(torch_model, weight_key_prefix)

    features = torch.randn(features_shape, dtype=torch.bfloat16)
    ref_out = torch_model(features)

    ttnn_features = ttnn.from_torch(
        features.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ttnn_model = TtnnSharedMLP(parameters, device)
    ttnn_out = ttnn_model(ttnn_features)

    ttnn_out = ttnn.to_torch(ttnn_out)
    ttnn_out = ttnn_out.reshape(1, ref_out.shape[2], ref_out.shape[3], ref_out.shape[1])
    ttnn_out = ttnn_out.permute(0, 3, 1, 2)

    passing, pcc_message = comp_pcc(ref_out, ttnn_out, 0.99)
    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_out, ttnn_out))
    logger.info(f"Weight prefix: {weight_key_prefix}, MLP: {mlp}, Features shape: {features_shape}")

    if passing:
        logger.info("SharedMLP Test Passed!")
    else:
        logger.warning("SharedMLP Test Failed!")

    assert passing, f"PCC value is lower than 0.999. Check implementation! {pcc_message}"
