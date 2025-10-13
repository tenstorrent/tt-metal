# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.detr3d.ttnn.ttnn_generic_mlp import TttnnGenericMLP
from models.experimental.detr3d.reference.detr3d_model import GenericMLP
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
    activation,
    use_conv,
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
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
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ttnn_model = TttnnGenericMLP(torch_model, parameters, device)
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1).unsqueeze(dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_out = ttnn_model(ttnn_x)
    ttnn_out = ttnn.to_torch(ttnn_out)
    ttnn_out = ttnn_out.squeeze(dim=0)
    ttnn_out = torch.reshape(ttnn_out, (x_shape[-3], ttnn_out.shape[-2] // x_shape[-3], -1))
    ttnn_out = ttnn_out.permute(0, 2, 1)
    assert_with_pcc(torch_out, ttnn_out, 0.999)
