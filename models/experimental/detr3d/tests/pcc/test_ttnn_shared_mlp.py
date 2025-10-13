# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.detr3d.ttnn.shared_mlp import TtnnSharedMLP
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.detr3d.reference.detr3d_model import SharedMLP
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "mlp,bn,features_shape,weight_key_prefix",
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

    # Load actual model weights from checkpoint
    weights_path = "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
    state_dict = torch.load(weights_path, map_location="cpu")["model"]

    # Extract weights for the specific SharedMLP module
    shared_mlp_state_dict = {k: v for k, v in state_dict.items() if k.startswith(weight_key_prefix)}

    # Map the checkpoint keys to the SharedMLP model keys
    new_state_dict = {}
    model_keys = [name for name, parameter in torch_model.state_dict().items()]
    checkpoint_values = [parameter for name, parameter in shared_mlp_state_dict.items()]

    for i in range(len(model_keys)):
        new_state_dict[model_keys[i]] = checkpoint_values[i]

    # Load the mapped weights into the model
    torch_model.load_state_dict(new_state_dict, strict=True)
    print(f"Successfully loaded weights for {weight_key_prefix}")
    torch_model.eval()
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
    ttnn_model = TtnnSharedMLP(torch_model, parameters, device)
    ttnn_out = ttnn_model(ttnn_features)

    ttnn_out = ttnn.to_torch(ttnn_out)
    ttnn_out = ttnn_out.reshape(1, ref_out.shape[2], ref_out.shape[3], ref_out.shape[1])
    ttnn_out = ttnn_out.permute(0, 3, 1, 2)

    assert_with_pcc(
        ref_out, ttnn_out, 0.999
    )  # pre_encoder.mlp_module - 0.9993508842817294; encoder.interim_downsampling.mlp_module - 0.99924764727396
