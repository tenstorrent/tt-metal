# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.detr3d.ttnn.ttnn_shared_mlp import TtnnSharedMLP
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.detr3d.reference.detr3d_model import SharedMLP


def custom_preprocessor_whole_model(model, name):
    parameters = {}
    if isinstance(model, SharedMLP):
        print("model.layer0.conv: ", model.layer0.conv)
        print("model.layer0.bn: ", model.layer0.bn.bn)
        weight, bias = fold_batch_norm2d_into_conv2d(model.layer0.conv, model.layer0.bn.bn)
        parameters["layer0"] = {}
        parameters["layer0"]["conv"] = {}
        parameters["layer0"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer1.conv, model.layer1.bn.bn)
        parameters["layer1"] = {}
        parameters["layer1"]["conv"] = {}
        parameters["layer1"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.layer2.conv, model.layer2.bn.bn)
        parameters["layer2"] = {}
        parameters["layer2"]["conv"] = {}
        parameters["layer2"]["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["layer2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


@pytest.mark.parametrize(
    "mlp,bn,features_shape",
    [
        ([3, 64, 128, 256], True, (1, 3, 2048, 64)),  # mlp  # bn
        # ([259, 256, 256, 256], True, (1, 259, 1024, 32)),  # mlp  # bn
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_shared_mlp(device, mlp, bn, features_shape, reset_seeds):
    torch_model = SharedMLP(mlp, bn=bn).to(torch.bfloat16)

    # weights_path = "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
    # state_dict = torch.load(weights_path)["model"]

    # pointnet_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("pre_encoder.mlp_module"))}
    # new_state_dict = {}
    # keys = [name for name, parameter in torch_model.state_dict().items()]
    # values = [parameter for name, parameter in pointnet_state_dict.items()]

    # for i in range(len(keys)):
    #     new_state_dict[keys[i]] = values[i]
    torch_model.eval()
    features = torch.randn(features_shape, dtype=torch.bfloat16)
    ref_out = torch_model(features)

    ttnn_features = ttnn.from_torch(
        features.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor_whole_model,
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
