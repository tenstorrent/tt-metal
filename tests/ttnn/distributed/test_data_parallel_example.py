# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


class TtFalconMLP:
    def __init__(self, parameters):
        super().__init__()
        self.dense_h_to_4h_weights = parameters.dense_h_to_4h.weight
        self.dense_4h_to_h_weights = parameters.dense_4h_to_h.weight

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ff1_linear: ttnn.Tensor = ttnn.linear(x, self.dense_h_to_4h_weights)
        gelu = ttnn.gelu(ff1_linear)
        ff2_linear: ttnn.Tensor = ttnn.linear(gelu, self.dense_4h_to_h_weights)

        return ff2_linear


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
def test_data_parallel_falcon_mlp(mesh_device):
    # Load Falcon MLP model from huggingface
    config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
    model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()

    # Initialize hidden states
    batch_size, sequence_length = 4, 128
    torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
    torch_output = model.forward(torch_hidden_states)

    # Shard input activations on batch dimension to devices in the mesh
    with ttnn.distribute(mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)):
        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

    # Replicate model parameters to devices in the mesh
    with ttnn.distribute(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
        )

    # Initialize Model
    ttnn_model = TtFalconMLP(parameters)
    ttnn_output = ttnn_model(hidden_states)

    with ttnn.distribute(mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
        assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.98)
