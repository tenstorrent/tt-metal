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

        # Invoke CCL Ring All-Gather on gelu before passing to ff2_linear
        gelu = ttnn.all_gather(gelu, dim=3, num_links=1)

        ff2_linear: ttnn.Tensor = ttnn.linear(gelu, self.dense_4h_to_h_weights)

        return ff2_linear


def test_tensor_parallel_falcon_mlp():
    if ttnn.get_num_devices() < 8:
        pytest.skip()

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(2, 4),
        mesh_type=ttnn.MeshType.Ring,
    )

    # Set PyTorch seed for reproducibility
    torch.manual_seed(0)

    # Load Falcon MLP model from huggingface
    config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
    model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()

    # Initialize hidden states
    batch_size, sequence_length = 1, 256
    torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
    torch_output = model.forward(torch_hidden_states)

    # Initialize input activations on all devices in the mesh
    # Alternatively, we can shard the input activations on the height dimension and
    # subsequently invoke all-gather on the height dimension to form a complete tensor per device.
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

    # Shard model parameters on width dimension to devices in the mesh
    with ttnn.distribute(ttnn.ShardTensorToMesh(mesh_device, dim=-1)):
        parameters = ttnn.model_preprocessing.preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
        )

    # Initialize Model
    ttnn_model = TtFalconMLP(parameters)

    # Run Model
    ttnn_output = ttnn_model(hidden_states)

    with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=3)):
        assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.98)

    ttnn.close_mesh_device(mesh_device)
