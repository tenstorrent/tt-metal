# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
import transformers
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

CLUSTER_AXIS_X = 1


class TtFalconMLP:
    def __init__(self, parameters, mesh_device):
        super().__init__()
        self.mesh_device = mesh_device
        self.dense_h_to_4h_weights = parameters.dense_h_to_4h.weight
        self.dense_4h_to_h_weights = parameters.dense_4h_to_h.weight

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ff1_linear: ttnn.Tensor = ttnn.linear(x, self.dense_h_to_4h_weights)
        gelu = ttnn.gelu(ff1_linear)

        # Effectively invokes CCL Line All Gather for every row of the mesh
        gelu = ttnn.all_gather(
            gelu,
            dim=-1,
            num_links=1,
            cluster_axis=CLUSTER_AXIS_X,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Linear,
        )

        ff2_linear: ttnn.Tensor = ttnn.linear(gelu, self.dense_4h_to_h_weights)

        return ff2_linear


def test_tensor_parallel_falcon_mlp():
    if ttnn.get_num_devices() < 8:
        pytest.skip()

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(2, 4),
    )

    # Set PyTorch seed for reproducibility
    torch.manual_seed(0)

    # Load Falcon MLP model from huggingface
    config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
    model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()

    # Initialize hidden states
    batch_size, sequence_length = 2, 256
    torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
    torch_output = model.forward(torch_hidden_states)

    # DP = 2; shard activations on batch-dim: [2,1,sequence_length,hidden_size] and replicate along columns of the mesh
    # [A0, A0, A0, A0]
    # [A1, A1, A1, A1]
    hidden_states, parameters = None, None
    mesh_shape = tuple(mesh_device.shape)

    with ttnn.distribute(ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(0, None))):
        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

    # TP = 4; ctx manager replicate model weights along rows of the mesh and shards replicas on columns of the mesh
    # [W0, W1, W2, W3]
    # [W0, W1, W2, W3]
    with ttnn.distribute(ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, -1))):
        parameters = ttnn.model_preprocessing.preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
        )

    # Initialize Model
    ttnn_model = TtFalconMLP(parameters, mesh_device)

    # Run Model
    ttnn_output = ttnn_model(hidden_states)

    with ttnn.distribute(ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(2, 4), dims=(0, -1))):
        assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.98)
