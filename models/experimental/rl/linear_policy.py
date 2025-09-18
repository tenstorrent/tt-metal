import pytest
import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utility_functions import comp_pcc


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.bfloat16)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.bfloat16)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TTMLPPolicy:
    def __init__(self, state_dict, mesh_device):
        self.fc1 = ttnn.from_torch(
            state_dict["fc1.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2 = ttnn.from_torch(
            state_dict["fc2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc3 = ttnn.from_torch(
            state_dict["fc3.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["fc1.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["fc2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc3_b = ttnn.from_torch(state_dict["fc3.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, obs: ttnn.Tensor) -> ttnn.Tensor:
        x1 = ttnn.linear(obs, self.fc1, bias=self.fc1_b, activation="relu")
        obs.deallocate()

        x2 = ttnn.linear(x1, self.fc2, bias=self.fc2_b, activation="relu")
        x1.deallocate()

        x3 = ttnn.linear(x2, self.fc3, bias=self.fc3_b, activation="relu")
        x2.deallocate()

        return x3


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_mlp_policy(mesh_device):
    # Params
    batch_size = 1
    action_dim = 10
    hidden_dim = 64
    obs_dim = 128

    # Create input
    x = torch.randn(1, obs_dim, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    # Create torch model
    policy = MLPPolicy(obs_dim, action_dim, hidden_dim)

    # Create TT model
    tt_policy = TTMLPPolicy(policy.state_dict(), mesh_device)

    # Run inference
    y = policy(x)
    tt_y = tt_policy(tt_x)
    tt_y = ttnn.to_torch(tt_y)

    # Do PCC check
    passing, pcc_message = comp_pcc(y, tt_y, 0.99)
