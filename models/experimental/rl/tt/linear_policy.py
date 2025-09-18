import ttnn
import torch.nn as nn
import torch.nn.functional as F
from models.common.lightweightmodule import LightweightModule


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(ThreeLayerMLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TTMLPPolicy(LightweightModule):
    def __init__(self, mesh_device: ttnn.MeshDevice, action_dim: int, hidden_dim: int, state_dict: dict):
        super().__init__()

        self.fc1_weight = ttnn.from_torch(state_dict["fc1.weight"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc1_bias = ttnn.from_torch(state_dict["fc1.bias"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc2_weight = ttnn.from_torch(state_dict["fc2.weight"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc2_bias = ttnn.from_torch(state_dict["fc2.bias"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc3_weight = ttnn.from_torch(state_dict["fc3.weight"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc3_bias = ttnn.from_torch(state_dict["fc3.bias"], mesh_device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc1_weight = ttnn.transpose(self.fc1_weight, -2, -1)
        self.fc2_weight = ttnn.transpose(self.fc2_weight, -2, -1)
        self.fc3_weight = ttnn.transpose(self.fc3_weight, -2, -1)

    def forward(self, obs):
        obs = ttnn.matmul(obs, self.fc1_weight)
        obs = ttnn.add(obs, self.fc1_bias)
        obs = ttnn.relu(obs)

        obs = ttnn.matmul(obs, self.fc2_weight)
        obs = ttnn.add(obs, self.fc2_bias)
        obs = ttnn.relu(obs)

        obs = ttnn.matmul(obs, self.fc3_weight)
        obs = ttnn.add(obs, self.fc3_bias)
        obs = ttnn.relu(obs)
        return obs
