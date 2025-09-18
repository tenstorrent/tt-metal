import pytest
import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.lightweightmodule import LightweightModule
from models.utility_functions import comp_pcc

### -------- LSTM Cell --------


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # A single big linear that outputs all 4 gates at once
        # Input: concatenated [x_t, h_{t-1}] -> Output: 4 * hidden_dim
        self.gate_linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim, dtype=torch.bfloat16)

    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: [batch_size, input_dim]
        h_prev: [batch_size, hidden_dim]
        c_prev: [batch_size, hidden_dim]
        """
        # Concatenate input and previous hidden
        combined = torch.cat([x_t, h_prev], dim=-1)

        # Compute all gates in one pass
        gates = self.gate_linear(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)

        # Apply activations
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)  # candidate cell state

        # Update cell state and hidden state
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class TTLSTMCell(LightweightModule):
    def __init__(self, mesh_device: ttnn.MeshDevice, state_dict: dict):
        self.mesh_device = mesh_device
        self.hidden_dim = state_dict["gate_linear.weight"].shape[0] // 4
        self.gate_linear_weight = ttnn.from_torch(
            state_dict["gate_linear.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.gate_linear_bias = ttnn.from_torch(
            state_dict["gate_linear.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden
        x_t = ttnn.to_layout(x_t, layout=ttnn.ROW_MAJOR_LAYOUT)
        h_prev = ttnn.to_layout(h_prev, layout=ttnn.ROW_MAJOR_LAYOUT)
        combined = ttnn.concat([x_t, h_prev], dim=-1)
        ttnn.deallocate(x_t)
        ttnn.deallocate(h_prev)

        combined = ttnn.to_layout(combined, layout=ttnn.TILE_LAYOUT)

        gates_out = ttnn.matmul(combined, self.gate_linear_weight)
        gates_out = ttnn.add(gates_out, self.gate_linear_bias)

        i, f, g, o = ttnn.chunk(gates_out, 4, dim=-1)

        i_act = ttnn.sigmoid(i)
        f_act = ttnn.sigmoid(f)
        g_act = ttnn.sigmoid(g)
        o_act = ttnn.tanh(o)

        ttnn.deallocate(i)
        ttnn.deallocate(f)
        ttnn.deallocate(g)
        ttnn.deallocate(o)

        act_1 = ttnn.mul(f_act, c_prev)
        act_2 = ttnn.mul(i_act, g_act)

        c_t = act_1 + act_2
        act_4 = ttnn.tanh(c_t)
        h_t = ttnn.mul(act_4, o_act)

        return h_t, c_t


### -------- RecurrentMLPPolicy --------


class RecurrentMLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RecurrentMLPPolicy, self).__init__()

        # MLP feature extractor
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)

        # Custom LSTM Cell
        self.lstm_cell = LSTMCell(hidden_dim, hidden_dim)

        # Output layer for action logits
        self.action_head = nn.Linear(hidden_dim, action_dim, dtype=torch.bfloat16)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: [batch_size, seq_len, obs_dim]
        hidden_state: (h_0, c_0), each [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = obs_seq.shape

        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=obs_seq.device, dtype=torch.bfloat16)
            c_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=obs_seq.device, dtype=torch.bfloat16)
        else:
            h_t, c_t = hidden_state

        outputs = []

        # Process sequence step-by-step
        for t in range(seq_len):
            x = F.relu(self.fc1(obs_seq[:, t, :]))
            x = F.relu(self.fc2(x))

            # LSTM step
            h_t, c_t = self.lstm_cell(x, h_t, c_t)

            # Compute action
            action = self.action_head(h_t)
            outputs.append(action)

        # Stack outputs along time
        actions = torch.stack(outputs, dim=1)  # [batch_size, seq_len, action_dim]
        return actions, (h_t, c_t)

    def init_hidden(self, batch_size=1, device="cpu"):
        """Initialize LSTM hidden and cell states."""
        return (
            torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=device),
            torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=device),
        )


class TTRecurrentMLPPolicy(LightweightModule):
    def __init__(self, mesh_device, state_dict):
        self.mesh_device = mesh_device
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
        self.action_head = ttnn.from_torch(
            state_dict["action_head.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc1_b = ttnn.from_torch(state_dict["fc1.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["fc2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.action_head_b = ttnn.from_torch(
            state_dict["action_head.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        prefix = "lstm_cell."
        cell_state_dict = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
        self.lstm_cell = TTLSTMCell(mesh_device, cell_state_dict)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: [batch_size, seq_len, obs_dim]
        hidden_state: (h_0, c_0), each [batch_size, hidden_dim]
        """
        batch_size, seq_len, obs_dim = obs_seq.shape
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, dtype=torch.bfloat16)
            h_t = ttnn.from_torch(h_t, device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            c_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, dtype=torch.bfloat16)
            c_t = ttnn.from_torch(c_t, device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            h_t, c_t = hidden_state

        outputs = []

        # Process sequence step-by-step
        for t in range(seq_len):
            obs_seq_slice = ttnn.slice(obs_seq, [0, t, 0], [batch_size, t + 1, obs_dim])
            obs_seq_slice = ttnn.reshape(obs_seq_slice, (batch_size, obs_dim))

            # FC1
            x = ttnn.matmul(obs_seq_slice, self.fc1)
            x = ttnn.add(x, self.fc1_b)
            x = ttnn.relu(x)

            # FC2
            x = ttnn.matmul(x, self.fc2)
            x = ttnn.add(x, self.fc2_b)
            x = ttnn.relu(x)

            # LSTM step
            h_t, c_t = self.lstm_cell(x, h_t, c_t)

            # Compute action
            action = ttnn.matmul(h_t, self.action_head)
            action = ttnn.add(action, self.action_head_b)
            action = ttnn.relu(action)
            outputs.append(action)

        # Stack outputs along time
        actions = ttnn.stack(outputs, dim=1)  # [batch_size, seq_len, action_dim]
        return actions, (h_t, c_t)


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_lstm_cell(mesh_device):
    # Parameters
    batch_size = 2
    input_dim = 32
    hidden_dim = 64

    # Initialize the LSTM cell
    cell = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
    tt_cell = TTLSTMCell(mesh_device, cell.state_dict())

    # Create dummy inputs
    x_t = torch.randn(batch_size, input_dim, dtype=torch.bfloat16)  # current timestep input
    h_prev = torch.zeros(batch_size, hidden_dim, dtype=torch.bfloat16)  # previous hidden state
    c_prev = torch.zeros(batch_size, hidden_dim, dtype=torch.bfloat16)  # previous cell state

    # Create TT inputs
    tt_x_t = ttnn.from_torch(x_t, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_h_prev = ttnn.from_torch(h_prev, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_c_prev = ttnn.from_torch(c_prev, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    # Forward pass
    h_t, c_t = cell(x_t, h_prev, c_prev)
    tt_h_t, tt_c_t = tt_cell(tt_x_t, tt_h_prev, tt_c_prev)
    tt_h_t = ttnn.to_torch(tt_h_t)
    tt_c_t = ttnn.to_torch(tt_c_t)

    # Do PCC Check
    passing, pcc_message = comp_pcc(h_t, tt_h_t, 0.99)
    passing, pcc_message = comp_pcc(c_t, tt_c_t, 0.99)


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_lstm_policy(mesh_device):
    # Example setup
    obs_dim = 8
    action_dim = 32
    seq_len = 5
    batch_size = 2

    policy = RecurrentMLPPolicy(obs_dim, action_dim, hidden_dim=64)
    tt_policy = TTRecurrentMLPPolicy(mesh_device, policy.state_dict())

    # Random input
    obs_seq = torch.randn(batch_size, seq_len, obs_dim, dtype=torch.bfloat16)
    tt_obs_seq = ttnn.from_torch(obs_seq, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    # Forward pass
    actions, next_hidden = policy(obs_seq)
    tt_actions, tt_next_hidden = tt_policy(tt_obs_seq)
    tt_actions = ttnn.to_torch(tt_actions)

    # Do PCC check
    passing, pcc_message = comp_pcc(actions, tt_actions, 0.99)
