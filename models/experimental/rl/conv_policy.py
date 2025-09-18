import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.rl.recurrent_policy import LSTMCell


class CNNPolicy(nn.Module):
    def __init__(self, obs_channels, action_dim):
        """
        obs_channels: number of input channels (e.g., 4 for Atari frame stacks)
        action_dim: number of actions in the environment
        """
        super(CNNPolicy, self).__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened size (assuming input is 84x84 like Atari)
        example_input = torch.zeros(1, obs_channels, 84, 84)
        with torch.no_grad():
            conv_out = self._forward_conv(example_input)
            self.flattened_size = conv_out.shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def forward(self, obs):
        """
        obs: [batch_size, channels, height, width]
        returns: [batch_size, action_dim]
        """
        x = self._forward_conv(obs)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNNLSTMPolicy(nn.Module):
    def __init__(self, obs_channels, action_dim, hidden_dim=512):
        """
        obs_channels: input image channels
        action_dim: number of actions
        hidden_dim: size of LSTM hidden state
        """
        super(CNNLSTMPolicy, self).__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Determine flattened size
        example_input = torch.zeros(1, obs_channels, 84, 84)
        with torch.no_grad():
            conv_out = self._forward_conv(example_input)
            self.flattened_size = conv_out.shape[1]

        # Feature projection before LSTM
        self.fc = nn.Linear(self.flattened_size, hidden_dim)

        # Custom LSTM cell (no nn.LSTM used)
        self.lstm_cell = LSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: [batch_size, seq_len, channels, height, width]
        hidden_state: (h, c) each [batch_size, hidden_dim]
        """
        batch_size, seq_len, C, H, W = obs_seq.shape

        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=obs_seq.device)
            c_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=obs_seq.device)
        else:
            h_t, c_t = hidden_state

        outputs = []

        for t in range(seq_len):
            # CNN feature extraction
            features = self._forward_conv(obs_seq[:, t, :, :, :])
            features = F.relu(self.fc(features))

            # Pass through custom LSTM
            h_t, c_t = self.lstm_cell(features, h_t, c_t)

            # Action output
            actions = self.action_head(h_t)
            outputs.append(actions)

        return torch.stack(outputs, dim=1), (h_t, c_t)

    def init_hidden(self, batch_size, device="cpu"):
        """Initialize LSTM hidden and cell states"""
        return (
            torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=device),
            torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=device),
        )


# # Example: Atari with 4 stacked frames, 6 possible actions
# policy = CNNPolicy(obs_channels=4, action_dim=6)

# # Random batch of images
# obs = torch.randn(2, 4, 84, 84)

# actions = policy(obs)
# print("CNN Policy output shape:", actions.shape)  # [2, 6]


# # Example: Atari with 4 stacked frames, 6 possible actions
# policy = CNNLSTMPolicy(obs_channels=4, action_dim=6, hidden_dim=512)

# # Batch of 2 sequences, each 5 timesteps long
# obs_seq = torch.randn(2, 5, 4, 84, 84)

# # Initialize hidden state
# hidden = policy.init_hidden(batch_size=2, device=obs_seq.device)

# # Forward pass
# actions, next_hidden = policy(obs_seq, hidden)

# print("CNN + LSTM Policy output shape:", actions.shape)  # [2, 5, 6]
