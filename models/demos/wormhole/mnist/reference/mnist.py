# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


class MnistModel(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.fc1 = torch.nn.Linear(784, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        self.load_state_dict(state_dict)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)
        x = torch.nn.functional.relu(x)

        return torch.nn.functional.softmax(x)
