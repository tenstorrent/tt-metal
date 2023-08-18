import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib


class TtConvLayerNorm(nn.Module):
    def __init__(
        self,
        config,
        cin: int,
        cout: int,
        groups: int,
        base_address="",
        state_dict=None,
        device=None,
    ) -> None:
        super().__init__()

        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device
        self.cout = cout

        self.conv1d = nn.Conv1d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=1,
            groups=groups,
        )
        self.conv1d.weight = nn.Parameter(
            state_dict[f"{self.base_address}.conv1d.weight"].to(dtype=torch.bfloat16)
        )
        self.conv1d.bias = nn.Parameter(
            state_dict[f"{self.base_address}.conv1d.bias"].to(dtype=torch.bfloat16)
        )

        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.layernorm.weight"], self.device
        )
        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.layernorm.bias"], self.device
        )

        self.LayerNorm = tt_lib.tensor.layernorm

    def forward(
        self, hidden_state: tt_lib.tensor.Tensor, input_tensor: tt_lib.tensor.Tensor
    ) -> tt_lib.tensor.Tensor:
        hidden_state = tt_to_torch_tensor(hidden_state).squeeze(0)
        hidden_state = self.conv1d(hidden_state)
        x = torch_to_tt_tensor_rm(hidden_state, self.device, put_on_device=False)

        x = tt_lib.tensor.add(x, input_tensor)
        x = tt_lib.tensor.transpose(x)
        x = self.LayerNorm(x, eps=1e-12, gamma=self.gamma, beta=self.beta)
        return tt_lib.tensor.transpose(x)
