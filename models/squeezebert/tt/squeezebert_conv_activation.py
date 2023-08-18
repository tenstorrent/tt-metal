import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib


class TtConvActivation(nn.Module):
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

        self.act = tt_lib.tensor.gelu

    def forward(self, input: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        input = tt_to_torch_tensor(input).squeeze(0)
        output = self.conv1d(input)
        output = torch_to_tt_tensor_rm(output, self.device, put_on_device=False)
        return self.act(output)
