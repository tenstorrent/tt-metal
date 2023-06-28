import torch
import tt_lib

from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)

from tt_lib.fallback_ops import fallback_ops


class TtYolov5Concat(torch.nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, state_dict, base_address, device, dimension=1):
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.d = dimension

    def forward(self, x):
        return fallback_ops.concat(x, self.d)
