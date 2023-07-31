import torch
from torch import nn
import tt_lib

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch2tt_tensor


class TtFalconMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.bias = None

        self.dense_h_to_4h_weight = torch2tt_tensor(
            self.state_dict[f"{base_url}.{layer_num}.mlp.dense_h_to_4h.weight"],
            self.device,
        )
        self.dense_4h_to_h_weight = torch2tt_tensor(
            self.state_dict[f"{base_url}.{layer_num}.mlp.dense_4h_to_h.weight"],
            self.device,
        )

        self.act_fn = tt_lib.tensor.gelu

        self.dense_h_to_4h = TTLinear(
            self.hidden_size,
            self.hidden_size * 4,
            self.dense_h_to_4h_weight,
            self.bias,
        )
        self.dense_4h_to_h = TTLinear(
            self.hidden_size * 4,
            self.hidden_size,
            self.dense_4h_to_h_weight,
            self.bias,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = self.dense_h_to_4h(x)

        # apply gelu activation function
        x = self.act_fn(x)

        hidden_states = self.dense_4h_to_h(x)

        # return TT Tensor
        return hidden_states
