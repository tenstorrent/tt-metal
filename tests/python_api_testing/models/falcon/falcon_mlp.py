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

        # TODO: Take in model_config instead of hardcoding dtypes/mem_configs
        self.dense_h_to_4h_weights = tt_lib.tensor.transpose(
            torch2tt_tensor(
                self.state_dict[f"{base_url}.{layer_num}.mlp.dense_h_to_4h.weight"],
                self.device,
                tt_dtype=tt_lib.tensor.DataType.BFLOAT8_B,
            )
        )
        self.dense_4h_to_h_weights = tt_lib.tensor.transpose(
            torch2tt_tensor(
                self.state_dict[f"{base_url}.{layer_num}.mlp.dense_4h_to_h.weight"],
                self.device,
                tt_dtype=tt_lib.tensor.DataType.BFLOAT8_B,
            )
        )

        self.act_fn = tt_lib.tensor.gelu

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = tt_lib.tensor.falcon_dense_h_to_4h_matmul(x, self.dense_h_to_4h_weights)

        # apply gelu activation function
        x = self.act_fn(x)

        hidden_states = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
            x, self.dense_4h_to_h_weights
        )

        # return TT Tensor
        return hidden_states
