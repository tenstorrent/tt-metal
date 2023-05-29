from torch import nn
import tt_lib

from python_api_testing.models.t5.t5_utils import torch2tt_tensor
from tt_lib.fused_ops.linear import Linear as TtLinear


# class T5DenseActDense(nn.Module):
#     # def __init__(self, config: T5Config):
#     #     super().__init__()
#     #     self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
#     #     self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
#     #     self.dropout = nn.Dropout(config.dropout_rate)
#     #     self.act = ACT2FN[config.dense_act_fn]

#     def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
#         super().__init__()
#         self.wi = nn.Linear(d_model, d_ff, bias=False)
#         self.wo = nn.Linear(d_ff, d_model, bias=False)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.act = ACT2FN[dense_act_fn]

#     def forward(self, hidden_states):
#         hidden_states = self.wi(hidden_states)
#         hidden_states = self.act(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         if (
#             isinstance(self.wo.weight, torch.Tensor)
#             and hidden_states.dtype != self.wo.weight.dtype
#             and self.wo.weight.dtype != torch.int8
#         ):
#             hidden_states = hidden_states.to(self.wo.weight.dtype)
#         hidden_states = self.wo(hidden_states)
#         return hidden_states


class TtT5DenseActDense(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        d_model = config["d_model"]
        d_ff = config["d_ff"]
        dropout_rate = config["dropout_rate"]
        # dense_act_fn = config["dense_act_fn"]

        self.out_proj_wi = torch2tt_tensor(
            state_dict[f"{base_address}.wi.weight"], tt_lib.device.GetHost()
        )
        self.out_proj_w0 = torch2tt_tensor(
            state_dict[f"{base_address}.wo.weight"], tt_lib.device.GetHost()
        )

        self.wi = TtLinear(
            in_features=d_model,
            out_features=d_ff,
            weight=self.out_proj_wi.data(),
            bias=None,
            device=device,
        )
        self.wo = TtLinear(
            in_features=d_ff,
            out_features=d_model,
            weight=self.out_proj_w0.data(),
            bias=None,
            device=device,
        )

        # self.dropout = nn.Dropout(dropout_rate)

        # activation function
        self.act = tt_lib.tensor.relu

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
