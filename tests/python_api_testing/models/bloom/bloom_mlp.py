import torch
from torch.nn import functional as F

from libs import tt_lib as ttm
from fused_ops.linear import Linear as TtLinear

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_gelu_forward as bloom_gelu_forward



# class BloomMLP(nn.Module):
#     def __init__(self, config: BloomConfig):
#         super().__init__()
#         hidden_size = config.hidden_size

#         self.pretraining_tp = config.pretraining_tp
#         self.slow_but_exact = config.slow_but_exact
#         self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
#         self.gelu_impl = BloomGelu()
#         self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
#         self.hidden_dropout = config.hidden_dropout

#     def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

#         if self.pretraining_tp > 1 and self.slow_but_exact:
#             intermediate_output = torch.zeros_like(residual)
#             slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
#             for i in range(self.pretraining_tp):
#                 intermediate_output = intermediate_output + F.linear(
#                     hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
#                     self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
#                 )
#         else:
#             intermediate_output = self.dense_4h_to_h(hidden_states)

#         output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

#         return output


class TtBloomMLP(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.training = False

        self.tt_weight_mlp_h4h = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_h_to_4h.weight", state_dict)
        self.tt_bias_mlp_h4h =  bloom_utils.tt_load_layer_weights(f"{base_address}.dense_h_to_4h.bias", state_dict)

        self.tt_weight_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_4h_to_h.weight", state_dict)
        self.tt_bias_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_4h_to_h.bias", state_dict)

        self.dense_h_to_4h = TtLinear(self.hidden_size, 4 * self.hidden_size, self.tt_weight_mlp_h4h.data(), self.tt_bias_mlp_h4h.data(), device)
        self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward
        self.dense_4h_to_h = TtLinear(4*self.hidden_size, self.hidden_size, self.tt_weight_mlp_4hh.data(), self.tt_bias_mlp_4hh.data(), device)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, device) -> torch.Tensor:

        tt_hs = bloom_utils.torch2tt_tensor(hidden_states, device)
        tt_h4h = self.dense_h_to_4h(tt_hs)

        tt_hidden_states = self.gelu_impl(tt_h4h, device)
        tt_intermediate_output = self.dense_4h_to_h(tt_hidden_states)

        tt_res = bloom_utils.torch2tt_tensor(residual, device)

        # Dropout is used in training only
        # tt_intermediate_output = F.dropout(tt_intermediate_output, p=self.hidden_dropout, training=self.training)
        output = ttm.tensor.add(tt_res, tt_intermediate_output)

        return output
