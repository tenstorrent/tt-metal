import torch
from torch.nn import functional as F

from libs import tt_lib as ttm
from fused_ops.linear import Linear as TtLinear

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_gelu_forward as bloom_gelu_forward


# class BloomMLP(torch.nn.Module):

#     def __init__(self, dict_name, num, state_dict, hidden_dropout, hidden_size, training):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.hidden_dropout = hidden_dropout
#         self.training = training

#         self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
#         self.gelu_impl = bloom_gelu_forward.bloom_gelu_forward
#         self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)

#         weight_mlp_h_to_4h = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_h_to_4h.weight", state_dict)
#         bias_mlp_h_to_4h =  bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_h_to_4h.bias", state_dict)

#         weight_mlp_4hh = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_4h_to_h.weight", state_dict)
#         bias_mlp_4hh = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_4h_to_h.bias", state_dict)

#         self.dense_4h_to_h.weight = weight_mlp_4hh
#         self.dense_4h_to_h.bias = bias_mlp_4hh
#         self.dense_h_to_4h.weight = weight_mlp_h_to_4h
#         self.dense_h_to_4h.bias = bias_mlp_h_to_4h

#     def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
#         intermediate_output = self.dense_4h_to_h(hidden_states)

#         # Dropout is used in training only
#         # intermediate_output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
#         output = residual + intermediate_output

#         return output


class TtBloomMLP(torch.nn.Module):
    def __init__(self, device, dict_name, num, hugging_bloom_reference_model, hidden_dropout, hidden_size, training):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.training = training

        state_dict = hugging_bloom_reference_model.state_dict()

        tt_weight_mlp_h4h = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_h_to_4h.weight", state_dict)
        tt_bias_mlp_h4h =  bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_h_to_4h.bias", state_dict)

        tt_weight_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_4h_to_h.weight", state_dict)
        tt_bias_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{dict_name}.{num}.mlp.dense_4h_to_h.bias", state_dict)

        self.dense_h_to_4h = TtLinear(hidden_size, 4 * hidden_size, tt_weight_mlp_h4h, tt_bias_mlp_h4h, device)
        self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward
        self.dense_4h_to_h = TtLinear(4*hidden_size, hidden_size, tt_weight_mlp_4hh, tt_bias_mlp_4hh, device)

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
