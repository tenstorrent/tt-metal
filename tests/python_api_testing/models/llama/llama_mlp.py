import torch
from torch import nn
from libs import tt_lib as ttl
from python_api_testing.models.t5.t5_utils import tt2torch_tensor, torch2tt_tensor
from fused_ops.linear import Linear as TtLinear


class TtLlamaMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.out_gate_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.gate_proj.weight"], ttl.device.GetHost())
        self.out_down_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.down_proj.weight"], ttl.device.GetHost())
        self.out_up_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.up_proj.weight"], ttl.device.GetHost())

        self.gate_proj = TtLinear(in_features=self.hidden_size, out_features=self.intermediate_size, weight=self.out_gate_proj.data(), bias=None, device=self.device)
        self.down_proj = TtLinear(in_features=self.intermediate_size, out_features=self.hidden_size, weight=self.out_down_proj.data(), bias=None, device=self.device)
        self.up_proj = TtLinear(in_features=self.hidden_size, out_features=self.intermediate_size, weight=self.out_up_proj.data(), bias=None, device=self.device)

        if hidden_act == "silu": # $$ silu
            self.act_fn = ttl.tensor.sigmoid

    def forward(self, x):
        # gate proj
        gate = self.gate_proj(x)
        # apply silu activation function
        activation = self.act_fn(gate)
        gate = ttl.tensor.mul(gate, activation)
        # up proj
        up = self.up_proj(x)
        # product
        prod = ttl.tensor.mul(gate, up)
        # down
        hidden_states = self.down_proj(prod)
        # return TT Tensor
        return hidden_states
