from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn

from typing import Optional

from libs.tt_lib.fallback_ops import fallback_ops
from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from utility_functions import pad_weight, tilize_to_list, torch_to_tt_tensor, tt_to_torch_tensor


class TtGEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int, device=None, host=None, state_dict=None, base_address="mid_block.attentions.0.transformer_blocks.0.ff.net.0"):
        super().__init__()
        self.device = device
        self.host = host

        weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj.weight"]))
        bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj.bias"]))
        self.proj = TtLinear(dim_in, dim_out * 2, weights, bias, device)


    def gelu(self, gate):
        return ttl.tensor.gelu(gate)
        # if gate.device.type != "mps":
            # return F.gelu(gate)
        # mps: gelu is not implemented for float16
        # return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)

        hidden_states, gate = fallback_ops.chunk(hidden_states, 2, -1)

        # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
        # hidden_states, gate = hidden_states.chunk(2, dim=-1)
        # hidden_states = torch_to_tt_tensor(hidden_states, self.device)
        # gate = torch_to_tt_tensor(gate, self.device)

        act = self.gelu(gate)
        return ttl.tensor.mul(hidden_states, act)


class TtFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        device=None,
        host=None,
        state_dict=None,
        base_address="mid_block.attentions.0.transformer_blocks.0.ff"
    ):
        super().__init__()
        assert dropout == 0.0, "we do not support dropout"
        assert final_dropout == False, "we do not support dropout"
        assert activation_fn == "geglu", "except GEGLU, other activation functions are not supported."
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim


        # if activation_fn == "gelu":
        #     act_fn = GELU(dim, inner_dim)
        # if activation_fn == "gelu-approximate":
        #     act_fn = GELU(dim, inner_dim, approximate="tanh")
        if activation_fn == "geglu":
            act_fn = TtGEGLU(dim, inner_dim, device, host, state_dict, base_address=f"{base_address}.net.0")
        else:
            assert False, "other activation ops are not implemented"
        # elif activation_fn == "geglu-approximate":
        #     act_fn = ApproximateGELU(dim, inner_dim)

        # project in
        self.act_fn = act_fn
        # project dropout

        # project out
        weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.net.2.weight"]))
        bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.net.2.bias"]))
        self.linear = TtLinear(inner_dim, dim_out, weights, bias, device)

        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout

        # if final_dropout: #Note: commented since dropout is not supported
        #     self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        hidden_states = self.act_fn(hidden_states)
        return self.linear(hidden_states)
