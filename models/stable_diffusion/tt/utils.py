import torch

import tt_lib as ttl
from models.utility_functions import torch_to_tt_tensor_rm

from models.stable_diffusion.tt.fused_ops import Linear as SDLinear

def make_linear(in_features: int, out_features: int, weights: ttl.tensor.Tensor, bias: ttl.tensor.Tensor, device) -> SDLinear:
    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False) if bias is not None else None
    return SDLinear(in_features, out_features, weights, bias)
