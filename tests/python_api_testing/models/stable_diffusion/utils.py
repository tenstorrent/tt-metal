
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch

# from libs import tt_lib as ttl
import tt_lib as ttl
from utility_functions_new import torch_to_tt_tensor_rm

from python_api_testing.models.stable_diffusion.fused_ops import Linear as SDLinear

def make_linear(in_features: int, out_features: int, weights, bias, device):
    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False) if bias is not None else None
    return SDLinear(in_features, out_features, weights, bias)
