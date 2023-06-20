import tt_lib
import tt_lib.tensor as tensor
from typing import Union, Optional, Tuple, Dict, Set, List
from utility_functions_new import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor



def linear(in_features: int, out_features: int, weight: tensor.Tensor, bias: Optional[tensor.Tensor] = None):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight_T = tensor.transpose(weight)

    def linear_(activation):
        assert activation.shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_

def make_address(base_address, op_name):
    return op_name if base_address == "" else f"{base_address}.{op_name}"

def make_linear(in_feature, out_feature, op_name, state_dict, base_address, device):
            # print('\nmake linear base address:', base_address)
            # print('\nmake linear final address:', make_address(base_address, f"{op_name}.weight"))
            q_weight = state_dict[make_address(base_address, f"{op_name}.weight")]
            q_weight = torch_to_tt_tensor_rm(q_weight, device)
            if make_address(base_address, f"{op_name}.bias") in state_dict:
                q_bias = state_dict[make_address(base_address, f"{op_name}.bias")]
                q_bias = torch_to_tt_tensor_rm(q_bias, device)
            else:
                q_bias = None
            return linear(in_feature, out_feature, weight=q_weight, bias=q_bias)
