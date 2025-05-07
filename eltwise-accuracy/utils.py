import numpy as np
import torch

import pytest


def all_bf16_tensor():
    size = [2**9, 2**7]

    input_np = np.arange(0, 2**16, dtype=np.uint16)
    print(f"input_np: ({input_np.size}) \n{input_np}")

    torch_values = torch.from_numpy(input_np.view(np.int16)).reshape(size)
    torch_values_bf16 = torch_values.view(torch.bfloat16)

    return torch_values_bf16


def setsgn_bf16_(input_tensor, sign):
    tensor_sign = torch.full(sign, dtype=torch.int16)
    tensor_sign = torch.bitwise_left_shift_(tensor_sign, 15)
    tensor_mask0x7FFF = torch.full(input_tensor.size(), 0x7FFF, dtype=torch.int16)
    input_tensor = torch.bitwise_and_(input_tensor, tensor_mask0x7FFF)
    input_tensor = torch.bitwise_or_(input_tensor, tensor_sign)

    return input_tensor


def exexp_bf16(input_tensor):
    tensor_mask0x7F80 = torch.full(input_tensor.size(), 0x7F80, dtype=torch.int16)
    tensor_i127 = torch.full(input_tensor.size(), 127, dtype=torch.int16)
    tensor_exp_bits = torch.bitwise_and(tensor_mask0x7F80, input_tensor)
    tensor_exp_bits = torch.bitwise_right_shift_(tensor_exp_bits, 7)

    tensor_exp_bits = torch.sub_(tensor_exp_bits, tensor_i127)

    return tensor_exp_bits


def setexp_bf16_(input_tensor, exponent):
    exp_bits = ((exponent + 127) & 0xFF) << 7
    tensor_exp_bits = torch.full(input_tensor.size(), exp_bits, dtype=torch.int16)
    exp_bits = torch.full(input_tensor.size(), exponent, dtype=torch.int16)
    tensor_mask0x807F = torch.full(input_tensor.size(), 0x807F, dtype=torch.int16)
    input_tensor = torch.bitwise_and_(input_tensor, tensor_mask0x807F)
    input_tensor = torch.bitwise_or_(input_tensor, tensor_exp_bits)

    return input_tensor


# For each element, returns the ULP distance to the nearest representable number
# Does not work with infinite values
def ulp_bf16(input_tensor):
    # input_tensor_i16 = input_tensor.view(torch.int16)
    # tensor_last_bit = torch.full(input_tensor.size(), 1, dtype=torch.int16)
    # tensor_mask0x7F80 = torch.full(input_tensor.size(), 0x7F80, dtype=torch.int16)
    # tensor_exp_bits = torch.bitwise_and(tensor_mask0x7F80, input_tensor_i16)
    # eps = torch.bitwise_or(tensor_exp_bits, tensor_last_bit)

    # eps = eps.view(torch.bfloat16)

    torch_max_bfloat16 = torch.full(input_tensor.size(), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16)
    return torch.abs(input_tensor - torch.nextafter(input_tensor, torch_max_bfloat16))


# Tests
def test_ulp_bf16():
    all_inputs = all_bf16_tensor()

    # np_all_inputs = np.arange(0, 2**16, dtype=np.uint16)
    # np_all_inputs_u16 = np_all_inputs.view(np.bfloat16)
    # np_ulp = np.spacing(np_all_inputs_u16)

    # torch_expected_ulp = torch.from_numpy(np_ulp).reshape(all_inputs.size()).view(torch.bfloat16)

    torch_ulp = ulp_bf16(all_inputs)

    print(f"all_inputs: ({all_inputs.size()}) \n{all_inputs}")
    # print(f"torch_expected_ulp: ({torch_expected_ulp.size()}) \n{torch_expected_ulp}")
    print(f"torch_ulp: ({torch_ulp.size()}) \n{torch_ulp}")

    ulp_one = torch_ulp[127]
    val_one = all_inputs[127]

    print(f"ulp_one: {ulp_one}")
    print(f"val_one: {val_one}")
    # assert torch_ulp == torch_expected_ulp
