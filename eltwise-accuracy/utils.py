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
    input_tensor_i16 = input_tensor.view(torch.int16)

    tensor_sign = torch.full(input_tensor.size(), sign, dtype=torch.int16)
    tensor_sign = tensor_sign.bitwise_left_shift_(15)
    tensor_mask0x7FFF = torch.full(input_tensor.size(), 0x7FFF, dtype=torch.int16)
    input_tensor_i16 = input_tensor_i16.bitwise_and_(tensor_mask0x7FFF)
    input_tensor_i16 = input_tensor_i16.bitwise_or_(tensor_sign)

    return input_tensor


def exexp_bf16(input_tensor):
    input_tensor_i16 = input_tensor.view(torch.int16)

    volume = np.prod(input_tensor.size())
    np_mask0x7F80 = np.full(input_tensor.size(), 0x7F80, dtype=np.uint16).view(np.int16)

    tensor_mask0x7F80 = torch.from_numpy(np_mask0x7F80)
    tensor_i127 = torch.full(input_tensor.size(), 127, dtype=torch.int16)
    tensor_exp_bits = torch.bitwise_and(tensor_mask0x7F80, input_tensor_i16)
    tensor_exp_bits = tensor_exp_bits.bitwise_right_shift_(7)

    tensor_exp_bits = tensor_exp_bits.sub_(tensor_i127)

    return tensor_exp_bits


def setexp_bf16_(input_tensor, exponent):
    input_tensor_i16 = input_tensor.view(torch.int16)

    np_mask0x807F = np.full(input_tensor.size(), 0x807F, dtype=np.uint16).view(np.int16)

    tensor_mask0x807F = torch.from_numpy(np_mask0x807F).reshape(input_tensor.size())

    exp_bits_scalar = ((exponent + 127) & 0xFF) << 7
    tensor_exp_bits = torch.full(input_tensor.size(), exp_bits_scalar, dtype=torch.int16)

    input_tensor_i16 = input_tensor_i16.bitwise_and_(tensor_mask0x807F)
    input_tensor_i16 = input_tensor_i16.bitwise_or_(tensor_exp_bits)

    input_tensor = input_tensor_i16.view(torch.bfloat16)

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


def test_setsgn_bf16_():
    tensor_dims = [3, 4]
    all_inputs = torch.randint(-4, 5, tensor_dims, dtype=torch.bfloat16)

    print(f"INPUT:\n\n{all_inputs}")

    setsgn_bf16_(all_inputs, 0)

    print(f"OUTPUT:\n\n{all_inputs}")


def test_exexp_bf16():
    # all_inputs = torch.randint(-32, 32, tensor_dims, dtype=torch.bfloat16)

    np_input = np.array([0, 0.5, 1, 1.25, 1.5, 1.75, 2], dtype=np.float32)
    all_inputs = torch.from_numpy(np_input).type(torch.bfloat16)

    print(f"INPUT:\n\n{all_inputs}")

    tensor_exp = exexp_bf16(all_inputs)

    print(f"OUTPUT:\n\n{tensor_exp}")


def test_sexexp_bf16_():
    #    tensor_dims = [3, 4]
    #    all_inputs = torch.randint(8, 16, tensor_dims, dtype=torch.bfloat16)

    np_input = np.array([0, 0.5, 1, 1.25, 1.5, 1.75, 2], dtype=np.float32)
    all_inputs = torch.from_numpy(np_input).type(torch.bfloat16)

    print()
    print(f"INPUT:\n\n{all_inputs}")

    setexp_bf16_(all_inputs, 1)

    print(f"OUTPUT:\n\n{all_inputs}")
