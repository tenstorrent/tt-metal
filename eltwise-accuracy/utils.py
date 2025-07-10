import numpy as np
import torch
import traceback

import pytest

TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"


def all_bf16_tensor():
    size = [2**9, 2**7]

    input_np = np.arange(0, 2**16, dtype=np.uint16)
    print(f"input_np: ({input_np.size}) \n{input_np}")

    torch_values = torch.from_numpy(input_np.view(np.int16)).reshape(size)
    torch_values_bf16 = torch_values.view(torch.bfloat16)

    return torch_values_bf16


def generate_binary_tensors_bf16():
    batch_size = 128
    num_batches = 2**16 // batch_size

    shape = [batch_size, 2**9, 2**7]

    all_bf16 = all_bf16_tensor()
    tensor_a = all_bf16.reshape([1, 2**9, 2**7]).expand(batch_size, -1, -1)

    # each slice of tensor_b contains the same value
    # tensor_b cotnains 128 slices. Each slice has a unique value.
    # Value(slice[i]) = nextafter(Vallue(slice[i-1]))

    tensor_b_i16 = torch.arange(-(2**16), -(2**16) + batch_size, dtype=torch.int16)
    assert tensor_b_i16.shape == torch.Size([batch_size])

    tensor_b_i16 = tensor_b_i16.reshape([batch_size, 1, 1]).repeat(1, 2**9, 2**7)
    assert tensor_b_i16.shape == tensor_a.shape

    increment = batch_size

    for i in range(0, 2**16, batch_size):
        tensor_b = tensor_b_i16.view(torch.bfloat16)

        yield (tensor_a, tensor_b)

        tensor_b_i16 += increment


# Reduce tensor on first dimension
# Return a dictionary with reduced tensors
def reduce_tensor(tensor):
    return {
        "mean": tensor.mean(dim=0),
        "median": tensor.median(dim=0),
        "max": tensor.max(dim=0),
        "min": tensor.min(dim=0),
    }


def execute_benchmarks(benchmark_fun, operations, dest_dir):
    success_count = 0
    successfull_operations = []
    failed_operations = []

    cnt = 0
    total_operation_cnt = len(operations)
    for operation in operations:
        cnt += 1
        print(f"Running operation {operation}  #{cnt} / {total_operation_cnt}", end="\r")
        try:
            benchmark_fun(operation, dest_dir)
            success_count += 1
            successfull_operations += [operation]
        except Exception as e:
            print(f"{TERM_RED}Could not run operation {operation}: {e}{TERM_RESET}")
            print(f"{TERM_RED}{traceback.format_exc()}{TERM_RESET}")
            failed_operations += [operation]

    return (successfull_operations, failed_operations)


def setsgn_bf16_(input_tensor, sign):
    input_tensor_i16 = input_tensor.view(torch.int16)

    tensor_sign = torch.full(input_tensor.size(), sign, dtype=torch.int16)
    tensor_sign = tensor_sign.bitwise_left_shift_(15)
    tensor_mask0x7FFF = torch.full(input_tensor.size(), 0x7FFF, dtype=torch.int16)
    input_tensor_i16 = input_tensor_i16.bitwise_and_(tensor_mask0x7FFF)
    input_tensor_i16 = input_tensor_i16.bitwise_or_(tensor_sign)

    return input_tensor


# Return exponent of each tensor
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


def ulp(input_tensor):
    torch_max = torch.full(input_tensor.size(), torch.finfo(input_tensor.dtype).max, dtype=input_tensor.dtype)
    return torch.abs(input_tensor - torch.nextafter(input_tensor, torch_max))


def ulp_diff(output, golden):
    ulp_tensor = ulp(golden)
    diff = torch.abs(output - golden)
    return diff / ulp_tensor


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


def test_generate_binary_tensors_bf16():
    i = 0
    for tensor_a, tensor_b in generate_binary_tensors_bf16():
        # print(f"tensor_a: ({tensor_a.size()}) \n{tensor_a}")
        print(f"tensor_b: ({tensor_b.size()}) \n{tensor_b}")

        # All elements of tensor_b must have same exponent
        tensor_b_exp = exexp_bf16(tensor_b)
        assert tensor_b_exp.unique().size() == torch.Size([1])
        i += 1
