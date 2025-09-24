# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_wormhole_b0
import copy
import pytest


TILE_HEIGHT = 32
TILE_WIDTH = 32


compute_kernel_options = [
    False,  # for grayskull
]
compute_kernel_ids = ["fp32_dest_acc_en=False"]
if is_wormhole_b0:
    compute_kernel_options.append(True)
    compute_kernel_ids.append("fp32_dest_acc_en=True")


def get_compute_kernel_options(compute_kernel_options):
    if compute_kernel_options is None:
        return None
    if is_wormhole_b0():
        fp32_dest_acc_en = compute_kernel_options
        packer_l1_acc = False
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
        )
    return compute_kernel_config


def to_cpu(npu_tensor, shape, *, cpu_layout=ttnn.ROW_MAJOR_LAYOUT):
    if npu_tensor is None:
        return None

    shape = list(shape)

    unpad_shape = copy.copy(shape)

    if shape == []:
        unpad_shape = [1, 1]

    if len(shape) == 1:
        unpad_shape = [1] + shape

    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(unpad_shape).to_torch().reshape(shape)

    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttnn.TILE_LAYOUT,
    npu_dtype=ttnn.bfloat16,
    shape=None,
):
    if cpu_tensor is None:
        return None

    if shape is not None:
        cpu_tensor = cpu_tensor.view(shape)

    if len(cpu_tensor.shape) == 1:
        cpu_tensor = cpu_tensor.reshape([1, len(cpu_tensor)])

    if len(cpu_tensor.shape) == 0:
        cpu_tensor = cpu_tensor.reshape([1, 1])

    npu_tensor = ttnn.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor


# For keepdim in torch
def filter_indices(output_shape, dims):
    def not_in_dims(index_value_pair):
        index, value = index_value_pair
        return index not in dims

    filtered_elements = list(filter(not_in_dims, enumerate(output_shape)))
    filtered_values = [value for index, value in filtered_elements]

    return filtered_values


# For keepdim in tt
def filter_indices_with_last_two(output_shape, dims):
    last_two_elements = output_shape[-2:]
    remaining_elements = output_shape[:-2]

    def not_in_dims(index_value_pair):
        index, _ = index_value_pair
        return index not in dims

    filtered_remaining_elements = list(filter(not_in_dims, enumerate(remaining_elements)))
    filtered_remaining_values = [value for index, value in filtered_remaining_elements]
    final_output_shape = filtered_remaining_values + last_two_elements

    return final_output_shape


def compute_output_shape(input_shape, dim, keepdim=False):
    if dim is None or dim == []:
        dim = list(range(len(input_shape)))

    if isinstance(dim, int):
        dim = [dim]

    output_shape = list(input_shape)

    for d in dim:
        output_shape[d] = 1

    if keepdim:
        torch_output_shape = output_shape.copy()
        tt_output_shape = output_shape.copy()
    else:
        torch_output_shape = filter_indices(output_shape, dim)
        tt_output_shape = filter_indices_with_last_two(output_shape, dim)

    return torch_output_shape, tt_output_shape


def check_dim(input_shape, dim, keepdim):
    if type(dim) == int and dim >= len(input_shape):
        pytest.skip("dim bigger than input rank")

    if type(dim) == list:
        for i in dim:
            if i >= len(input_shape):
                pytest.skip("dim bigger than input rank")

    if keepdim == False:
        if dim in [None, []]:
            pytest.skip("`keepdim == false` don't support last 2-dim")

        if type(dim) == int and len(input_shape) - 2 <= dim:
            pytest.skip("`keepdim == false` don't support last 2-dim")

        if type(dim) == list:
            for i in dim:
                if len(input_shape) - 2 <= i:
                    pytest.skip("`keepdim == false` don't support last 2-dim")
