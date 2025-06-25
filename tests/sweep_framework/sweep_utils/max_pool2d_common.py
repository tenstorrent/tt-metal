# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    [pad_h, pad_w] = test_vector["padding"]
    [_, _, kernel_h, kernel_w] = test_vector["shape"]
    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        return True, "double of padding can not be greater than kernel size."
    return False, None


def mesh_device_fixture():
    num_devices = ttnn.GetNumPCIeDevices()
    # As of now take device id as 0.
    device_id = 0
    assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
    device = ttnn.CreateDevice(device_id=device_id, l1_small_size=32768)
    ttnn.SetDefaultDevice(device)

    device_name = "Unknown"
    if ttnn.device.is_grayskull(device):
        device_name = "grayskull"
    elif ttnn.device.is_wormhole_b0(device):
        device_name = "wormhole_b0"
    yield device, device_name

    ttnn.close_device(device)


def run_max_pool2d(
    in_n,
    in_c,
    in_h,
    in_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    dtype,
    device,
    sharding=None,
    ceil_mode=False,
    memory_config=None,
    in_place=False,
):
    kernel_size = [kernel_h, kernel_w]
    stride = [stride_h, stride_h]
    padding = [pad_h, pad_w]
    dilation = [dilation_h, dilation_w]

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    act_shape = [in_n, in_c, in_h, in_w]
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    ttact_device = ttnn.to_device(ttact, device)
    start_time = start_measuring_time()
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        memory_config=memory_config,
        applied_shard_scheme=sharding,
        in_place_halo=in_place,
    )

    output_host = output.cpu()
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_host))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]
    e2e_perf = stop_measuring_time(start_time)

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)
    golden_pytorch = golden_pytorch[:, :in_c, :, :]

    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35
    ## test for equivalance
    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose, " Reference and output tensor are not close"
    if dtype == ttnn.bfloat16:
        assert isequal, " Reference and output tensor are not equal"

    # check pcc and return
    return [check_with_pcc(output_pytorch, golden_pytorch, pcc=0.998), e2e_perf]
