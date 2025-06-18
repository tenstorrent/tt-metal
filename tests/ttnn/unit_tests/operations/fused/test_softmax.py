# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, is_grayskull
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_vector",
    [
        # [100.0, 101.0],
        # [100.0, 1000.0],
        # [-100.0, -99.0],
        # [-100.0, -101.0],
        [-1000.0, -100.0],
        # [-100, -108, -99, -100, -101, -98],
    ],
)
@pytest.mark.parametrize("math_approx", [True])
@pytest.mark.parametrize("fp32_acc_en", [True])
def test_softmax_stable_neg_values(device, input_vector, math_approx, fp32_acc_en):
    torch.manual_seed(0)

    torch_input_tensor = torch.tensor([[[input_vector]]], dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
            fp32_dest_acc_en=fp32_acc_en,
            packer_l1_acc=False,
        )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1, compute_kernel_config=compute_kernel_config, numeric_stable=True)
    output_tensor = ttnn.to_torch(output_tensor)
    torch.set_printoptions(profile="full")
    with open("tensor_output.txt", "w") as f:
        print(output_tensor, file=f)
    with open("torch_tensor_output.txt", "w") as f:
        print(torch_output_tensor, file=f)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


def run_softmax_stable_with_program_cache(
    device, batch_size, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype
):
    torch.manual_seed(0)

    scale = 1.0
    attention_mask = torch.rand(batch_size, 1, 1, w)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, torch.tensor(float("-inf"), dtype=torch.bfloat16))
    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0)
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_input_tensor = torch_random((batch_size, 1, h, w), -1000, 1000, dtype=torch.bfloat16)
    if not skip_scale_mask:
        torch_output_tensor = torch_input_tensor * scale + attention_mask
    else:
        torch_output_tensor = torch_input_tensor
    torch_output_tensor = F.softmax(torch_output_tensor, dim=-1, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
            fp32_dest_acc_en=fp32_acc_en,
            packer_l1_acc=False,
        )

    if not skip_scale_mask:
        output_tensor = ttnn.scale_mask_softmax(
            input_tensor, scale, attention_mask_t, compute_kernel_config=compute_kernel_config, numeric_stable=True
        )
    else:
        output_tensor = ttnn.softmax(
            input_tensor, dim=-1, compute_kernel_config=compute_kernel_config, numeric_stable=True
        )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [32, 128])
@pytest.mark.parametrize("w", [1024, 1500])
@pytest.mark.parametrize("skip_scale_mask", [True, False])
@pytest.mark.parametrize("math_approx", [True, False])
@pytest.mark.parametrize("fp32_acc_en", [True, False])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_softmax_stable_with_program_cache(
    device, batch_size, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype, use_program_cache
):
    for _ in range(2):
        run_softmax_stable_with_program_cache(
            device, batch_size, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype
        )
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


def run_softmax_sharded_stable(
    device, batch_size, num_heads, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype
):
    torch.manual_seed(0)

    grid_size = (batch_size, num_heads)

    scale = 1.0
    attention_mask = torch.rand(batch_size, 1, 1, w)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, torch.tensor(float("-inf"), dtype=torch.bfloat16))
    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0)
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_input_tensor = torch_random((batch_size, num_heads, h, w), -1000, 1000, dtype=torch.bfloat16)
    if not skip_scale_mask:
        torch_output_tensor = torch_input_tensor * scale + attention_mask
    else:
        torch_output_tensor = torch_input_tensor
    torch_output_tensor = F.softmax(torch_output_tensor, dim=-1, dtype=torch.bfloat16)

    memory_config = ttnn.create_sharded_memory_config(
        torch_input_tensor.shape,
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=6 if not fp32_acc_en else 3,
        block_h=h // 32,
        block_w=w // 32,
    )
    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
            fp32_dest_acc_en=fp32_acc_en,
            packer_l1_acc=False,
        )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    if not skip_scale_mask:
        output_tensor = ttnn.scale_mask_softmax_in_place(
            input_tensor,
            scale,
            attention_mask_t,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            numeric_stable=True,
        )
    else:
        output_tensor = ttnn.scale_mask_softmax_in_place(
            input_tensor,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            numeric_stable=True,
        )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("h", [384])
@pytest.mark.parametrize("w", [384])
@pytest.mark.parametrize("skip_scale_mask", [True, False])
@pytest.mark.parametrize("math_approx", [True, False])
@pytest.mark.parametrize("fp32_acc_en", [True, False])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_softmax_sharded_stable_with_program_cache(
    device, batch_size, num_heads, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype, use_program_cache
):
    for _ in range(2):
        run_softmax_sharded_stable(
            device, batch_size, num_heads, h, w, skip_scale_mask, math_approx, fp32_acc_en, in_dtype
        )
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
def test_softmax(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    print(torch_input_tensor)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    print("hi before assert")
    for i in range(0, w):
        # print(i)
        # torch.set_printoptions(profile="full")
        # with open("tensor_output.txt", "w") as f:
        #     print(output_tensor[:, :, i], file=f)
        # with open("torch_tensor_output.txt", "w") as f:
        #     print(torch_output_tensor[:, :, i], file=f)
        assert_with_pcc(torch_output_tensor[:, :, i], output_tensor[:, :, i], 0.997)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, h, w, dim",
    [
        (1, 2048, 128000, -1),
        (1, 512, 128000, -1),
        (1, 128, 128000, -1),
        (1, 32, 128000, -1),
        (1, 2048, 32000, -1),
        (1, 512, 32000, -1),
        (1, 32, 32000, -1),  # base case
    ],
)
def test_large_softmax(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    print(torch_input_tensor)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=dim)
    print("hi")
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    print("hi")
    output_tensor = ttnn.from_device(output_tensor)
    print("hi")
    output_tensor = ttnn.to_torch(output_tensor)
    print("hi before assert")

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


def test_softmax_with_3D(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 1500, 1500), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


def test_softmax_with_padded_tile_layout(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 2, 2), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


def test_softmax_with_padded_tile_layout_large(device):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((8, 100, 1200), -10, 10, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.skip(reason="#4629: softmax pcc at 0.948 when comparing to torch")
def test_specific_tensor_combination(device):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_file = os.path.join(current_dir, "softmax_weights.pt")
    torch_input_tensor = torch.load(tensor_file)

    torch_output_tensor = torch.softmax(torch_input_tensor, -1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.softmax(input_tensor, -1)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 3, 56, 56, 3), -1),
        ((3, 5, 63, 56, 33), -2),
        ((7, 2, 56, 67, 31), -3),
        ((4, 9, 6, 86, 13), -4),
        ((32, 32, 32, 32, 32), -5),
    ],
)
def test_5d_softmax(device, input_shape, dim):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.softmax(torch_input_tensor, dim)
    print(torch_output_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.softmax(input_tensor, dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@pytest.mark.parametrize("input_shape", [(16, 7, 7)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("dlayout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("numeric_stable", [True])
@pytest.mark.parametrize(
    "fill_value",
    [
        -338953138925153547590470800371487866880.00000,  # 7E,7M
        # -337623910929368631717566993311207522304.00000,  # 7E,6M
        # -329648542954659136480144150949525454848.00000,  # 7E,4M
        # -297747071055821155530452781502797185024.00000,  # 7E,2M
        # -255211775190703847597530955573826158592.00000,  # 7E,1M
        # -170141183460469231731687303715884105728.00000,  # 7E,0M
        # -84738284731288386897617700092871966720.00000,  # 6E,7M
        # -42535295865117307932921825928971026432.00000,  # 6E,0M
    ],
)
def test_large_fill_softmax(device, input_shape, dtype, dlayout, dim, numeric_stable, fill_value):
    """
    Test softmax with specific fill values.
    This test is designed to check the stability of the softmax operation
    when using specific fill values that may cause overflow or underflow.
    Addresses bug #19781.
    """
    torch_input_tensor = torch.full(
        size=input_shape, fill_value=fill_value, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    )
    torch_output_tensor = torch.softmax(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=dlayout, device=device)
    output_tensor = ttnn.softmax(input_tensor, dim, numeric_stable=numeric_stable)
    output_tensor = ttnn.to_torch(output_tensor)

    assert len(output_tensor.shape) == len(torch_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
