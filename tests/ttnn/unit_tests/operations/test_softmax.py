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
    [[100.0, 101.0], [100.0, 1000.0], [-100.0, -101.0], [-1000.0, -100.0], [-100, -108, -99, -100, -101, -98]],
)
def test_softmax_stable_neg_values(device, input_vector):
    torch.manual_seed(0)

    torch_input_tensor = torch.tensor([[[input_vector]]], dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1, numeric_stable=True)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


def run_softmax_stable_with_program_cache(device, batch_size, h, w, skip_scale_mask, math_approx):
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

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx,
            fp32_dest_acc_en=False,
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
def test_softmax_stable_with_program_cache(device, batch_size, h, w, skip_scale_mask, math_approx, use_program_cache):
    for _ in range(2):
        run_softmax_stable_with_program_cache(device, batch_size, h, w, skip_scale_mask, math_approx)
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


def run_softmax_sharded_stable(device, batch_size, num_heads, h, w, skip_scale_mask):
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
        subblock_w=6,
        block_h=h // 32,
        block_w=w // 32,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    if not skip_scale_mask:
        output_tensor = ttnn.scale_mask_softmax_in_place(
            input_tensor, scale, attention_mask_t, program_config=program_config, numeric_stable=True
        )
    else:
        output_tensor = ttnn.scale_mask_softmax_in_place(
            input_tensor, program_config=program_config, numeric_stable=True
        )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("h", [384])
@pytest.mark.parametrize("w", [384])
@pytest.mark.parametrize("skip_scale_mask", [True, False])
def test_softmax_sharded_stable_with_program_cache(
    device, batch_size, num_heads, h, w, skip_scale_mask, use_program_cache
):
    for _ in range(2):
        run_softmax_sharded_stable(device, batch_size, num_heads, h, w, skip_scale_mask)
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


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_softmax(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

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
