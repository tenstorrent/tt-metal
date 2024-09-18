# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, is_wormhole_b0, is_grayskull, is_blackhole


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [2])
@pytest.mark.parametrize("h", [71])
@pytest.mark.parametrize("w", [35])
@pytest.mark.parametrize("tile_h", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
def test_tiny_tiles(device, n, c, h, w, tile_h, tile_w):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        tile=(tile_h, tile_w),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_input_tensor, output_tensor, 1)


@pytest.mark.parametrize("b", [8])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
@pytest.mark.parametrize("in0_sharded", [True, False])
@pytest.mark.parametrize("in1_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_matmul_reuse_config_sharded_tiny_tile(
    device, b, h, m, k, n, tile_h, tile_w, in0_sharded, in1_sharded, out_sharded
):
    torch.manual_seed(0)

    grid_size = (b, h)

    in0 = torch.ones([b, h, m, k]).bfloat16().float()
    in1 = torch.randn([b, h, k, n]).bfloat16().float()

    if in0_sharded:
        in0_memory_config = ttnn.create_sharded_memory_config(
            (b, h, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_memory_config = ttnn.L1_MEMORY_CONFIG
    in0_t = ttnn.from_torch(
        in0,
        tile=(tile_h, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )

    if in1_sharded:
        in1_memory_config = ttnn.create_sharded_memory_config(
            (b, h, k, n),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in1_memory_config = ttnn.L1_MEMORY_CONFIG
    in1_t = ttnn.from_torch(
        in1,
        tile=(32, tile_w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    out_block_h = m // tile_h
    out_block_w = n // tile_w
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=k // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
    )
    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem_config = ttnn.L1_MEMORY_CONFIG
    output_t = ttnn.matmul(in0_t, in1_t, program_config=program_config, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    assert_with_pcc(pt_out, output_tensor, 0.999)


def pad_to_dram_banks(num, tile_w, lcm=32 * 12):
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


@pytest.mark.parametrize("k", [8192])
@pytest.mark.parametrize("n", [1280])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("grid_size", [(8, 1)])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
def test_matmul_in1_dram_sharded_tiny_tile(device, k, n, has_bias, grid_size, tile_h, tile_w):
    # PCC issue when height not equal to tile height
    m = tile_h
    if is_grayskull():
        n_padded = n
        num_banks = 8
    else:
        num_banks = 12
        n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)

    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]
    in1_shard_shape = [k, n_padded // num_banks]
    bias_shape = [1, 1, n]
    bias_shard_shape = [tile_h, n_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_w = k // num_cores // 32
    out_block_h = m // tile_h
    out_block_w = n // num_cores // tile_w

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        in0,
        tile=(tile_h, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=(32, tile_w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, tile_h - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = ttnn.from_torch(
            bias_padded,
            tile=(tile_h, tile_w),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_mem_config,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    assert_with_pcc(pt_out, output_tensor, 0.999)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("m_size,k_size,n_size", [
    (1, 2, 2),
    (1, 2, 4),
    (1, 4, 2),
    (1, 4, 4),
    (3, 2, 2),
    (3, 2, 4),
    (3, 4, 2),
    (3, 4, 4),
])
# fmt: on
def test_matmul_with_matched_width_height(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.99981)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("k_size, n_size", [
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    ])
# fmt: on
def test_matmul_with_matched_width_height_from_1D(device, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output, torch_rank=1)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.skip(reason="ttnn.reshape doesn't support reshaping the input tensors used in this test")
@pytest.mark.parametrize("w", [(4), (2)])
def test_matmul_does_dot_product(device, w):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_input_tensor_b = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.from_device(output)

    output = ttnn.to_torch(output)

    assert torch_output_tensor.shape == ()
    assert output.shape == (32,)
    assert torch.allclose(torch_output_tensor, output[0], atol=1e-2)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("n_size,c,h,w", [
    (1, 1, 2, 4),
    (1, 1, 4, 2),
    (3, 3, 2, 4),
    (3, 3, 4, 2),
    (1, 3, 2, 4),
    (3, 1, 4, 2),
    ])
# fmt: on
def test_matmul_with_matched_width_height_4D(device, n_size, c, h, w):
    torch_input_tensor_a = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, w, h), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.999599)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("n_size,c,h,w", [
    (1, 1, 2, 2),
    (1, 1, 4, 4),
    (3, 3, 4, 4),
    (3, 1, 4, 4),
    (1, 3, 4, 4)
    ])
# fmt: on
def test_matmul_same_shape_and_valid(device, n_size, c, h, w):
    torch_input_tensor_a = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9997)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("input_a,input_b", [
        ([1.0,2.0,3.0],[3.0,4.0,5.0])
    ])
# fmt: on
def test_matmul_same_shape_but_invalid(device, input_a, input_b):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    input_b += [0.0] * (32 - len(input_b))

    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_a)))
    torch_input_tensor_b = torch.as_tensor(input_b, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_b)))

    with pytest.raises(RuntimeError) as exception:
        torch.matmul(torch_input_tensor_a, torch_input_tensor_b)
    assert "Expected size for first two dimensions of batch2 tensor to be: [1, 32] but got: [1, 1]." in str(
        exception.value
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exception:
        ttnn.matmul(input_tensor_a, input_tensor_b)
    assert "The width of the first tensor must be equal to the height of the second tensor" in str(exception.value)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul_inputs_and_output_in_l1_memory(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul_with_inputs_and_output_in_l1_memory_and_user_specified_core_grid(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.matmul(
        input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=4, x=4)
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize(
    "batch_size_0, batch_size_1, m_size, k_size, n_size, bcast_batch, input_a_sharded_memory_config_args, input_b_sharded_memory_config_args",
    [
        (
            2,
            3,
            1600,
            224,
            896,
            True,
            dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        ),  # mcast 2d
        (
            2,
            3,
            1600,
            224,
            896,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=5),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            None,
        ),  # mcast 2d transposed
        (
            2,
            1,
            128,
            256,
            512,
            True,
            dict(core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        ),  # mcast 2d with shard width > 1 TILE
        (
            2,
            3,
            64,
            32 * 7,
            1024,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=1, x=7),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
            None,
        ),  # mcast in0
        (
            2,
            3,
            160 * 7,
            64,
            64,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            None,
        ),  # mcast in1
        (
            7,
            7,
            384,
            64,
            384,
            False,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
        ),  # bmm
    ],
    ids=["mcast_2d", "mcast_2d_transposed", "mcast_2d_shard_width_gt_1", "mcast_in0", "mcast_in1", "bmm"],
)
def test_sharded_matmul(
    device,
    batch_size_0,
    batch_size_1,
    m_size,
    k_size,
    n_size,
    bcast_batch,
    input_a_sharded_memory_config_args,
    input_b_sharded_memory_config_args,
):
    torch.manual_seed(0)

    input_a_shape = [batch_size_0, batch_size_1, m_size, k_size]
    if bcast_batch:
        input_b_shape = [k_size, n_size]
    else:
        input_b_shape = [batch_size_0, batch_size_1, k_size, n_size]

    torch_input_tensor_a = torch.randn(input_a_shape)
    torch_input_tensor_b = torch.randn(input_b_shape)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a.to(torch.bfloat16))
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b.to(torch.bfloat16))

    input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.TILE_LAYOUT)

    input_a_sharded_memory_config = ttnn.create_sharded_memory_config(
        input_a_shape, **input_a_sharded_memory_config_args
    )
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_sharded_memory_config)

    if input_b_sharded_memory_config_args:
        input_b_sharded_memory_config = ttnn.create_sharded_memory_config(
            input_b_shape, **input_b_sharded_memory_config_args
        )
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, input_b_sharded_memory_config)

    output = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 7])
def test_matmul_with_core_grid(device, batch_size):
    torch.manual_seed(0)

    m_size = 384
    k_size = 1024
    n_size = 1024

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=ttnn.CoreGrid(y=batch_size, x=8),
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [30, 61])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [1021, 2048])
def test_wide_matmul_with_argument_for_core_grid_set_to_device_grid(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [1024, 2048])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [32, 61])
def test_tall_matmul_with_argument_for_core_grid_set_to_device_grid(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [31, 63])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1023, 2047])
def test_matmul_by_passing_in_1D_systolic_array_program_config(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


@pytest.mark.parametrize(
    "n_size, c, m, k, n",
    [
        (1, 1, 2, 3, 4),
        (1, 1, 1024, 64, 512),
    ],
)
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("transpose_a", [True, False])
def test_matmul_with_transpose_a_or_b(device, n_size, c, m, k, n, transpose_a, transpose_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((n_size, c, m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, k, n), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    if transpose_a:
        torch_input_tensor_a = torch_input_tensor_a.transpose(-1, -2)
    if transpose_b:
        torch_input_tensor_b = torch_input_tensor_b.transpose(-1, -2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b, transpose_a=transpose_a, transpose_b=transpose_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.999)


##########################
# MODEL SPECIFIC MATMULS #
##########################
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [4544])
@pytest.mark.parametrize("n_size", [4672])
@pytest.mark.parametrize("core_grid", [None, ttnn.CoreGrid(y=7, x=8)])
def test_falcon_query_key_value_matmul(device, batch_size, m_size, k_size, n_size, core_grid):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.996)


# @skip_for_grayskull()
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize(
    "batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias",
    [
        (1, 2, 1, 1024, 640, 2560, False),
        (2, 8, 8, 64, 96, 160, False),
        (1, 2, 1, 4096, 320, 1280, False),
        (1, 2, 1, 64, 1280, 5120, False),
        (2, 8, 8, 64, 64, 160, False),
        (1, 2, 1, 1024, 640, 768, False),
        (2, 8, 8, 96, 160, 96, False),
        (2, 8, 8, 1024, 1024, 96, False),
        (1, 2, 1, 96, 768, 1024, False),
        (1, 1, 1, 32, 1280, 1280, True),
        (2, 8, 8, 4096, 96, 64, False),
        (1, 2, 1, 64, 5120, 1280, True),
        (2, 8, 8, 4096, 64, 96, False),
        (1, 2, 1, 1024, 768, 640, True),
        (1, 2, 1, 256, 1280, 1280, True),
        (2, 8, 8, 1024, 96, 96, False),
        (1, 2, 1, 1024, 640, 2304, False),
        (1, 1, 1, 32, 1280, 320, True),
        (1, 2, 1, 96, 768, 2560, False),
        (1, 2, 1, 4096, 1280, 320, True),
        (1, 2, 1, 1024, 2560, 640, True),
        (1, 2, 1, 256, 1280, 3840, False),
        (1, 1, 1, 32, 320, 1280, True),
        (1, 2, 1, 4096, 512, 320, True),
        (1, 2, 1, 64, 1280, 1280, True),
        (1, 2, 1, 256, 5120, 1280, True),
        (1, 2, 1, 256, 1280, 1280, False),
        (2, 8, 8, 256, 160, 96, False),
        (2, 8, 8, 256, 256, 160, False),
        (1, 2, 1, 96, 768, 1536, False),
        (1, 2, 1, 64, 1280, 3840, False),
        (2, 8, 8, 1024, 96, 1024, False),
        (2, 8, 8, 256, 96, 160, False),
        (1, 2, 1, 64, 1280, 1280, False),
        (2, 8, 8, 4096, 64, 4096, False),
        (1, 1, 1, 32, 1280, 640, True),
        (2, 8, 8, 64, 160, 64, False),
        (1, 2, 1, 4096, 320, 1536, False),
        (1, 2, 1, 256, 1280, 5120, False),
        (2, 8, 8, 4096, 4096, 64, False),
        (2, 8, 8, 256, 160, 256, False),
        (1, 2, 1, 4096, 320, 512, False),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_sd_matmul(device, batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")
    core_grid = ttnn.CoreGrid(x=8, y=8)
    TILE_HEIGHT = 32

    if batch_size == 2:
        if (m_size == 1024 and k_size == 96 and n_size == 1024) or (m_size == 4096 and k_size == 64 and n_size == 4096):
            # NOTE: matmul errors out with OOM otherwise
            core_grid = None

    # if batch_size == 2:
    #     if m_size == 1024 and k_size == 96 and n_size == 1024 and (dtype == ttnn.bfloat16 or is_grayskull()):
    #         pytest.skip("skip: Raises OOM")
    #     if m_size == 4096 and k_size == 64 and n_size == 4096:
    #         pytest.skip("skip: Raises OOM without decomposition")
    #     if is_grayskull():
    #         if m_size == 4096 and (
    #             (k_size == 96 and n_size == 64) or (k_size == 64 and n_size == 96) or (k_size == 4096 and n_size == 64)
    #         ):
    #             pytest.skip("skip: Raises OOM on GS")

    torch_input_tensor_a = torch.randn((batch_size, channel_a, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch_size, channel_b, k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if has_bias:
        torch_input_tensor_c = torch.randn((1, 1, TILE_HEIGHT, n_size), dtype=torch.bfloat16)
        _torch_input_tensor_c = torch.repeat_interleave(
            torch_input_tensor_c, torch_output_tensor.shape[2] // TILE_HEIGHT, dim=2
        )
        torch_output_tensor = torch_output_tensor + _torch_input_tensor_c

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_c = (
        ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype) if has_bias else None
    )
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    if has_bias:
        output_tensor = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=input_tensor_c,
            core_grid=core_grid,
        )
    else:
        output_tensor = ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            core_grid=core_grid,
        )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
