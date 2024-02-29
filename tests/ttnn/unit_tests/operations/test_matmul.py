# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


# fmt: off
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
    assert_with_pcc(torch_output_tensor, output, 0.99987)


# fmt: off
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
    assert_with_pcc(torch_output_tensor, output, 0.999649)


# fmt: off
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
    assert_with_pcc(torch_output_tensor, output, 0.999877)


# fmt: off
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


def test_tutorial_matmul(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 1024

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


def test_tutorial_matmul_inputs_and_output_in_l1_memory(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 1024

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


def test_tutorial_matmul_with_inputs_and_output_in_l1_memory_and_user_specified_core_grid(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 1024

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
        input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8)
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


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
                orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
            ),
            None,
        ),  # mcast 2d transposed
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
        ),  # mcast input_a
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
        ),  # mcast input_a
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
    ids=["mcast_2d", "mcast_2d_transposed", "mcast_in0", "mcast_in1", "bmm"],
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


@pytest.mark.parametrize("batch_size", [1, 8])
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

    if batch_size == 1:
        with pytest.raises(RuntimeError) as exception:
            output_tensor = ttnn.matmul(
                input_tensor_a,
                input_tensor_b,
                core_grid=ttnn.CoreGrid(y=batch_size, x=12),
            )
        assert "ttnn.matmul: ttl.operations.primary.matmul failed" in str(exception.value)
    else:
        output_tensor = ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=12),
        )

        output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [30, 61])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [1021, 2048])
def test_wide_matmul_with_argument_for_using_1D_systolic_array_set_to_true(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        use_1d_systolic_array=True,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [1024, 2048])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [32, 61])
def test_tall_matmul_with_argument_for_using_1D_systolic_array_set_to_true(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        use_1d_systolic_array=True,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


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

    program_config = ttnn.create_matmul_1d_systolic_array_program_config(
        input_shape_a=input_tensor_a.shape.with_tile_padding(),
        input_shape_b=input_tensor_b.shape.with_tile_padding(),
        core_grid=input_tensor_a.device.core_grid,
    )

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        program_config=program_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [4544])
@pytest.mark.parametrize("n_size", [4672])
@pytest.mark.parametrize("core_grid", [None, ttnn.CoreGrid(8, 8)])
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
        use_1d_systolic_array=True,
        core_grid=core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.996)
