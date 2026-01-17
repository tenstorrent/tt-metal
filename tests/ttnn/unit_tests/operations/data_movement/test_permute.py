# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.common.utility_functions import is_blackhole


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_transpose(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_on_4D_tensor_with_smaller_tuple_size(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    with pytest.raises(
        RuntimeError,
        match="The number of dimensions in the tensor input does not match the length of the desired ordering",
    ) as exception:
        ttnn.permute(input_tensor, (0, 1, 2))


@pytest.mark.parametrize(
    "perm", [(0,), (0, 1), (1, 0), (0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_on_less_than_4D(device, perm, dtype):
    torch.manual_seed(2005)
    shape = tuple([32 * (value + 1) for value in perm])
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, perm)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("s", [8])
@pytest.mark.parametrize("h", [1500])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_for_specific_case(device, b, s, h, w, dtype):
    torch.manual_seed(2005)
    shape = (b, s, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


def test_add_after_permute(device):
    torch.manual_seed(2005)
    torch_a = torch.randn(2, 1280, 8, 8)
    torch_b = torch.randn(1, 1, 2, 1280)
    torch_b_permuted = torch.permute(torch_b, (2, 3, 0, 1))
    torch_output = torch_a + torch_b_permuted

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.permute(b, (2, 3, 0, 1))
    output = a + b
    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_negative_dim(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, -3, -1, -2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, -3, -1, -2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


def test_permute_bfloat8(device):
    torch.manual_seed(2005)
    input_a = torch.randn(1, 160, 32, 32)
    torch_output = torch.permute(input_a, (0, 2, 3, 1))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_output = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape", [(8, 2, 2, 3, 4), [1, 1370, 1, 3, 1280], [1, 197, 1, 3, 1024], [1, 197, 1, 3, 768], [1, 50, 1, 3, 1024]]
)
@pytest.mark.parametrize("perm", [(0, 3, 2, 1, 4), (3, 1, 2, 0, 4), (0, 3, 2, 1, 4), (1, 3, 2, 0, 4), (0, 3, 1, 2, 4)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d(device, shape, perm, dtype):
    torch.manual_seed(2005)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)
    assert_equal(torch_output, tt_output)


@pytest.mark.parametrize("pad_value", [float("-inf"), 0.0])
def test_permute_pad_value(device, pad_value):
    if pad_value != 0.0 and is_blackhole():
        pytest.skip("Blackhole reduce is needed for the full test to work")
    torch.manual_seed(2005)
    input_a = torch.randn((2, 11, 33, 17), dtype=torch.bfloat16)
    torch_output = torch.permute(input_a, (3, 2, 1, 0))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = ttnn.permute(tt_input, (3, 2, 1, 0), pad_value=pad_value)
    tt_output = ttnn.to_torch(tt_output)
    assert_equal(torch_output, tt_output)


def generate_permutations(N):
    """
    Generator function that yields all permutations of tuples with values 0 to N-1.

    :param N: The number defining the range of values (0 to N-1).
    :yield: Tuples representing each permutation.
    """
    for perm in itertools.permutations(range(N)):
        yield perm


@pytest.mark.parametrize("shape", [(7, 7, 7, 7, 7)])
@pytest.mark.parametrize("perm", generate_permutations(5))
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permute_5d_width(device, shape, perm, memory_config, dtype):
    torch.manual_seed(2005)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(
        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
    )

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, tt_output, 0.9999)
    else:
        assert_equal(torch_output, tt_output)


@pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65), (1, 6, 256, 20, 50), (6, 20, 50, 1, 256)])
@pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1), (1, 3, 4, 0, 2), (3, 0, 4, 1, 2)])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.int32,
    ],
)
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(
        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
    )

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)

    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, tt_output, 0.9999)
    else:
        assert_equal(torch_output, tt_output)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_nd(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 3, 16, 16, 16, 16)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 2, 4, 3, 5, 1))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 2, 4, 3, 5, 1))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_squeeze(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, 3)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    assert_equal(output_tensor, ttnn.to_torch(input_tensor))


@pytest.mark.parametrize("shape", [(1, 49, 768)])
@pytest.mark.parametrize("perm", generate_permutations(3))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.int32,
    ],
)
def test_permute_3D(device, shape, perm, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=memory_config)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_nil_volume_permute(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 0, 30, 32)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 3, 2))
    assert torch_output.shape == output_tensor.shape


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d_tiled_basic(device, dtype):
    torch.manual_seed(2005)
    shape = (10, 10, 10, 100, 100)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (2, 1, 0, 3, 4))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (2, 1, 0, 3, 4))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d_tiled_swap(device, dtype):
    torch.manual_seed(2005)
    shape = (10, 10, 10, 100, 100)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (2, 1, 0, 4, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (2, 1, 0, 4, 3))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_cn(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (1, 0, 2, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (1, 0, 2, 3))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_wh(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 3, 2))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_4d_cnwh(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (1, 0, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (1, 0, 3, 2))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[2, 2, 2, 2, 2, 2, 32, 32]])
@pytest.mark.parametrize("dims", [(5, 4, 3, 2, 1, 0, 7, 6), (5, 4, 3, 2, 1, 0, 6, 7)])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_8d_swapped(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_identity(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 2, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 2, 3))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[2, 2, 67, 67, 65]])
@pytest.mark.parametrize("perm", [(0, 1, 3, 2, 4)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permute_5d_xh_pad(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


def generate_fixed_w_permutations(N):
    perms_Nd = generate_permutations(N - 1)
    for perm in perms_Nd:
        yield perm + (N - 1,)


@pytest.mark.parametrize("shape", [[7, 7, 7, 33, 33]])
@pytest.mark.parametrize("perm", generate_fixed_w_permutations(5))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permutations_5d_fixed_w(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[1, 9, 91, 7, 9]])
@pytest.mark.parametrize("perm", [[0, 3, 4, 1, 2]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_adversarial(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 32, 32], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("perm", generate_fixed_w_permutations(4))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_fixed_w(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert_equal(torch_output, output_tensor)


def generate_fixed_no_dim0_dim1_transpose_permutations(N, dim0, dim1):
    perms_Nd = generate_permutations(N)
    for perm in perms_Nd:
        if perm[dim0] != dim1:
            yield perm


@pytest.mark.parametrize("shape", [[7, 7, 7, 17, 17]])
@pytest.mark.parametrize("perm", [[0, 1, 4, 3, 2]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("pad_value", [35.0, float("-inf"), 0.0])
def test_permute_5d_yw_padded(device, shape, perm, dtype, pad_value):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    ttnn_output = ttnn.permute(input_tensor, perm, pad_value=pad_value)
    output_tensor = ttnn.to_torch(ttnn_output)
    torch_output = torch.permute(torch_tensor, perm)

    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)

    if pad_value != 0.0:
        logical_shape = torch_output.shape
        output_padded = ttnn.from_device(ttnn_output).to_torch()
        padded_shape = output_padded.shape
        num_padded_values = torch.prod(torch.tensor(padded_shape)) - torch.prod(torch.tensor(logical_shape))
        assert torch.sum(output_padded == pad_value) == num_padded_values


@pytest.mark.parametrize("shape", [[33, 1, 17, 33, 33]])
@pytest.mark.parametrize("perm", generate_fixed_no_dim0_dim1_transpose_permutations(5, 4, 3))
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.int32,
    ],
)
def test_permute_5d_yw_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[0, 3, 2, 1], [3, 1, 2, 0], [1, 3, 2, 0], [3, 0, 2, 1]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_4d_yw_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[2, 3, 0, 1], [3, 2, 1, 0], [2, 3, 1, 0], [3, 2, 0, 1]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_4d_whyx_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 3, 1, 2], [1, 2, 3, 0], [2, 1, 3, 0], [2, 0, 3, 1]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.int32,
    ],
)
def test_permute_4d_other_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[33, 1, 17, 33, 33]])
@pytest.mark.parametrize("perm", [[0, 1, 4, 2, 3], [0, 4, 1, 2, 3], [2, 4, 1, 0, 3], [4, 2, 1, 0, 3]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.int32,
    ],
)
def test_permute_5d_wyh(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm, pad_value=0.0)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    if dtype == ttnn.float32:
        # float32 permute internally truncates to tf32 at the moment
        # https://github.com/tenstorrent/tt-metal/issues/23663
        assert_with_pcc(torch_output, output_tensor, 0.9999)
    else:
        assert_equal(torch_output, output_tensor)


# TODO: Fix sharded permute bugs for width and block shard strategies
@pytest.mark.parametrize("shape", [[16, 8, 224, 224]])
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 3, 2, 1], [1, 2, 3, 0], [1, 3, 2, 0]])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_sharding", [None, ttnn.ShardStrategy.HEIGHT])
@pytest.mark.parametrize("output_sharding", [ttnn.ShardStrategy.HEIGHT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_permute_sharded(device, shape, perm, dtype, layout, input_sharding, output_sharding):
    torch.manual_seed(2005)
    if input_sharding is None and output_sharding is None:
        pytest.skip("both sharding strategies are None")

    output_shape = [shape[dim] for dim in perm]
    if input_sharding:
        input_shard_memory_config = ttnn.create_sharded_memory_config(
            shape, core_grid=ttnn.CoreGrid(x=8, y=8), strategy=input_sharding
        )
    else:
        input_shard_memory_config = ttnn.DRAM_MEMORY_CONFIG

    if output_sharding:
        output_shard_memory_config = ttnn.create_sharded_memory_config(
            output_shape, core_grid=ttnn.CoreGrid(x=8, y=8), strategy=output_sharding
        )
    else:
        output_shard_memory_config = ttnn.DRAM_MEMORY_CONFIG

    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(
        torch_tensor, layout=layout, dtype=dtype, device=device, memory_config=input_shard_memory_config
    )
    output_tensor = ttnn.permute(input_tensor, perm, memory_config=output_shard_memory_config)

    output_tensor = ttnn.from_device(output_tensor).to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    torch_output = torch.permute(torch_tensor, perm)

    assert_equal(torch_output, output_tensor)
