# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
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
def test_permute_on_4D_tensor_with_smaller_tuple_size(device, h, w, dtype, expect_error):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    with expect_error(
        RuntimeError,
        "The number of dimensions in the tensor input does not match the length of the desired ordering",
    ):
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
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
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


@pytest.mark.parametrize(
    "shape, perm",
    [
        ((1, 1, 32, 64), (0, 1, 2, 3)),
        ((1, 1, 32, 64), (1, 0, 2, 3)),
        ((1, 2, 3, 32, 64), (1, 2, 0, 3, 4)),
    ],
)
@pytest.mark.parametrize("implementation", ["auto", "codegen"])
def test_permute_layout_noop_bypasses_reshape(device, shape, perm, implementation):
    """These shapes/perms all hit is_permute_layout_nop() in permute.cpp, which returns a
    zero-copy ttnn.reshape() before the "auto"/"codegen" selector is ever consulted. This test
    does NOT exercise the codegen kernel path (RowInvariant/BlockedGeneric) at all -- it only
    confirms the reshape-bypass shortcut is a true no-copy no-op regardless of which
    `implementation` was requested. See test_permute_codegen_* below for actual codegen coverage.
    """
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    cache_entries_before = device.num_program_cache_entries()

    output_tensor = ttnn.permute(input_tensor, perm, implementation=implementation)

    assert output_tensor.buffer_address() == input_tensor.buffer_address()
    assert device.num_program_cache_entries() == cache_entries_before
    assert_equal(torch.permute(torch_tensor, perm), ttnn.to_torch(output_tensor))


def test_permute_codegen_rejects_unsupported_input(device, expect_error):
    """Finding #3: implementation="codegen" on an input rejected by supported_by_codegen()
    (permute_codegen_supported.cpp) must raise rather than silently falling back to native.
    TILE layout is entirely out of scope for this port -- supported_by_codegen() rejects it
    unconditionally -- and (0, 1, 3, 2) on a 4D tensor is not a layout/reshape no-op, so this
    reaches the codegen TT_FATAL rather than the reshape-bypass shortcut above.
    """
    torch_tensor = torch.rand((1, 1, 32, 64), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(
        RuntimeError,
        'ttnn::permute: implementation="codegen" requested but this input is not supported by the '
        "codegen implementation",
    ):
        ttnn.permute(input_tensor, (0, 1, 3, 2), implementation="codegen")


def test_permute_codegen_program_cache(device):
    """Finding #4: program-cache regression coverage for implementation="codegen". An identical
    (shape, dims, dtype) config must cache-hit on the second call (no growth), and a differing
    config that switches from the RowInvariant factory (dims[-1] unchanged) to the BlockedGeneric
    factory (W-changing) must add its own new entry rather than colliding with the first.
    """
    torch.manual_seed(2005)

    def run_codegen(shape, dims, dtype):
        torch_tensor = random_torch_tensor(dtype, shape)
        input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
        output_tensor = ttnn.permute(input_tensor, dims, implementation="codegen")
        return ttnn.to_torch(output_tensor), torch.permute(torch_tensor, dims)

    # Row-invariant: dims[-1] == rank - 1 -> RowInvariant::create_descriptor.
    row_invariant_shape, row_invariant_dims = (2, 3, 32, 64), (1, 0, 2, 3)
    # W-changing: dims[-1] != rank - 1 (and not the fused-WH-transpose case) -> BlockedGeneric.
    w_changing_shape, w_changing_dims = (2, 3, 64, 96), (2, 0, 3, 1)

    entries_before = device.num_program_cache_entries()

    out_a, ref_a = run_codegen(row_invariant_shape, row_invariant_dims, ttnn.bfloat16)
    entries_after_first = device.num_program_cache_entries()
    assert entries_after_first == entries_before + 1, "first codegen call must add exactly one cache entry"
    assert_equal(ref_a, out_a)

    out_b, ref_b = run_codegen(row_invariant_shape, row_invariant_dims, ttnn.bfloat16)
    entries_after_repeat = device.num_program_cache_entries()
    assert entries_after_repeat == entries_after_first, "identical (shape, dims, dtype) must cache-hit"
    assert_equal(ref_b, out_b)

    out_c, ref_c = run_codegen(w_changing_shape, w_changing_dims, ttnn.bfloat16)
    entries_after_second_config = device.num_program_cache_entries()
    assert entries_after_second_config == entries_after_repeat + 1, (
        "a different (W-changing/BlockedGeneric) config must add a new cache entry, not collide "
        "with the RowInvariant entry cached above"
    )
    assert_equal(ref_c, out_c)


@pytest.mark.parametrize(
    "shape, dims",
    [
        pytest.param((2, 3, 32, 64), (1, 0, 2, 3), id="row_invariant"),
        pytest.param((2, 3, 64, 96), (2, 0, 3, 1), id="w_changing"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permute_codegen_correctness(device, shape, dims, dtype):
    """Finding #9 / Finding #15: explicit implementation="codegen" correctness across the
    RowInvariant (dims[-1] unchanged) and BlockedGeneric (W-changing) factories, for all 3
    codegen-supported dtypes (permute_codegen_supported.cpp's dtype check: BFLOAT16, FLOAT32,
    INT32). The w_changing/int32 case is Finding #15 (int32 through BlockedGeneric); it's kept
    here rather than duplicated since this parametrization already covers it explicitly.
    """
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, dims, implementation="codegen")
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape, dims, dtype",
    [
        # W not a multiple of 32: row-invariant, so aligned_stick_bytes/tt::align handles the
        # unaligned stick size rather than the 32x32-block path.
        pytest.param((3, 5, 17, 50), (1, 0, 2, 3), ttnn.bfloat16, id="row_invariant_unaligned_w"),
        # Single-tile-equivalent: 32x32 with nc == 1 < kFusedMinNc, so despite dims[-1] == rank - 2
        # (a WH transpose), fused_wh_ok() is false and this is accepted as BlockedGeneric.
        pytest.param((1, 1, 32, 32), (0, 1, 3, 2), ttnn.int32, id="single_tile_w_changing"),
    ],
)
def test_permute_codegen_shape_edge_cases(device, shape, dims, dtype):
    """Finding #9: an unaligned-W (W % 32 != 0) shape and a single-tile-equivalent small shape,
    both forced through implementation="codegen"."""
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, dims, implementation="codegen")
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "shape, dims, dtype",
    [
        pytest.param((64, 96), (1, 0), ttnn.bfloat16, id="demoted_2d"),
        pytest.param((2, 96, 128), (0, 2, 1), ttnn.float32, id="demoted_3d"),
        pytest.param((1, 2, 3, 64, 96), (1, 2, 0, 3, 4), ttnn.int32, id="demoted_5d"),
    ],
)
def test_permute_codegen_demoted_entries_still_correct(device, shape, dims, dtype):
    """Finding #12: these (shape, dims, dtype) tuples are taken directly from
    permute_codegen_supported.cpp's demoted_entries() perf-demotion ledger. is_demoted() is only
    consulted by the implementation="auto" branch in permute.cpp, never when codegen is forced --
    so forcing implementation="codegen" here proves these entries are still numerically correct
    on the codegen path, just not perf-preferred by "auto".
    """
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, dims, implementation="codegen")
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
    assert_equal(torch_output, output_tensor)


def test_permute_codegen_uneven_core_split(device):
    """Finding #17: BlockedGeneric's create_descriptor (permute_codegen_program_factory.cpp)
    splits num_blocks_total work units across the device's compute_with_storage_grid_size() via
    split_work_to_cores(), which only produces an uneven split (core_group_1 getting one more
    unit than core_group_2) when units_to_divide > num_cores available and doesn't divide evenly.
    shape=(total_cores + 1, 32, 17) with dims=(0, 2, 1) yields x_blocks == w_blocks == 1, so
    num_blocks_total == shape[0] == total_cores + 1: guaranteed non-divisible across the full
    grid on any hardware. dims=(0, 2, 1) also puts dims[-1] == rank - 2 (a WH transpose shape),
    but W=17 fails fused_wh_ok()'s 32-alignment check, so this still routes to BlockedGeneric
    rather than being rejected as the fused-WH-transpose case.
    """
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    shape = (total_cores + 1, 32, 17)
    dims = (0, 2, 1)
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(ttnn.bfloat16, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    output_tensor = ttnn.permute(input_tensor, dims, implementation="codegen")
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
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
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
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
        ttnn.uint32,
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
        ttnn.uint32,
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


@pytest.mark.parametrize("shape", [[1, 1, 32, 64], [2, 3, 32, 32], [1, 1, 64, 96], [1, 8, 96, 32]])
def test_transpose_wh_tiled_uint32(device, shape):
    # ttnn.transpose(-2,-1) on TILE_LAYOUT → prim::TransposeWH → transpose_wh_program_factory
    # compute/transpose_wh.cpp → transpose_tile() → MOVD2B dest_32b_lo=1
    # Unlike ttnn.permute({0,1,3,2}), ttnn.transpose dispatches to transpose_wh_program_factory
    # directly (not permute_tiled_program_factory), so a dedicated test is needed.
    torch.manual_seed(2005)
    t = random_torch_tensor(ttnn.uint32, shape)
    tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device)
    out = ttnn.transpose(tt, -2, -1)
    assert_equal(t.permute(0, 1, 3, 2), ttnn.to_torch(out))


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
