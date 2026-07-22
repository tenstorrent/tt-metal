# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from math import prod

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

layouts = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]

dtypes = [
    (torch.float32, ttnn.float32),
    (torch.bfloat16, ttnn.bfloat16),
    (torch.bfloat16, ttnn.bfloat8_b),
    (torch.int32, ttnn.int32),
    (torch.int32, ttnn.uint32),
    (torch.int16, ttnn.uint16),
]
shapes = [(1,), (2,), (2, 3), (4, 16, 3, 1), (4, 3, 1, 2, 2)]
repeat_shapes = [
    (1,),
    (1, 2),
    (4, 3, 2, 1),
    (2, 3, 4, 5, 2),
    (2048,),
]


def _get_size(larger, smaller) -> int:
    return prod([a * b for a, b in zip(((1,) * (len(larger) - len(smaller)) + smaller), larger)])


def _get_final_size(shape, reshape):
    if len(shape) > len(reshape):
        return _get_size(shape, reshape)
    else:
        return _get_size(reshape, shape)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_repeat(device, layout, dtype, shape, repeat_shape):
    torch_dtype, ttnn_dtype = dtype

    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")

    if layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal config")

    if layout == ttnn.TILE_LAYOUT and ttnn_dtype == ttnn.uint16:
        pytest.skip("UINT16 tensors cannot be tilized - only bfloat16/float32/int32/uint32 supported")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=torch_dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=ttnn_dtype)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)
    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    if ttnn_dtype == ttnn.bfloat8_b:
        assert_with_pcc(torch_result, output, 0.9999)
    else:
        assert_equal(torch_result, output)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_pc_repeat(device, layout, shape, repeat_shape):
    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")
    num_iters = 3
    input_tensors = []
    torch_results = []
    for i in range(num_iters):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch_tensor.repeat(repeat_shape))
        input_tensors.append(ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16))
    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = ttnn.repeat(input_tensors[i], ttnn.Shape(repeat_shape))
        output = ttnn.to_torch(output)
        assert (
            output.shape == torch_results[i].shape
        ), f"Output shape {output.shape} does not match torch shape {torch_results[i].shape}"

        assert_equal(torch_results[i], output)
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


# 17975 test cases


# --- Codegen-path coverage (implementation="codegen") ---
#
# ttnn.repeat now defaults to the native (device-1.0) path; the codegen prim is
# only reached via implementation="codegen"/"auto". These duplicate the
# correctness / program-cache checks above but force the codegen path so the
# nightly data_movement suite exercises it despite the native default.
#
# The codegen prim supports only a subset of cases (see
# repeat_codegen_supported.cpp): interleaved input/output (no sharding), rank
# 2-4, at least one repeated dim, and per-dim rules -- ROW_MAJOR rejects
# bfloat8_b and needs last-dim width >= 2; TILE requires the repeated H/W axis
# to be tile-aligned. Every shape below is hand-picked to satisfy that gate, so
# implementation="codegen" resolves instead of TT_FATAL-ing. random inputs (not
# arange) keep bf16 comparisons exact -- repeat copies values verbatim, so a
# lossless round-trip means assert_equal holds.
codegen_supported_cases = [
    # (shape, repeat_shape, layout) -- TILE
    ((1, 1, 32, 32), (2, 1, 1, 1), ttnn.TILE_LAYOUT),  # N (batch) repeat
    ((1, 1, 32, 32), (1, 3, 1, 1), ttnn.TILE_LAYOUT),  # C repeat
    ((1, 1, 32, 32), (1, 1, 2, 1), ttnn.TILE_LAYOUT),  # H repeat (tile-aligned)
    ((1, 1, 32, 64), (1, 1, 1, 2), ttnn.TILE_LAYOUT),  # W repeat (tile-aligned)
    ((32, 64), (2, 1), ttnn.TILE_LAYOUT),  # rank-2 TILE
    # ROW_MAJOR
    ((2, 3, 4, 8), (2, 1, 1, 1), ttnn.ROW_MAJOR_LAYOUT),  # N repeat (higher-dim)
    ((1, 2, 4, 8), (1, 1, 2, 1), ttnn.ROW_MAJOR_LAYOUT),  # H repeat (higher-dim)
    ((1, 1, 4, 8), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT),  # last-dim (within-stick), width >= 2
]

codegen_dtypes = [
    (torch.bfloat16, ttnn.bfloat16),
    (torch.float32, ttnn.float32),
]


@pytest.mark.parametrize("shape, repeat_shape, layout", codegen_supported_cases)
@pytest.mark.parametrize("dtype", codegen_dtypes)
def test_repeat_codegen(device, shape, repeat_shape, layout, dtype):
    torch_dtype, ttnn_dtype = dtype

    torch_input_tensor = torch.rand(shape, dtype=torch_dtype)
    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=ttnn_dtype)
    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape), implementation="codegen")
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"
    assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "shape, repeat_shape, layout",
    [
        ((1, 1, 32, 32), (2, 1, 1, 1), ttnn.TILE_LAYOUT),
        ((1, 1, 4, 8), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_pc_repeat_codegen(device, shape, repeat_shape, layout):
    num_iters = 3
    input_tensors = []
    torch_results = []
    for i in range(num_iters):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch_tensor.repeat(repeat_shape))
        input_tensors.append(ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16))
    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = ttnn.repeat(input_tensors[i], ttnn.Shape(repeat_shape), implementation="codegen")
        output = ttnn.to_torch(output)
        assert (
            output.shape == torch_results[i].shape
        ), f"Output shape {output.shape} does not match torch shape {torch_results[i].shape}"

        assert_equal(torch_results[i], output)
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


# Single tiny codegen case wired into the merge gate (ttnn_merge_gate_tests.yaml).
# Kept as its own function (not a parametrization of test_repeat_codegen) so the
# merge-gate cmd can pin a stable node id. One 1-tile bf16 TILE N-repeat: the
# smallest call that still exercises the codegen device op end to end.
def test_repeat_codegen_smoke(device):
    shape = (1, 1, 32, 32)
    repeat_shape = (2, 1, 1, 1)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape), implementation="codegen")
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"
    assert_equal(torch_result, output)


def test_pc_with_different_shapes_in_sequence(device):
    y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)
    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x = torch.zeros((64, 1, 256, 384), dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    num_iters = 4
    z_tt = x_tt + y_tt

    y = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x = torch.zeros((4, 1, 32, 32), dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # codegen-supported (interleaved, tile-aligned N-axis repeat) -> exercise the codegen path
    ttnn.repeat(y_tt, [4, 1, 1, 1], implementation="codegen")
    z_tt = ttnn.add(x_tt, y_tt)
    z_tt = x_tt + y_tt

    for i in range(num_iters):
        z_torch = ttnn.to_torch(z_tt[i : i + 1])
        assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"
    for _ in range(num_iters):
        y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        x = torch.zeros((4, 1, 32, 32), dtype=torch.bfloat16)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        base_count = device.cache_entries_counter.total
        with device.cache_entries_counter.measure():
            ttnn.repeat(y_tt, [4, 1, 1, 1], implementation="codegen")
        assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"
        z_tt = ttnn.add(x_tt, y_tt)
        z_tt = x_tt + y_tt

        for i in range(num_iters):
            z_torch = ttnn.to_torch(z_tt[i : i + 1])
            assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"
    y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)

    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.repeat(y_tt, ttnn.Shape([64, 1, 1, 1]))
    for i in range(64):
        z_torch = ttnn.to_torch(z_tt[i : i + 1])
        assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"

    for _ in range(num_iters):
        y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)
        y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        base_count = device.cache_entries_counter.total
        with device.cache_entries_counter.measure():
            z_tt = ttnn.repeat(y_tt, ttnn.Shape([64, 1, 1, 1]))

        assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"

        for i in range(64):
            z_torch = ttnn.to_torch(z_tt[i : i + 1])
            assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"
