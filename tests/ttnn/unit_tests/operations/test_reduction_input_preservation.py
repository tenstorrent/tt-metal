# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40), (2, 32, 128)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_preserves_input_bits(device, shape, operation):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    # Negative zero, a subnormal, and a NaN payload distinguish a raw copy from
    # a floating-point unpack/pack round trip through fill_pad.
    values.view(torch.int16).flatten()[:3] = torch.tensor([-32768, 1, 0x7FC1], dtype=torch.int16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before = ttnn.to_torch(tensor).view(torch.int16).clone()
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1)
    ttnn.to_torch(result)
    after = ttnn.to_torch(tensor).view(torch.int16)
    assert torch.equal(after, before)


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40), (2, 32, 128)])
def test_prod_preserves_input_bits(device, shape):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    values.view(torch.int16).flatten()[:3] = torch.tensor([-32768, 1, 0x7FC1], dtype=torch.int16)

    # Control: sum leaves input bits untouched
    tensor_sum = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sum = ttnn.to_torch(tensor_sum).view(torch.int16).clone()
    _ = ttnn.sum(tensor_sum, dim=-1)
    assert torch.equal(ttnn.to_torch(tensor_sum).view(torch.int16), before_sum)

    # Prod test: preserves input bits on non-tile-aligned inputs
    tensor_prod = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_prod = ttnn.to_torch(tensor_prod).view(torch.int16).clone()
    _ = ttnn.prod(tensor_prod, dim=-1)
    assert torch.equal(ttnn.to_torch(tensor_prod).view(torch.int16), before_prod)


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_with_private_padding_matches_reference(device, shape, operation):
    torch.manual_seed(0)
    values = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
        expected = torch.topk(values, k=8, dim=-1, largest=False).values
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1, keepdim=True)
        expected = getattr(torch, operation)(values.float(), dim=-1, keepdim=True)
        if operation in ("max", "min"):
            expected = expected.values
    torch.testing.assert_close(ttnn.to_torch(result).float(), expected.float(), rtol=0.02, atol=0.01)


# Two ranges away from core (0, 0), like a sub-device partition. 8 cores give an uneven work split.
SUB_CORE_GRIDS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 3)),
        ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 1)),
    ]
)


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40)])
@pytest.mark.parametrize("operation", ["max", "min", "sum", "mean"])
def test_reduction_on_sub_core_grids_preserves_input(device, shape, operation):
    torch.manual_seed(0)
    values = torch.rand(shape, dtype=torch.bfloat16)
    # Negative zero and a subnormal change if fill_pad round-trips the caller's border tiles.
    values.view(torch.int16).flatten()[:2] = torch.tensor([-32768, 1], dtype=torch.int16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before = ttnn.to_torch(tensor).view(torch.int16).clone()
    result = getattr(ttnn, operation)(tensor, dim=-1, keepdim=True, sub_core_grids=SUB_CORE_GRIDS)
    expected = getattr(torch, operation)(values.float(), dim=-1, keepdim=True)
    if operation in ("max", "min"):
        expected = expected.values
    torch.testing.assert_close(ttnn.to_torch(result).float(), expected.float(), rtol=0.02, atol=0.01)
    assert torch.equal(ttnn.to_torch(tensor).view(torch.int16), before)


# Bit patterns that a raw copy preserves but a float unpack/pack round trip does not.
TRAP_BITS = [-32768, 1, 0x7FC1]
# Last two dims unaligned so every case carries implicit tile padding to fill.
PRIVATE_PADDING_SHAPES = [(2, 7, 33), (1, 33, 40), (2, 33, 40)]


def _read_padded_bits(tensor):
    """The full padded buffer, read as raw bf16 bits."""
    return ttnn.to_torch(tensor).view(torch.int16).clone()


@pytest.mark.parametrize("shape", PRIVATE_PADDING_SHAPES)
def test_sort_preserves_input_bits(device, shape):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    values.view(torch.int16).flatten()[:3] = torch.tensor(TRAP_BITS, dtype=torch.int16)

    # Control: sum leaves input bits untouched.
    tensor_sum = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sum = _read_padded_bits(tensor_sum)
    _ = ttnn.sum(tensor_sum, dim=-1)
    assert torch.equal(_read_padded_bits(tensor_sum), before_sum)

    # Sort must not rewrite its caller's tensor while filling the implicit tile padding.
    tensor_sort = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sort = _read_padded_bits(tensor_sort)
    ttnn.sort(tensor_sort, dim=-1)
    assert torch.equal(_read_padded_bits(tensor_sort), before_sort)


@pytest.mark.parametrize("shape", PRIVATE_PADDING_SHAPES)
def test_pad_preserves_input_bits(device, shape):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    values.view(torch.int16).flatten()[:3] = torch.tensor(TRAP_BITS, dtype=torch.int16)

    # Control: sum leaves input bits untouched.
    tensor_sum = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sum = _read_padded_bits(tensor_sum)
    _ = ttnn.sum(tensor_sum, dim=-1)
    assert torch.equal(_read_padded_bits(tensor_sum), before_sum)

    # Pad the last dim by one element so the tile path runs and the implicit padding is filled.
    padding = [(0, 0)] * len(shape)
    padding[-1] = (0, 1)
    tensor_pad = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_pad = _read_padded_bits(tensor_pad)
    ttnn.pad(tensor_pad, padding=padding, value=0.0)
    assert torch.equal(_read_padded_bits(tensor_pad), before_pad)


@pytest.mark.parametrize("shape", PRIVATE_PADDING_SHAPES)
def test_gather_preserves_input_bits(device, shape):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    values.view(torch.int16).flatten()[:3] = torch.tensor(TRAP_BITS, dtype=torch.int16)
    # The index keeps the input's rank and leading dims; only the gather dim shrinks.
    # tests/ttnn/unit_tests/operations/data_movement/test_gather.py builds it this way, and
    # gather.cpp:322 requires input and index ranks to match.
    index_shape = list(shape)
    index_shape[-1] = min(32, shape[-1])
    index = torch.randint(0, shape[-1], index_shape, dtype=torch.int64)

    # Control: sum leaves input bits untouched.
    tensor_sum = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sum = _read_padded_bits(tensor_sum)
    _ = ttnn.sum(tensor_sum, dim=-1)
    assert torch.equal(_read_padded_bits(tensor_sum), before_sum)

    # Gather must not rewrite its caller's input tensor while filling the implicit padding.
    tensor_gather = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_gather = _read_padded_bits(tensor_gather)
    index_tensor = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    _ = ttnn.gather(tensor_gather, dim=-1, index=index_tensor)
    assert torch.equal(_read_padded_bits(tensor_gather), before_gather)


@pytest.mark.parametrize("shape", [(1, 1, 7, 33), (1, 1, 33, 40), (1, 2, 33, 40)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_pre_all_gather_preserves_input_bits(device, shape, with_residual):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    values.view(torch.int16).flatten()[:3] = torch.tensor(TRAP_BITS, dtype=torch.int16)
    residual = torch.randn(shape, dtype=torch.bfloat16) if with_residual else None

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Control: sum leaves input bits untouched.
    tensor_sum = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_sum = _read_padded_bits(tensor_sum)
    _ = ttnn.sum(tensor_sum, dim=-1)
    assert torch.equal(_read_padded_bits(tensor_sum), before_sum)

    tensor_input = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before_input = _read_padded_bits(tensor_input)
    residual_tensor = ttnn.from_torch(residual, device=device, layout=ttnn.TILE_LAYOUT) if with_residual else None
    before_residual = _read_padded_bits(residual_tensor) if with_residual else None
    ttnn.layer_norm_pre_all_gather(
        tensor_input,
        residual_input_tensor=residual_tensor,
        dtype=ttnn.bfloat16,
        compute_kernel_config=kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert torch.equal(_read_padded_bits(tensor_input), before_input)
    if with_residual:
        assert torch.equal(_read_padded_bits(residual_tensor), before_residual)
