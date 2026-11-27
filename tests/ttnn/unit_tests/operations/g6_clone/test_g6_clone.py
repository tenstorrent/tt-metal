# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_clone, Phase 0 scope: float32, TILE_LAYOUT, DRAM interleaved.

The op's TOLERANCES are ("exact", None) for every dtype, so every check compares stored bit
patterns with torch.equal; no PCC or RMS limit applies.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_clone import g6_clone


def pytorch_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor.clone()


# float32 patterns a conversion or an arithmetic engine changes while randn values survive:
# -0.0, +inf, -inf, NaN with payload 1, smallest subnormal, largest negative subnormal, max finite.
F32_SPECIAL_BITS = [0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x00000001, 0x807FFFFF, 0x7F7FFFFF]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


def _signed32(pattern: int) -> int:
    return pattern - (1 << 32) if pattern >= (1 << 31) else pattern


def _random_bit_patterns(shape) -> torch.Tensor:
    bits = torch.randint(-(1 << 31), 1 << 31, tuple(shape), dtype=torch.int64).to(torch.int32)
    return bits.view(torch.float32)


def _with_special_bits(tensor: torch.Tensor) -> torch.Tensor:
    bits = _bits(tensor).reshape(-1).clone()
    pattern = torch.tensor([_signed32(p) for p in F32_SPECIAL_BITS], dtype=torch.int32)
    slots = bits[0::2].numel()
    bits[0::2] = pattern.repeat(-(-slots // pattern.numel()))[:slots]
    return bits.view(torch.float32).reshape(tensor.shape)


def _upload(torch_input: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    # preserve_nan_values builds the tile layout on the host, so NaN payloads and subnormals reach
    # the device unchanged; a device-side tilize is not guaranteed to keep them.
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        preserve_nan_values=True,
    )
    readback = ttnn.to_torch(ttnn_input)
    if not torch.equal(_bits(readback), _bits(torch_input)):
        pytest.fail("HARNESS PRECONDITION (not the op): the uploaded input does not hold the host bits")
    return ttnn_input


def _assert_clone(ttnn_output: ttnn.Tensor, ttnn_input: ttnn.Tensor, torch_input: torch.Tensor) -> None:
    assert isinstance(ttnn_output, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(ttnn_output)
    assert list(ttnn_output.shape) == list(torch_input.shape), f"shape {list(ttnn_output.shape)}"
    assert ttnn_output.dtype == ttnn.float32, f"dtype {ttnn_output.dtype}"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"layout {ttnn_output.layout}"
    out_mc, in_mc = ttnn_output.memory_config(), ttnn_input.memory_config()
    assert out_mc.buffer_type == in_mc.buffer_type == ttnn.BufferType.DRAM
    assert out_mc.memory_layout == in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert ttnn_output.buffer_address() != ttnn_input.buffer_address(), "output aliases the input buffer"

    got = _bits(ttnn.to_torch(ttnn_output))
    want = _bits(pytorch_reference(torch_input))
    if not torch.equal(got, want):
        differ = got != want
        first = tuple(int(i) for i in differ.nonzero()[0])
        pytest.fail(
            f"{int(differ.sum())} of {differ.numel()} logical elements differ; first at {first}: "
            f"got 0x{int(got[first]) & 0xFFFFFFFF:08X} expected 0x{int(want[first]) & 0xFFFFFFFF:08X}"
        )
    assert torch.equal(_bits(ttnn.to_torch(ttnn_input)), _bits(torch_input)), "the call modified the input"


# Production float32 TILE shapes of the contract.
SHAPES = [
    pytest.param((1, 1), id="single_tile_1x1_partial"),
    pytest.param((1, 12, 128, 128), id="multi_tile_1x12x128x128_grid48x4"),
    pytest.param((8, 12, 128, 128), id="multi_batch_8x12x128x128"),
    pytest.param((16, 12, 128, 128), id="multi_batch_16x12x128x128"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_g6_clone(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = _upload(torch_input, device)
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 12, 128, 128), id="1x12x128x128"),
    ],
)
def test_g6_clone_bit_patterns(device, shape):
    """Every float32 bit pattern survives: NaN payloads, signed zeros, infinities, subnormals."""
    torch.manual_seed(42)
    torch_input = _with_special_bits(_random_bit_patterns(shape))
    ttnn_input = _upload(torch_input, device)
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input)


def test_g6_clone_poisoned_padding(device):
    """The logical element of a partial tile must not depend on the input's tile padding."""
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1), dtype=torch.float32)
    padded_bits = torch.full((32, 32), _signed32(0x7FC00001), dtype=torch.int32)
    padded_bits[:1, :1] = _bits(torch_input)
    backing = _upload(padded_bits.view(torch.float32), device)
    # A view at the logical shape over the padded buffer: its tile padding holds NaN, not zeros.
    poisoned = ttnn.reshape(backing, ttnn.Shape([1, 1]), ttnn.Shape([32, 32]))
    if poisoned.buffer_address() != backing.buffer_address():
        pytest.fail("HARNESS PRECONDITION (not the op): the poisoned view must share the padded buffer")
    ttnn_output = g6_clone(poisoned)
    _assert_clone(ttnn_output, poisoned, torch_input)


def test_g6_clone_repeat_other_address(device):
    """A second call of the same shape, at other addresses with other values, copies its own input."""
    shape = (1, 12, 128, 128)
    torch.manual_seed(42)
    first_input = torch.randn(shape, dtype=torch.float32)
    second_input = torch.randn(shape, dtype=torch.float32)

    first_ttnn = _upload(first_input, device)
    first_output = g6_clone(first_ttnn)
    _assert_clone(first_output, first_ttnn, first_input)

    second_ttnn = _upload(second_input, device)
    second_output = g6_clone(second_ttnn)
    _assert_clone(second_output, second_ttnn, second_input)

    assert torch.equal(_bits(ttnn.to_torch(first_output)), _bits(first_input)), "second call wrote into first output"


def test_g6_clone_multi_block_ring(device, monkeypatch):
    """Depth-2 staging ring with a ragged last block, the plan the L1 cap selects for large tensors.

    No INPUTS shape reaches it (one block per core), so the knob forces it on p2: 14 / 13 pages per
    core in blocks of 5, 5, 4 / 5, 5, 3, so the ring wraps once and every push and pop stays at the
    nominal block_pages while the transfers narrow."""
    # The package re-exports the g6_clone function under its module's name, so the module comes
    # from sys.modules, not from attribute access.
    op_module = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    descriptor_module = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

    monkeypatch.setattr(descriptor_module, "BLOCKS_PER_CORE_KNOB", 3)
    # The knob is a process constant outside the cache key; isolate the cache so the forced plan is
    # built here and never served to another test.
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})

    shape = (8, 12, 128, 128)
    torch.manual_seed(42)
    torch_input = _with_special_bits(_random_bit_patterns(shape))
    ttnn_input = _upload(torch_input, device)
    plan = op_module.g6_clone_plan(ttnn_input)
    assert (plan["block_pages"], plan["cb_depth_blocks"], plan["num_blocks_max"]) == (5, 2, 3), plan
    assert (plan["core_pages_g1"], plan["core_pages_g2"]) == (14, 13), plan
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input)


def test_g6_clone_rejects_host_tensor(expect_error):
    host_tensor = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), "g6_clone: input_tensor must be a device tensor"):
        g6_clone(host_tensor)


def test_g6_clone_rejects_rank_above_4(device, expect_error):
    torch.manual_seed(42)
    rank5 = ttnn.from_torch(
        torch.randn((1, 1, 1, 32, 32), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), "g6_clone: input_tensor rank must be 1 to 4"):
        g6_clone(rank5)


def test_g6_clone_rejects_l1_input(device, expect_error):
    torch.manual_seed(42)
    in_l1 = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), "g6_clone: input_tensor must be DRAM interleaved"):
        g6_clone(in_l1)
