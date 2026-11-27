# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_copy, Phase 0: float32, TILE, DRAM interleaved.

The grade is bit equality of every logical element: helpers.TOLERANCES[float32]
is ("exact", None). Cases are limited to the float32 production calls plus the
cases the contract's rules require (BB-22); op_design.md Appendix D maps each
test to its reason.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import describe_plan, g6_copy
from ttnn.operations.g6_copy import g6_copy_program_descriptor as plan_knobs

# Per-core L1 the model leaves to the op (BB-05).
L1_BYTES_LEFT_BY_MODEL = 1347776

PRODUCTION_SHAPES = [
    (1, 1),  # FLUX 35: one tile, H and W partial
    (1, 12, 128, 128),  # Qwen 187, 189
    (2, 12, 128, 128),  # Qwen 172
    (8, 12, 128, 128),  # Qwen 177, 179
    (16, 12, 128, 128),  # Qwen 168, 170: the perf-focus shape
]
# The only float32 h_non_aligned (and non-square) shape: SUPPORTED claims
# h_non_aligned from Phase 0, so it must run natively.
H_NON_ALIGNED_SHAPE = (1, 1, 1, 256)
PHASE0_SHAPES = PRODUCTION_SHAPES + [H_NON_ALIGNED_SHAPE]

# quiet NaN, NaN payloads (quiet, signalling, negative), +inf, -inf, -0.0, +0.0,
# smallest and largest denormal, a negative denormal, smallest normal, +-largest finite.
FLOAT32_SPECIAL_BITS = [
    0x7FC00000, 0x7FC00001, 0x7FA5A5A5, 0x7F800001, 0xFFC00000, 0xFFF00F0F,
    0x7F800000, 0xFF800000, 0x80000000, 0x00000000, 0x00000001, 0x007FFFFF,
    0x80000001, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,
]  # fmt: skip


def torch_g6_copy(input_tensor, output_tensor):
    """The destination after the call: output_tensor.copy_(input_tensor)."""
    assert input_tensor.shape == output_tensor.shape and input_tensor.dtype == output_tensor.dtype
    return output_tensor.clone().copy_(input_tensor)


def to_device(t, device, *, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    # preserve_nan_values keeps the conversion on the host, so every float32 bit reaches the device.
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=device, memory_config=memory_config, preserve_nan_values=True
    )


def bits(t):
    return t.contiguous().view(torch.int32)


def assert_bit_identical(actual, expected, what):
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    wrong = bits(actual) != bits(expected)
    if bool(wrong.any()):
        first = int(wrong.reshape(-1).nonzero()[0])
        got = int(bits(actual).reshape(-1)[first]) & 0xFFFFFFFF
        want = int(bits(expected).reshape(-1)[first]) & 0xFFFFFFFF
        pytest.fail(
            f"{what}: {int(wrong.sum())} of {wrong.numel()} logical elements differ; "
            f"first at flat index {first}: got 0x{got:08x} expected 0x{want:08x}"
        )


class CopyCase:
    """A source/destination pair on device, and the expected destination."""

    def __init__(self, device, src_torch, dst_torch, *, poison_padding=None):
        self.shape = tuple(src_torch.shape)
        self.src = to_device(src_torch, device)
        if poison_padding is not None:
            self.src = ttnn.fill_implicit_tile_padding(self.src, poison_padding, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.dst = to_device(dst_torch, device)
        self.dst_before = ttnn.to_torch(self.dst)
        # Graded against the source as the device holds it.
        self.src_before = ttnn.to_torch(self.src)
        self.expected = torch_g6_copy(self.src_before, dst_torch)


def call_and_check(case):
    src, dst = case.src, case.dst
    dst_address = dst.buffer_address()

    out = g6_copy(src, dst)

    assert out is dst, "g6_copy must return the output_tensor object it was given"
    assert out.buffer_address() == dst_address, "result is not in output_tensor's buffer"
    assert out.dtype == ttnn.float32
    assert out.layout == ttnn.TILE_LAYOUT
    assert tuple(out.shape) == case.shape
    memory = out.memory_config()
    assert memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory.buffer_type == ttnn.BufferType.DRAM

    assert_bit_identical(ttnn.to_torch(dst), case.expected, "output_tensor after the call")
    assert_bit_identical(ttnn.to_torch(src), case.src_before, "input_tensor after the call (must be unchanged)")


def copy_and_check(device, src_torch, dst_torch, *, poison_padding=None):
    case = CopyCase(device, src_torch, dst_torch, poison_padding=poison_padding)
    call_and_check(case)
    return case


@pytest.mark.parametrize("shape", PHASE0_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_values(device, shape):
    torch.manual_seed(42)
    src = torch.randn(shape, dtype=torch.float32)
    dst = torch.randn(shape, dtype=torch.float32)
    copy_and_check(device, src, dst)


def test_g6_copy_adversarial_bits(device):
    torch.manual_seed(42)
    shape = (2, 12, 128, 128)
    src = torch.randn(shape, dtype=torch.float32).contiguous()
    flat_bits = src.view(torch.int32).reshape(-1)
    index = torch.arange(0, flat_bits.numel(), 7)
    specials = torch.tensor([b - (1 << 32) if b >= 1 << 31 else b for b in FLOAT32_SPECIAL_BITS], dtype=torch.int32)
    # Bit patterns written through an int32 view: no float arithmetic touches them.
    flat_bits[index] = specials[torch.arange(index.numel()) % len(specials)]
    dst = torch.randn(shape, dtype=torch.float32)
    copy_and_check(device, src, dst)


@pytest.mark.parametrize("shape", [(1, 1), H_NON_ALIGNED_SHAPE], ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_poisoned_padding(device, shape):
    torch.manual_seed(42)
    src = torch.randn(shape, dtype=torch.float32)
    dst = torch.randn(shape, dtype=torch.float32)
    copy_and_check(device, src, dst, poison_padding=1000.0)


def test_g6_copy_rebinds_buffers(device):
    """A repeat call of one shape with other buffers must write its own
    destination and no other (BB-07); repeating a call gives the same bits (BB-06).
    Both pairs stay alive so their buffers cannot share addresses."""
    torch.manual_seed(42)
    shape = (1, 12, 128, 128)
    case_a = CopyCase(device, torch.randn(shape), torch.randn(shape))
    case_b = CopyCase(device, torch.randn(shape), torch.randn(shape))
    assert case_a.dst.buffer_address() != case_b.dst.buffer_address()
    assert case_a.src.buffer_address() != case_b.src.buffer_address()

    call_and_check(case_a)
    assert_bit_identical(ttnn.to_torch(case_b.dst), case_b.dst_before, "untouched destination after the first call")

    call_and_check(case_b)
    assert_bit_identical(ttnn.to_torch(case_a.dst), case_a.expected, "first destination after the second call")

    call_and_check(case_a)
    assert_bit_identical(ttnn.to_torch(case_b.dst), case_b.expected, "second destination after the repeat call")


@pytest.mark.parametrize(
    "shape,block_pages,max_cores",
    [
        ((16, 12, 128, 128), 3, None),  # full grid: 28/27 pages per core in 10/9 blocks, ragged tail, ring wrap
        ((1, 12, 128, 128), 5, 3),  # 64 pages per core spanning 3-4 banks, runs cut by block edges
    ],
    ids=["full_grid_block3", "3cores_block5"],
)
def test_g6_copy_forced_multi_block(device, monkeypatch, shape, block_pages, max_cores):
    """Runs the multi-block and multi-bank walk that no production shape reaches
    with the default knobs, but every shape past the L1 clamp or a capped grid does."""
    page_bytes = 4096  # float32 tile
    monkeypatch.setattr(plan_knobs, "L1_BUDGET_BYTES", plan_knobs.CB_DEPTH_BLOCKS * block_pages * page_bytes)
    monkeypatch.setattr(plan_knobs, "MAX_CORES", max_cores)

    torch.manual_seed(42)
    src = torch.randn(shape, dtype=torch.float32)
    dst = torch.randn(shape, dtype=torch.float32)

    plan = describe_plan(to_device(src, device))
    assert plan["block_pages"] == block_pages
    assert max(plan["core_pages"]) > plan["block_pages"], "the forced plan must hold more than one block per core"
    if max_cores is not None:
        assert plan["num_cores"] == max_cores

    copy_and_check(device, src, dst)


@pytest.mark.parametrize("shape", PHASE0_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_g6_copy_plan(device, shape):
    x = to_device(torch.zeros(shape, dtype=torch.float32), device)
    plan = describe_plan(x)
    assert plan == describe_plan(x), "the plan must be deterministic (BB-08)"
    assert plan["regime"] == "dram_page_stream"
    assert plan["num_pages"] == x.buffer_num_pages()
    assert plan["num_cores"] == len(plan["core_pages"])
    assert sum(plan["core_pages"]) == plan["num_pages"]
    assert plan["l1_bytes_per_core"] <= L1_BYTES_LEFT_BY_MODEL
    assert sum(1 for role in plan["roles"].values() if role != "idle") == plan["num_cores"]


def test_g6_copy_rejects_shape_mismatch(device, expect_error):
    src = to_device(torch.zeros(1, 1, 32, 32), device)
    dst = to_device(torch.zeros(1, 1, 32, 64), device)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)


def test_g6_copy_rejects_dtype_mismatch(device, expect_error):
    src = to_device(torch.zeros(1, 1, 32, 32), device)
    dst = to_device(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), device, dtype=ttnn.bfloat16)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)


def test_g6_copy_rejects_layout_mismatch(device, expect_error):
    src = to_device(torch.zeros(1, 1, 32, 32), device)
    dst = to_device(torch.zeros(1, 1, 32, 32), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)


def test_g6_copy_rejects_memory_config_mismatch(device, expect_error):
    src = to_device(torch.zeros(1, 1, 32, 32), device)
    dst = to_device(torch.zeros(1, 1, 32, 32), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)


def test_g6_copy_rejects_rank_5(device, expect_error):
    src = to_device(torch.zeros(1, 1, 1, 32, 32), device)
    dst = to_device(torch.zeros(1, 1, 1, 32, 32), device)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)


def test_g6_copy_rejects_host_tensor(device, expect_error):
    src = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    dst = to_device(torch.zeros(1, 1, 32, 32), device)
    with expect_error((ValueError, RuntimeError), "g6_copy"):
        g6_copy(src, dst)
