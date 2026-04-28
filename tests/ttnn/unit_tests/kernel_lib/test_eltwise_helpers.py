# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Validation suite for the redesigned eltwise helper family
(`compute_kernel_lib::eltwise`).

Each test launches a custom compute kernel via `ttnn.generic_op` on a single
core (0, 0) with `num_tiles ∈ {1, 8, 64}` (single tile / fits-in-DEST /
multi-DEST-window) and validates the device output against a torch golden.

These are the pre-merge gate for changes touching:
  - eltwise_chain core (CRTP bases, EltwiseChain, eltwise_pipeline)
  - CopyTile policies (WaitAndPop / WaitNoPop / NoWaitPop / CumulativeWait)
  - eltwise_binary FPU path (add/sub/mul, same-CB dedup, PostOp clash reinit)

Compute kernels live in `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/compute_kernels/`.
Reader / writer in `.../dataflow_kernels/`.
"""

from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc, skip_for_blackhole


KERNEL_ROOT = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise"
READER_KERNEL_BY_N = {
    1: f"{KERNEL_ROOT}/dataflow_kernels/reader_1_input.cpp",
    2: f"{KERNEL_ROOT}/dataflow_kernels/reader_2_input.cpp",
    3: f"{KERNEL_ROOT}/dataflow_kernels/reader_3_input.cpp",
}
WRITER_KERNEL = f"{KERNEL_ROOT}/dataflow_kernels/writer_1_output.cpp"

IN_CB_0 = 0
IN_CB_1 = 1
IN_CB_2 = 2
OUT_CB = 16

# bfloat16 tile = 32 * 32 elements * 2 bytes = 2048 bytes.
TILE_BYTES_BF16 = 2 * 1024


def _make_tensor(shape, device, data):
    return ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _alloc_output(shape, device):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def _cb_descriptor(cb_id, core_ranges, *, double_buffered=True):
    total = (2 if double_buffered else 1) * TILE_BYTES_BF16
    return ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=ttnn.bfloat16,
                page_size=TILE_BYTES_BF16,
            )
        ],
    )


def _build_and_run(device, num_tiles, compute_kernel_path, input_tensors, expected):
    """Single-core dispatch; reader pushes input_tensors[i] to CB i."""
    shape = list(expected.shape)
    output_tensor = _alloc_output(shape, device)

    num_inputs = len(input_tensors)
    assert 1 <= num_inputs <= 3

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cbs = [_cb_descriptor(IN_CB_0 + i, core_range) for i in range(num_inputs)]
    cbs.append(_cb_descriptor(OUT_CB, core_range))

    # Reader compile-time args = concatenated per-input TensorAccessorArgs blobs.
    reader_ct = []
    for t in input_tensors:
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    src_addrs = [t.buffer_address() for t in input_tensors]
    reader_rt[core.x][core.y] = [*src_addrs, num_tiles, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL_BY_N[num_inputs],
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [OUT_CB]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [output_tensor.buffer_address(), num_tiles, 0]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=[num_tiles],
        runtime_args=[],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )

    result = ttnn.generic_op([*input_tensors, output_tensor], program)
    got = ttnn.to_torch(result)

    passing, info = comp_pcc(expected.float(), got.float(), pcc=0.9999)
    logger.info(f"{Path(compute_kernel_path).name} num_tiles={num_tiles}: {info}")
    assert passing, f"PCC mismatch: {info}"


def _shape_for(num_tiles):
    return [1, 1, 32, 32 * num_tiles]


# ---------------------------------------------------------------------------
# 1. Unary streaming — y = exp(x). Smoke test for chain core + CopyTile.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_unary_streaming(device, num_tiles):
    shape = _shape_for(num_tiles)
    x = torch.randn(shape, dtype=torch.bfloat16).clamp(-3.0, 3.0)
    expected = torch.exp(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_unary_streaming.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )


# ---------------------------------------------------------------------------
# 2. Chain fan-out — y = x * exp(x). Drives WaitNoPop + NoWaitPop.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_chain_fanout(device, num_tiles):
    shape = _shape_for(num_tiles)
    x = torch.randn(shape, dtype=torch.bfloat16).clamp(-2.0, 2.0)
    expected = x * torch.exp(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_chain_fanout.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )


# ---------------------------------------------------------------------------
# 3. Binary streaming — y = a + b.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_streaming(device, num_tiles):
    shape = _shape_for(num_tiles)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    expected = a + b

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_streaming.cpp",
        [_make_tensor(shape, device, a), _make_tensor(shape, device, b)],
        expected,
    )


# ---------------------------------------------------------------------------
# 4. Same-CB binary — y = x + x. Drives wait/pop dedup.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_same_cb(device, num_tiles):
    shape = _shape_for(num_tiles)
    x = torch.randn(shape, dtype=torch.bfloat16)
    expected = x + x

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_same_cb.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )


# ---------------------------------------------------------------------------
# 5. Binary PostOp chain with CopyTile — y = (a+b) * c.
#     Drives clashes_with_fpu reinit (CopyTile in PostOp clobbers unpack MOP,
#     binary_short_init re-fires before next iter's binary_exec).
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_postop_load(device, num_tiles):
    shape = _shape_for(num_tiles)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    c = torch.randn(shape, dtype=torch.bfloat16)
    expected = (a + b) * c

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_postop_load.cpp",
        [
            _make_tensor(shape, device, a),
            _make_tensor(shape, device, b),
            _make_tensor(shape, device, c),
        ],
        expected,
    )


# ---------------------------------------------------------------------------
# 6. Tier 2 trig: y = sin(x) — covers eltwise_trig.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_unary_sin(device, num_tiles):
    shape = _shape_for(num_tiles)
    x = torch.randn(shape, dtype=torch.bfloat16) * 3.0
    expected = torch.sin(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_unary_sin.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )


# ---------------------------------------------------------------------------
# 7. Tier 2 rounding: y = floor(x) — covers eltwise_rounding.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_unary_floor(device, num_tiles):
    shape = _shape_for(num_tiles)
    # Avoid values right on integer boundaries — bf16 quantization can flip
    # the floor result by ±1. Use offsets that keep us in the open interior
    # of each integer interval.
    x = (torch.randn(shape, dtype=torch.float32) * 5.0 + 0.37).to(torch.bfloat16)
    expected = torch.floor(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_unary_floor.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )


# ---------------------------------------------------------------------------
# 7. Tier 2 unary: y = silu(x) — covers eltwise_activations expansion.
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_unary_silu(device, num_tiles):
    shape = _shape_for(num_tiles)
    x = torch.randn(shape, dtype=torch.bfloat16).clamp(-3.0, 3.0)
    expected = x.float() * torch.sigmoid(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_unary_silu.cpp",
        [_make_tensor(shape, device, x)],
        expected,
    )
