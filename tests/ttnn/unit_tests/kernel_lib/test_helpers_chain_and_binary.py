# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for compute_kernel_lib::sfpu_chain and binary_op helpers.

Each test launches a custom compute kernel via ttnn.generic_op and compares
the device output against a torch golden. These are the validation gate for
changes touching:
- sfpu_chain Load / fan-out lifecycle (LoadPolicy, NoWaitPop, no compaction)
- binary_op same-CB wait/pop deduplication
- binary_op PostOp chain dispatch (chain-only, NoOp static_assert)
- DestReuseOp as chain element + clashes_with_fpu per-tile reinit path
- Load inside a PostOp chain (the primary FPU-clash regression surface)

See ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary/ for the compute/dataflow
kernel sources.
"""

from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc, skip_for_blackhole


KERNEL_ROOT = "ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary"
READER_KERNEL = f"{KERNEL_ROOT}/dataflow_kernels/reader_n_input.cpp"
WRITER_KERNEL = f"{KERNEL_ROOT}/dataflow_kernels/writer_1_output.cpp"

IN_CB_0 = 0
IN_CB_1 = 1
IN_CB_2 = 2
OUT_CB = 16

# bfloat16 tile = 32*32 elements * 2 bytes = 2048 bytes.
TILE_BYTES_BF16 = 2 * 1024


def _make_tensor(shape, device, data=None):
    if data is None:
        data = torch.randn(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tensor, data


def _alloc_output(shape, device):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def _cb_descriptor(cb_id, core_ranges, double_buffered=True):
    total = 2 * TILE_BYTES_BF16 if double_buffered else TILE_BYTES_BF16
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


def _build_and_run(
    device,
    num_tiles,
    compute_kernel_path,
    input_tensors,
    expected_output,
    *,
    extra_cbs=(),
):
    """
    Run a single compute kernel on core (0,0) with `num_tiles` tiles.

    input_tensors: list of ttnn tensors corresponding to CBs c_0..c_{N-1}.
    expected_output: torch tensor of the same shape as the output.
    extra_cbs: additional CB ids beyond the inputs (e.g. for PostOp chain Loads
               that reuse an existing input CB we don't need extras, but for
               cases where PostOp references a third CB we list it here).
    """
    shape = list(expected_output.shape)
    output_tensor = _alloc_output(shape, device)

    num_inputs = len(input_tensors)
    assert 1 <= num_inputs <= 3

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CBs: one per input, plus output, plus any extras.
    cbs = [_cb_descriptor(IN_CB_0 + i, core_range) for i in range(num_inputs)]
    for extra_id in extra_cbs:
        cbs.append(_cb_descriptor(extra_id, core_range))
    cbs.append(_cb_descriptor(OUT_CB, core_range))

    # Reader: compile-time arg 0 = num_inputs, followed by TensorAccessorArgs per input.
    reader_ct_args = [num_inputs]
    for t in input_tensors:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    src_addrs = [t.buffer_address() for t in input_tensors]
    # Pad src addresses to 3 entries so fixed offsets work.
    while len(src_addrs) < 3:
        src_addrs.append(0)
    reader_rt[core.x][core.y] = [*src_addrs, num_tiles, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = [OUT_CB]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [output_tensor.buffer_address(), num_tiles, 0]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
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

    passing, info = comp_pcc(expected_output.float(), got.float(), pcc=0.9999)
    logger.info(f"{Path(compute_kernel_path).name} num_tiles={num_tiles}: {info}")
    assert passing, f"PCC mismatch: {info}"


# ---------------------------------------------------------------------------
# Test 1: sfpu_chain fan-out (Load + NoWaitPop + CompactLoad removal path)
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_chain_fanout_x_times_exp_x(device, num_tiles):
    """y = x * exp(x) via sfpu_chain(Load<WaitNoPop>, Load<NoWaitPop>, Exp, SfpuMul)."""
    shape = [1, 1, 32, 32 * num_tiles]
    x = torch.randn(shape, dtype=torch.bfloat16)
    # Keep values modest so exp doesn't overflow bf16.
    x = x.clamp(-2.0, 2.0)

    input_tensor, _ = _make_tensor(shape, device, data=x)
    expected = x * torch.exp(x.float())

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_chain_fanout.cpp",
        [input_tensor],
        expected,
    )


# ---------------------------------------------------------------------------
# Test 2: Same-CB binary (wait/pop dedup)
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_same_cb_add(device, num_tiles):
    """y = x + x via add(cb_in, cb_in, cb_out). Validates same-CB wait/pop dedup."""
    shape = [1, 1, 32, 32 * num_tiles]
    x = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor, _ = _make_tensor(shape, device, data=x)
    expected = x + x

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_same_cb.cpp",
        [input_tensor],
        expected,
    )


# ---------------------------------------------------------------------------
# Test 3: binary_op SUB with DestReuseMul chain PostOp (clashes_with_fpu reinit)
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_dest_reuse_mul(device, num_tiles):
    """y = (a - b) * scale via sub(..., sfpu_chain(DestReuseMul<cb_scale>))."""
    shape = [1, 1, 32, 32 * num_tiles]
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)

    # cb_scale holds a single tile whose entire 32x32 block is a single constant-ish
    # pattern. Use a 32x32 tile of random values — the test multiplies each (a-b)
    # tile by scale[0], so every output tile is elementwise (a - b) * scale.
    scale_tile = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)

    a_tensor, _ = _make_tensor(shape, device, data=a)
    b_tensor, _ = _make_tensor(shape, device, data=b)
    scale_tensor, _ = _make_tensor((1, 1, 32, 32), device, data=scale_tile)

    # Output tile n = (a_tile_n - b_tile_n) * scale_tile (element-wise per 32x32).
    expected = torch.zeros_like(a)
    for t in range(num_tiles):
        a_t = a[..., t * 32 : (t + 1) * 32]
        b_t = b[..., t * 32 : (t + 1) * 32]
        expected[..., t * 32 : (t + 1) * 32] = (a_t - b_t) * scale_tile

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_dest_reuse.cpp",
        [a_tensor, b_tensor, scale_tensor],
        expected,
    )


# ---------------------------------------------------------------------------
# Test 4: binary_op ADD with PostOp chain that loads a third CB (Load in PostOp
# path; per-tile reinit restores AB MOP after copy_tile_to_dst_init_short)
# ---------------------------------------------------------------------------
@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_binary_postop_with_load(device, num_tiles):
    """y = (a + b) * c via add(..., sfpu_chain(Load<cb_c, D1>, SfpuMul<D0,D1,D0>))."""
    shape = [1, 1, 32, 32 * num_tiles]
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    c = torch.randn(shape, dtype=torch.bfloat16)

    a_tensor, _ = _make_tensor(shape, device, data=a)
    b_tensor, _ = _make_tensor(shape, device, data=b)
    c_tensor, _ = _make_tensor(shape, device, data=c)

    expected = (a + b) * c

    _build_and_run(
        device,
        num_tiles,
        f"{KERNEL_ROOT}/compute_kernels/compute_binary_postop_load.cpp",
        [a_tensor, b_tensor, c_tensor],
        expected,
    )
