# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive bit identity of Blackhole's sigmoid / SiLU / exp kernels with main's single-row form.

The reference (kernels/sfpu_reference.h) runs main's per-row loops over the single-row functions _sfpu_sigmoid_ and
_sfpu_exp_accurate_, which other ops still use. For every case, the SFPU writes every input bit pattern the
destination can hold into DEST (all 2**32 FP32 encodings, or all 2**16 BF16 ones: NaNs, infinities, zeros,
subnormals included) with no unpacker or packer conversion, runs the reference and the production kernel on copies,
and compares the results bit for bit on the SFPU (kernels/compute_bit_identity.cpp). The generator cases check that
DEST holds exactly the generated patterns. A BF16 destination's store flushes the subnormals (which shows the
comparison catches differences), so the BF16-path kernels also run in an FP32 destination on the BF16 patterns stored
as raw bits (bf16_lreg), where they load every pattern, subnormals included.

The FP32 sigmoid and SiLU clamp |x| to 88 before the exponential, so their NaN inputs may give another result than
main's; every other input of every case must match bit for bit.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

_KERNELS = "tests/ttnn/unit_tests/operations/eltwise/sfpu_bit_identity/kernels"
_RESULT_WORDS = 32
_HEADER_WORDS = 3  # non-NaN differences, tiles checked, NaN differences (writer_bit_identity.cpp)

OP_GENERATOR, OP_SIGMOID, OP_SIGMOID_R2, OP_SILU, OP_EXP, OP_EXP_R2 = range(6)
# (name, op, FP32 destination, bf16_lreg, NaN inputs may differ). bf16_lreg runs the BF16-destination kernel in an
# FP32 destination on the BF16 patterns, so its inputs include the subnormals a BF16-format SFPSTORE (the native BF16
# cases' generator) flushes.
CASES = [
    ("generator_fp32", OP_GENERATOR, True, 0, False),
    ("generator_bf16", OP_GENERATOR, False, 0, False),
    ("generator_bf16_lreg", OP_GENERATOR, True, 1, False),
    ("sigmoid_fp32", OP_SIGMOID, True, 0, True),
    ("sigmoid_fp32_r2", OP_SIGMOID_R2, True, 0, True),
    ("silu_fp32", OP_SILU, True, 0, True),
    ("exp_fp32", OP_EXP, True, 0, False),
    ("exp_fp32_r2", OP_EXP_R2, True, 0, False),
    ("sigmoid_bf16", OP_SIGMOID, False, 0, False),
    ("silu_bf16", OP_SILU, False, 0, False),
    ("sigmoid_bf16_lreg", OP_SIGMOID, True, 1, False),
    ("silu_bf16_lreg", OP_SILU, True, 1, False),
]


def _program(cores, core_list, op, fp32, bf16_lreg, out, ranges):
    flag_fmt, flag_page = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * flag_page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=flag_fmt, page_size=flag_page)],
        ),
        ttnn.CBDescriptor(
            total_size=4 * _RESULT_WORDS,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=17, data_format=ttnn.uint32, page_size=4 * _RESULT_WORDS)
            ],
        ),
    ]
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for page, (core, (first, count)) in enumerate(zip(core_list, ranges)):
        compute_rt[core.x][core.y] = [first, count]
        writer_rt[core.x][core.y] = [out.buffer_address(), first, count, page]
    bf16_inputs = int(bf16_lreg or not fp32)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/compute_bit_identity.cpp",
            core_ranges=cores,
            compile_time_args=[op, bf16_lreg],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, math_approx_mode=False
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/writer_bit_identity.cpp",
            core_ranges=cores,
            compile_time_args=[int(fp32), bf16_inputs, *ttnn.TensorAccessorArgs(out).get_compile_time_args()],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


@pytest.mark.parametrize("name,op,fp32,bf16_lreg,nan_may_differ", CASES, ids=[c[0] for c in CASES])
def test_sfpu_bit_identity(device, name, op, fp32, bf16_lreg, nan_may_differ):
    grid = device.compute_with_storage_grid_size()
    core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    bf16_inputs = bf16_lreg or not fp32
    total = 2**16 // 1024 if bf16_inputs else 2**32 // 1024
    per, extra = divmod(total, len(core_list))
    ranges, first = [], 0
    for i in range(len(core_list)):
        count = per + (i < extra)
        ranges.append((first, count))
        first += count
    # generic_op needs an input tensor; only result is written
    dummy_in, result_tensor = (
        ttnn.from_torch(
            torch.zeros(len(core_list), _RESULT_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    )
    ttnn.generic_op([dummy_in, result_tensor], _program(cores, core_list, op, fp32, bf16_lreg, result_tensor, ranges))
    result = ttnn.to_torch(result_tensor).to(torch.int64) & 0xFFFFFFFF
    checked = int(result[:, 1].sum()) * 1024
    differences = int(result[:, 0].sum())
    nan_differences = int(result[:, 2].sum())
    shift = 16 if bf16_inputs else 0
    max_examples = _RESULT_WORDS - _HEADER_WORDS
    examples = [
        int(v) << shift for row in result for v in row[_HEADER_WORDS : _HEADER_WORDS + min(int(row[0]), max_examples)]
    ][:16]
    print(
        f"SFPU_BIT_IDENTITY {name}: {checked} inputs ({'all BF16' if bf16_inputs else 'all FP32'} encodings), "
        f"{differences} non-NaN and {nan_differences} NaN inputs differ"
        + (f"; first non-NaN inputs: {[hex(v) for v in examples]}" if examples else "")
    )
    assert checked == total * 1024
    if not nan_may_differ:
        assert nan_differences == 0
    if op == OP_GENERATOR and not fp32:
        # The BF16-format store flushes the 2 * 127 subnormals, so these reach the native BF16 cases as zeros (the
        # bf16_lreg cases cover them); this case is also the comparison's positive control.
        assert differences == 2 * (2**7 - 1)
        assert all((v >> 23) & 0xFF == 0 and v & 0x7FFFFF for v in examples)
    else:
        assert differences == 0
