# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone device test for expm1_tile's iterations and vector_mode (api/compute/compute_kernel_api.h).

Drives expm1_tile<false, DST_ACCUM_MODE, iterations>(0, vector_mode) through ttnn.generic_op with the
eltwise_sfpu test kernel, in both dst-accumulator modes. Inside the region the call covers, the result must
equal ttnn.expm1 (the default expm1_tile(0)) bit for bit; outside it, the input must pass through bit for bit.
"""

import pytest
import torch
import ttnn

TILE = 32
FACE = 16

# (ttnn dtype, tile page bytes, fp32_dest_acc_en, unpack-to-DEST mode). fp32 unpacks straight to DEST, since SrcA
# would truncate it to TF32 and the passthrough check would fail.
DST_MODES = {
    "fp32_dest": (ttnn.float32, TILE * TILE * 4, True, ttnn.UnpackToDestMode.UnpackToDestFp32),
    "bf16_dest": (ttnn.bfloat16, TILE * TILE * 2, False, ttnn.UnpackToDestMode.Default),
}


def covered_mask(vector_mode, iterations):
    """Elements of a tile that expm1_tile<..., iterations>(0, vector_mode) computes.

    VectorMode::R visits faces 0 and 1, C faces 0 and 2, RC all four. Within a face, SFPU iteration k covers face rows
    4*(k//2) .. 4*(k//2)+3 in the even columns for even k and the odd columns for odd k (32 lanes = 4 rows x 8
    columns). So R with 2 iterations covers tile rows 0-3, and R with 1 iteration leaves row 0's odd columns unwritten.
    """
    faces = {"RC": [(0, 0), (0, 1), (1, 0), (1, 1)], "R": [(0, 0), (0, 1)], "C": [(0, 0), (1, 0)]}[vector_mode]
    mask = torch.zeros(TILE, TILE, dtype=torch.bool)
    for face_row, face_col in faces:
        for k in range(iterations):
            row = face_row * FACE + 4 * (k // 2)
            mask[row : row + 4, face_col * FACE + (k % 2) : (face_col + 1) * FACE : 2] = True
    return mask


def run_expm1_tile(device, x, dst_mode, vector_mode, iterations):
    dtype, page_bytes, fp32_dest_acc_en, unpack_mode = DST_MODES[dst_mode]
    num_tiles = x.shape[1]
    tt_in = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb_in, cb_out = 0, 16

    def cb(index):
        return ttnn.CBDescriptor(
            total_size=2 * page_bytes,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address(), num_tiles, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address(), num_tiles, 0]

    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en)
    # Host descriptors use the maximum CB count so this vector covers both Wormhole (32) and Blackhole (64).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    unpack_modes[cb_in] = unpack_mode
    compute_config.unpack_to_dest_mode = unpack_modes

    chain = f"expm1_tile_init(); expm1_tile<false, DST_ACCUM_MODE, {iterations}>(0, VectorMode::{vector_mode});"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(tt_in).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[cb_out] + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu.cpp",
            core_ranges=core,
            compile_time_args=[num_tiles, 1],
            defines=[("SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", chain)],
            runtime_args=[],
            config=compute_config,
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(cb_in), cb(cb_out)])
    out = ttnn.generic_op([tt_in, tt_out], program)
    return ttnn.to_torch(out), ttnn.to_torch(ttnn.expm1(tt_in)), ttnn.to_torch(tt_in)


def raw_bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize("dst_mode", list(DST_MODES))
@pytest.mark.parametrize(
    "vector_mode, iterations",
    [
        pytest.param("RC", 8, id="RC-8"),  # the defaults, spelled out
        pytest.param("R", 2, id="R-2"),  # rows 0-3, as KDA chunk preparation uses it
        pytest.param("C", 8, id="C-8"),
        pytest.param("R", 1, id="R-1"),  # even columns of rows 0-3 only
    ],
)
def test_expm1_tile_params(device, dst_mode, vector_mode, iterations):
    torch.manual_seed(0)
    num_tiles = 3
    # expm1(x) != x in either dtype for every x here, so a lane the call skips is visible.
    x = torch.rand(1, num_tiles, TILE, TILE) * 12.0 - 6.0

    out, reference, inp = run_expm1_tile(device, x, dst_mode, vector_mode, iterations)

    mask = covered_mask(vector_mode, iterations).expand_as(out)
    out_bits, ref_bits, in_bits = raw_bits(out), raw_bits(reference), raw_bits(inp)
    assert torch.equal(out_bits[mask], ref_bits[mask]), "covered elements differ from ttnn.expm1"
    assert torch.equal(out_bits[~mask], in_bits[~mask]), "uncovered elements did not pass through"
    assert (out_bits != in_bits)[mask].float().mean() > 0.99, "expm1 left covered elements unchanged"
