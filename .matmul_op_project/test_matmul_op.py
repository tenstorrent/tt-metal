# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test harness for the MatmulOp class.

Exercises MatmulOp in all three modes (Mode 1 / 2 / 3) by creating a TT-Metal
program with the test_matmul_op_kernel.cpp compute kernel, feeding tilized data,
running the program, and comparing the output to a torch.matmul reference.

Uses the ttnn.generic_op Python API to create programs with custom kernels.

IMPORTANT: Tile ordering constraints
-------------------------------------
The reader kernel reads tiles sequentially from DRAM (interleaved tile layout).
For this to produce the correct tile ordering within CB blocks, we constrain
the test dimensions so that sequential DRAM tile order matches what the compute
kernel expects:
- For in0 (A): M_tiles=1 when K blocking is needed, so each inner block is
  a contiguous run of tiles in DRAM.
- For in1 (B): tiles in row-major order (K rows, N cols) map naturally to
  the compute kernel's expected layout when read sequentially per inner block.
"""

import os
import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_H = 32
TILE_W = 32
TILE_SIZE_BF16 = 2 * 1024  # 32x32 * 2 bytes (Float16_b)
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "/localdev/wransom/tt-metal")
TEST_KERNEL_PATH = os.path.join(TT_METAL_HOME, ".matmul_op_project", "test_matmul_op_kernel.cpp")


# ---------------------------------------------------------------------------
# Inline reader kernel: reads tiles from interleaved DRAM into CB0 and CB1.
#
# Reads num_blocks iterations. Each iteration reads in0_block_num_tiles tiles
# into CB0 then in1_block_num_tiles tiles into CB1, sequentially from DRAM.
#
# Uses TensorAccessor for interleaved tile addressing.
#
# Compile-time args: TensorAccessorArgs for in0 (offset 0), then for in1.
# Runtime args: [in0_addr, in1_addr, num_blocks, in0_block_tiles, in1_block_tiles]
# ---------------------------------------------------------------------------
MATMUL_READER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(3);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    uint32_t in0_tile_bytes = get_tile_size(cb_in0);
    uint32_t in1_tile_bytes = get_tile_size(cb_in1);

    constexpr auto in0_accessor_args = TensorAccessorArgs<0>();
    constexpr auto in1_accessor_args = TensorAccessorArgs<in0_accessor_args.next_compile_time_args_offset()>();
    const auto in0_accessor = TensorAccessor(in0_accessor_args, in0_addr, in0_tile_bytes);
    const auto in1_accessor = TensorAccessor(in1_accessor_args, in1_addr, in1_tile_bytes);

    uint32_t in0_tile_id = 0;
    uint32_t in1_tile_id = 0;

    for (uint32_t b = 0; b < num_blocks; b++) {
        // Read in0 block
        cb_reserve_back(cb_in0, in0_block_num_tiles);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_in0);
        for (uint32_t t = 0; t < in0_block_num_tiles; t++) {
            uint64_t noc_addr = get_noc_addr(in0_tile_id, in0_accessor);
            noc_async_read(noc_addr, l1_write_addr_in0, in0_tile_bytes);
            l1_write_addr_in0 += in0_tile_bytes;
            in0_tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, in0_block_num_tiles);

        // Read in1 block
        cb_reserve_back(cb_in1, in1_block_num_tiles);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
        for (uint32_t t = 0; t < in1_block_num_tiles; t++) {
            uint64_t noc_addr = get_noc_addr(in1_tile_id, in1_accessor);
            noc_async_read(noc_addr, l1_write_addr_in1, in1_tile_bytes);
            l1_write_addr_in1 += in1_tile_bytes;
            in1_tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in1, in1_block_num_tiles);
    }
}
"""

# ---------------------------------------------------------------------------
# Inline writer kernel: writes output tiles from CB16 to DRAM (interleaved).
#
# Runtime args: [out_addr, num_tiles]
# ---------------------------------------------------------------------------
MATMUL_WRITER_SOURCE = """\
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;
    uint32_t tile_bytes = get_tile_size(cb_out);
    DataFormat data_format = get_dataformat(cb_out);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
"""


# ---------------------------------------------------------------------------
# Helper: create a CBDescriptor
# ---------------------------------------------------------------------------
def make_cb(buffer_index, core_ranges, num_tiles, data_format=ttnn.bfloat16):
    """Create a CBDescriptor for the given buffer_index with num_tiles pages."""
    page_size = TILE_SIZE_BF16
    total_size = num_tiles * page_size
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=buffer_index,
        data_format=data_format,
        page_size=page_size,
    )
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[fmt],
    )


# ---------------------------------------------------------------------------
# Helper: build and run a matmul test program
# ---------------------------------------------------------------------------
def run_matmul_test(
    device,
    # Matrix dimensions in elements (must be multiples of 32)
    M_elem,
    K_elem,
    N_elem,
    # Test kernel compile-time args (10 values)
    test_mode,
    batch,
    Mt_arg,
    Kt_arg,
    Nt_arg,
    out_subblock_h,
    out_subblock_w,
    in0_block_w,
    in0_num_subblocks,
    in1_num_subblocks,
    # CB sizing
    cb0_num_tiles,
    cb1_num_tiles,
    cb16_num_tiles,
    cb24_num_tiles=0,
    # Reader block sizing
    num_reader_blocks=1,
    in0_block_num_tiles_per_block=1,
    in1_block_num_tiles_per_block=1,
    # Total output tiles
    out_total_tiles=1,
    pcc_threshold=0.999,
):
    """
    Build and run a test program for the MatmulOp test kernel.

    Creates input tensors, sets up CBs, reader/writer/compute kernels,
    runs the program via generic_op, and validates output against torch.matmul.
    """
    torch.manual_seed(42)

    # Create torch tensors
    A_torch = torch.randn(1, 1, M_elem, K_elem, dtype=torch.bfloat16).float()
    B_torch = torch.randn(1, 1, K_elem, N_elem, dtype=torch.bfloat16).float()
    C_ref = torch.matmul(A_torch, B_torch)

    # Create TTNN tensors on device
    A_tt = ttnn.from_torch(
        A_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B_tt = ttnn.from_torch(
        B_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Allocate output tensor
    out_shape = [1, 1, M_elem, N_elem]
    C_tt = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # Core setup: single core (0, 0)
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular Buffers ---
    cbs = [
        make_cb(0, core_range, cb0_num_tiles),  # CB0: in0
        make_cb(1, core_range, cb1_num_tiles),  # CB1: in1
        make_cb(16, core_range, cb16_num_tiles),  # CB16: output
    ]
    if cb24_num_tiles > 0:
        cbs.append(make_cb(24, core_range, cb24_num_tiles))  # CB24: partials

    # --- Compile-time args for reader (TensorAccessor args) ---
    in0_accessor_ct_args = ttnn.TensorAccessorArgs(A_tt).get_compile_time_args()
    in1_accessor_ct_args = ttnn.TensorAccessorArgs(B_tt).get_compile_time_args()
    reader_ct_args = list(in0_accessor_ct_args) + list(in1_accessor_ct_args)

    # --- Reader kernel ---
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [
        A_tt.buffer_address(),
        B_tt.buffer_address(),
        num_reader_blocks,
        in0_block_num_tiles_per_block,
        in1_block_num_tiles_per_block,
    ]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=MATMUL_READER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[0][0] = [
        C_tt.buffer_address(),
        out_total_tiles,
    ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=MATMUL_WRITER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_range,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        test_mode,
        batch,
        Mt_arg,
        Kt_arg,
        Nt_arg,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        in0_num_subblocks,
        in1_num_subblocks,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=TEST_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
        ),
    )

    # --- Build program ---
    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )

    # --- Run ---
    io_tensors = [A_tt, B_tt, C_tt]
    ttnn.generic_op(io_tensors, program)
    ttnn.synchronize_device(device)

    # --- Validate ---
    C_actual = ttnn.to_torch(C_tt)
    passed, pcc_val = assert_with_pcc(C_ref, C_actual, pcc=pcc_threshold)
    return passed, pcc_val


# ===========================================================================
# Test Cases
# ===========================================================================


@pytest.mark.timeout(60)
def test_mode3_tile_basic(device):
    """
    Mode 3 (Tile Auto): TileMatmulOp::run()

    Verifies the fully automatic tile-mode matmul loop.
    A = (32, 128), B = (128, 32), C = (32, 32)
    1 batch, M=1, K=4, N=1 tiles. Accumulates K=4 inner tiles per output tile.

    Maps to call site T1 (bmm.cpp).

    Reader feeds 1 in0 tile + 1 in1 tile per iteration (K*M*N = 4 blocks).
    Tile ordering: M=1 means A tiles are sequential (row 0, cols 0..3).
    N=1 means B tiles are sequential (rows 0..3, col 0).
    """
    M_tiles, K_tiles, N_tiles = 1, 4, 1

    run_matmul_test(
        device,
        M_elem=M_tiles * TILE_H,
        K_elem=K_tiles * TILE_W,
        N_elem=N_tiles * TILE_W,
        test_mode=0,  # mode3_tile
        batch=1,
        Mt_arg=M_tiles,
        Kt_arg=K_tiles,
        Nt_arg=N_tiles,
        out_subblock_h=1,
        out_subblock_w=1,
        in0_block_w=1,
        in0_num_subblocks=1,
        in1_num_subblocks=1,
        cb0_num_tiles=2,
        cb1_num_tiles=2,
        cb16_num_tiles=1,
        # Reader: M*N*K = 4 iterations, 1 tile each
        num_reader_blocks=M_tiles * N_tiles * K_tiles,
        in0_block_num_tiles_per_block=1,
        in1_block_num_tiles_per_block=1,
        out_total_tiles=M_tiles * N_tiles,
    )


@pytest.mark.timeout(60)
def test_mode3_block_multisubblock(device):
    """
    Mode 3 (Block Auto): BlockMatmulOp::run() with subblocking and spill/reload.

    A = (32, 128), B = (128, 64), C = (32, 64)
    M=1, K=4, N=2 tiles. subblock_h=1, subblock_w=2, in0_block_w=2.
    in0_num_subblocks=1, in1_num_subblocks=1, num_blocks_inner=2.

    With M=1, the sequential DRAM tile order for A matches block order:
      Block 0: A[0,0], A[0,1]  |  Block 1: A[0,2], A[0,3]
    For B (K=4 rows, N=2 cols):
      Block 0: B[0,0], B[0,1], B[1,0], B[1,1]
      Block 1: B[2,0], B[2,1], B[3,0], B[3,1]

    This exercises spill/reload: first inner block spills to CB24, second
    reloads and outputs to CB16.

    Maps to call sites B9/B10/B15.
    """
    M_tiles, K_tiles, N_tiles = 1, 4, 2
    out_subblock_h, out_subblock_w = 1, 2
    in0_block_w = 2
    in0_num_subblocks = M_tiles // out_subblock_h  # 1
    in1_num_subblocks = N_tiles // out_subblock_w  # 1
    num_blocks_inner = K_tiles // in0_block_w  # 2

    in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks  # 2
    in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks  # 4
    out_subblock_num_tiles = out_subblock_h * out_subblock_w  # 2
    total_output_tiles = M_tiles * N_tiles  # 2

    run_matmul_test(
        device,
        M_elem=M_tiles * TILE_H,
        K_elem=K_tiles * TILE_W,
        N_elem=N_tiles * TILE_W,
        test_mode=5,  # mode3_block
        batch=1,
        Mt_arg=M_tiles,
        Kt_arg=num_blocks_inner,
        Nt_arg=N_tiles,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        in0_block_w=in0_block_w,
        in0_num_subblocks=in0_num_subblocks,
        in1_num_subblocks=in1_num_subblocks,
        cb0_num_tiles=in0_block_num_tiles * 2,
        cb1_num_tiles=in1_block_num_tiles * 2,
        cb16_num_tiles=out_subblock_num_tiles,
        cb24_num_tiles=out_subblock_num_tiles,  # partials
        num_reader_blocks=num_blocks_inner,
        in0_block_num_tiles_per_block=in0_block_num_tiles,
        in1_block_num_tiles_per_block=in1_block_num_tiles,
        out_total_tiles=total_output_tiles,
    )


@pytest.mark.timeout(60)
def test_mode2_block_spill_reload(device):
    """
    Mode 2 (Semi-Automatic with spill/reload): BlockMatmulOp.

    A = (32, 128), B = (128, 64), C = (32, 64)
    M=1, K=4, N=2 tiles. subblock_h=1, subblock_w=2, in0_block_w=2.
    num_blocks_inner=2. First block spills, second reloads and outputs.

    The semi-automatic mode exercises: init(), begin_subblock(),
    accumulate(), end_to_partials(), reload_partials(), end_to_output().

    Maps to call sites B1/B2/B3/B16.
    """
    M_tiles, K_tiles, N_tiles = 1, 4, 2
    out_subblock_h, out_subblock_w = 1, 2
    in0_block_w = 2
    in0_num_subblocks = 1
    in1_num_subblocks = 1
    num_blocks_inner = K_tiles // in0_block_w  # 2

    in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks  # 2
    in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks  # 4
    out_subblock_num_tiles = out_subblock_h * out_subblock_w  # 2
    total_output_tiles = M_tiles * N_tiles  # 2

    run_matmul_test(
        device,
        M_elem=M_tiles * TILE_H,
        K_elem=K_tiles * TILE_W,
        N_elem=N_tiles * TILE_W,
        test_mode=1,  # mode2_semi
        batch=1,
        Mt_arg=M_tiles,
        Kt_arg=num_blocks_inner,
        Nt_arg=N_tiles,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        in0_block_w=in0_block_w,
        in0_num_subblocks=in0_num_subblocks,
        in1_num_subblocks=in1_num_subblocks,
        cb0_num_tiles=in0_block_num_tiles * 2,
        cb1_num_tiles=in1_block_num_tiles * 2,
        cb16_num_tiles=out_subblock_num_tiles,
        cb24_num_tiles=out_subblock_num_tiles,  # partials
        num_reader_blocks=num_blocks_inner,
        in0_block_num_tiles_per_block=in0_block_num_tiles,
        in1_block_num_tiles_per_block=in1_block_num_tiles,
        out_total_tiles=total_output_tiles,
    )


@pytest.mark.timeout(60)
def test_mode2_block_no_spill(device):
    """
    Mode 2 (No Spill): BlockMatmulOp, single inner block.

    A = (32, 64), B = (64, 64), C = (32, 64)
    M=1, K=2, N=2 tiles. subblock_h=1, subblock_w=2, in0_block_w=2.
    Single inner block: begin_subblock -> accumulate -> end_to_output.
    No spill/reload path.

    Maps to call sites B4/B5 (SDPA matmul_blocks).
    """
    M_tiles, K_tiles, N_tiles = 1, 2, 2
    out_subblock_h, out_subblock_w = 1, 2
    in0_block_w = 2
    in0_num_subblocks = 1
    in1_num_subblocks = 1

    in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks  # 2
    in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks  # 4
    out_subblock_num_tiles = out_subblock_h * out_subblock_w  # 2
    total_output_tiles = M_tiles * N_tiles  # 2

    run_matmul_test(
        device,
        M_elem=M_tiles * TILE_H,
        K_elem=K_tiles * TILE_W,
        N_elem=N_tiles * TILE_W,
        test_mode=4,  # mode2_no_spill
        batch=1,
        Mt_arg=M_tiles,
        Kt_arg=K_tiles,
        Nt_arg=N_tiles,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        in0_block_w=in0_block_w,
        in0_num_subblocks=in0_num_subblocks,
        in1_num_subblocks=in1_num_subblocks,
        cb0_num_tiles=in0_block_num_tiles * 2,
        cb1_num_tiles=in1_block_num_tiles * 2,
        cb16_num_tiles=out_subblock_num_tiles,
        cb24_num_tiles=0,  # no partials
        num_reader_blocks=1,
        in0_block_num_tiles_per_block=in0_block_num_tiles,
        in1_block_num_tiles_per_block=in1_block_num_tiles,
        out_total_tiles=total_output_tiles,
    )


@pytest.mark.timeout(60)
def test_mode1_tile_single(device):
    """
    Mode 1 (Tile): TileMatmulOp::matmul() with manual DST management.

    A = (32, 32), B = (32, 32), C = (32, 32)
    Single tile: M=1, K=1, N=1.
    init() -> tile_regs_acquire -> matmul(0,0,0) -> pack -> release.

    Maps to call sites T5/T6/T7 (width reduction, moreh_matmul simple path).
    """
    run_matmul_test(
        device,
        M_elem=TILE_H,
        K_elem=TILE_W,
        N_elem=TILE_W,
        test_mode=2,  # mode1_tile
        batch=1,
        Mt_arg=1,
        Kt_arg=1,
        Nt_arg=1,
        out_subblock_h=1,
        out_subblock_w=1,
        in0_block_w=1,
        in0_num_subblocks=1,
        in1_num_subblocks=1,
        cb0_num_tiles=2,
        cb1_num_tiles=2,
        cb16_num_tiles=1,
        num_reader_blocks=1,
        in0_block_num_tiles_per_block=1,
        in1_block_num_tiles_per_block=1,
        out_total_tiles=1,
    )


@pytest.mark.timeout(60)
def test_mode1_block_call(device):
    """
    Mode 1 (Block): BlockMatmulOp::matmul() with ct_dim=2, rt_dim=1, kt_dim=1.

    A = (32, 32), B = (32, 64), C = (32, 64)
    M=1, K=1, N=2 tiles. One matmul_block call processes the full block.

    The caller manually acquires DST, calls matmul(0, 0, 0) once
    (which processes a 1x1x2 block), then packs 2 output tiles.

    Maps to call sites B8/B11 (TopK router, MOE gate with ct_dim=2).
    """
    ct_dim, rt_dim, kt_dim = 2, 1, 1
    in0_block_tiles = rt_dim * kt_dim  # 1
    in1_block_tiles = kt_dim * ct_dim  # 2
    out_tiles = rt_dim * ct_dim  # 2

    run_matmul_test(
        device,
        M_elem=TILE_H,
        K_elem=TILE_W,
        N_elem=ct_dim * TILE_W,
        test_mode=3,  # mode1_block
        batch=1,
        Mt_arg=1,
        Kt_arg=1,  # single K step
        Nt_arg=ct_dim,
        out_subblock_h=rt_dim,
        out_subblock_w=ct_dim,
        in0_block_w=kt_dim,
        in0_num_subblocks=1,
        in1_num_subblocks=1,
        cb0_num_tiles=in0_block_tiles * 2,
        cb1_num_tiles=in1_block_tiles * 2,
        cb16_num_tiles=out_tiles,
        num_reader_blocks=1,
        in0_block_num_tiles_per_block=in0_block_tiles,
        in1_block_num_tiles_per_block=in1_block_tiles,
        out_total_tiles=out_tiles,
    )
