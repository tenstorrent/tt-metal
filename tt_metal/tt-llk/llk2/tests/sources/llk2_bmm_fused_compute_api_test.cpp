// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the LLK 2.0 tt-metal kernel kernels/bmm_large_block_zm_fused_bias_activation.cpp, the file
// itself, on one core in the tt-llk harness.
//
// This file gives the kernel what tt-metal's build would (tests/helpers/tt_metal_kernel.h): its compile-time args from
// the test's parameters, named args for the CB ids and formats and the activation, and the harness's stand-in for
// tt-metal's DataflowBuffer (tests/helpers/test_dataflow_buffer.h). The test's defines (FUSE_BIAS, PACKER_L1_ACC,
// PACK_RELU, SFPU_ACTIVATION) come from params.h, as tt-metal passes them. run_kernel does the TEST ONLY setup below
// and calls the kernel's kernel_main.
//
// What is emulated, and what tt-metal does instead:
//   - in0, in1 and bias: the host wrote every block to L1 in the order the kernel pops them (the reader kernels'
//     job), and preload() marks them pushed. The buffers hold every block, so wait_front never waits.
//   - out: holds every output block; nothing pops it (tt-metal: the writer kernel).
//   - mm_partials: pushed by pack and popped by unpack inside this kernel, so both its counters must be 0 before
//     either thread looks at them, and stale values from an earlier run on the same core must be gone. The pack
//     thread zeroes them and all three threads meet at a barrier before the body starts. tt-metal: the firmware
//     zeroes every CB's stream counters (init_sync_registers, on TRISC0) before the kernel is released.
//   - buffer_configs stands in for the host's CB config, which the firmware reads to set up each TRISC's pointers.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#include "tt_metal_kernel.h" // TEST ONLY: what tt-metal's build gives a kernel
// The same definitions the kernel makes, here first so that the test can name the kernel's Dest format.
#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "kernel_settings.h"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

namespace
{

constexpr DataFormat in_format      = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format     = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat partial_format = llk2::kernel_dest_format; // mm_partials and bias: the intermediate format, as the host picks it

// Ids as the tt-metal host names the CBs: c_0 in0, c_1 in1, c_2 bias, c_16 out, c_24 intermed0.
constexpr std::uint16_t in0_id = 0, in1_id = 1, bias_id = 2, out_id = 16, partials_id = 24;

/// The 1.0 kernel's activation_type for the test's ACTIVATION (the kernel wires up silu, relu6 and hardtanh).
constexpr ttnn::operations::matmul::KernelActivation activation_type_arg()
{
    using ttnn::operations::matmul::KernelActivation;
    switch (ACTIVATION)
    {
        case llk2::VecFunction::Silu:
            return KernelActivation::SILU;
        case llk2::VecFunction::ReluMax:
            return KernelActivation::RELU6;
        case llk2::VecFunction::Hardtanh:
            return KernelActivation::HARDTANH;
        default:
            return KernelActivation::NONE;
    }
}

/// The 1.0 kernel's positional compile-time args, as the host sets them for one output subblock per block. The
/// kernel reads 0, 1, 4, 7, 8, 9, 10, 11, 13 and 15; the others stay 0.
constexpr std::uint32_t compile_time_args[] = {
    KB_DIM,     // 0  in0_block_w
    1,          // 1  in0_num_subblocks
    0,          // 2  in0_block_num_tiles (unread)
    0,          // 3  in0_subblock_num_tiles (unread)
    1,          // 4  in1_num_subblocks
    0,          // 5  in1_block_num_tiles (unread)
    0,          // 6  in1_per_core_w (unread)
    K_BLOCKS,   // 7  num_blocks_inner_dim
    1,          // 8  num_blocks_w_dim
    OUT_BLOCKS, // 9  num_blocks_h_dim
    RT_DIM,     // 10 out_subblock_h
    CT_DIM,     // 11 out_subblock_w
    0,          // 12 out_subblock_num_tiles (unread)
    1,          // 13 batch
    0,          // 14 out_block_num_tiles (unread)
    0,          // 15 untilize_out: off, the only mode the kernel takes
};

/// The named compile-time args the kernel reads.
constexpr llk2_test::NamedArg named_args[] = {
    {"cb_in0", in0_id},
    {"cb_in1", in1_id},
    {"cb_bias", bias_id},
    {"cb_out", out_id},
    {"cb_intermed0", partials_id},
    {"cb_in0_format", static_cast<std::uint32_t>(in_format)},
    {"cb_in1_format", static_cast<std::uint32_t>(in_format)},
    {"cb_bias_format", static_cast<std::uint32_t>(partial_format)},
    {"cb_intermed0_format", static_cast<std::uint32_t>(partial_format)},
    {"cb_out_format", static_cast<std::uint32_t>(out_format)},
    {"activation_type", static_cast<std::uint32_t>(activation_type_arg())},
    {"activation_param0", ACTIVATION_P0},
    {"activation_param1", ACTIVATION_P1},
};

} // namespace

/// tt-metal: the kernel's positional compile-time arg i.
constexpr std::uint32_t get_compile_time_arg_val(std::uint32_t i)
{
    return compile_time_args[i];
}

/// tt-metal: the kernel's named compile-time arg `name`.
constexpr std::uint32_t get_named_compile_time_arg_val(const char* name)
{
    return llk2_test::find_named_arg(named_args, name);
}

#include "bmm_large_block_zm_fused_bias_activation.cpp" // the kernel file, as tt-metal builds it

constexpr std::uint32_t out_block_tiles = RT_DIM * CT_DIM;

/// Bytes of one tile (a page) in format f.
constexpr std::uint32_t page_bytes(DataFormat f)
{
    return llk2::tile_bytes(f, llk2::default_tile_datums);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    // TEST ONLY: the host's CB config. buffer_C holds the bias (CT_DIM tiles) and then the partials buffer's pages
    // (one output block).
    using llk2_test::buffer_configs;
    buffer_configs[in0_id]      = {params.buffer_A[0], page_bytes(in_format), OUT_BLOCKS * K_BLOCKS * RT_DIM * KB_DIM};
    buffer_configs[in1_id]      = {params.buffer_B[0], page_bytes(in_format), OUT_BLOCKS * K_BLOCKS * KB_DIM * CT_DIM};
    buffer_configs[bias_id]     = {params.buffer_C[0], page_bytes(partial_format), CT_DIM};
    buffer_configs[partials_id] = {params.buffer_C[CT_DIM], page_bytes(partial_format), out_block_tiles};
    buffer_configs[out_id]      = {params.buffer_Res[0], page_bytes(out_format), OUT_BLOCKS * out_block_tiles};

    // TEST ONLY: what the reader kernels and the firmware do in tt-metal (see the top of the file).
    DataflowBuffer(in0_id).preload(OUT_BLOCKS * K_BLOCKS * RT_DIM * KB_DIM);
    DataflowBuffer(in1_id).preload(OUT_BLOCKS * K_BLOCKS * KB_DIM * CT_DIM);
    DataflowBuffer(bias_id).preload(CT_DIM);
    DataflowBuffer(partials_id).zero_counters();
    DataflowBuffer(out_id).zero_counters();
    llk2_test::harness_barrier(); // every counter written before any thread reads one

    kernel_main();
}
