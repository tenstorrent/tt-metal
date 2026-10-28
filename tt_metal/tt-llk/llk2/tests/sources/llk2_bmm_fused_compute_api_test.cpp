// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the LLK 2.0 body of ttnn's bmm_large_block_zm_fused_bias_activation
// (kernels/bmm_large_block_zm_fused_bias_activation.h), on one core in the tt-llk harness.
//
// run_kernel builds the five dataflow buffers with the harness's stand-in for tt-metal's DataflowBuffer
// (tests/helpers/test_dataflow_buffer.h), does the TEST ONLY setup below, and calls the same body the tt-metal kernel calls.
//
// What is emulated, and what tt-metal does instead:
//   - in0, in1 and bias: the host wrote every block to L1 in the order the kernel pops them (the reader kernels'
//     job), and preload() marks them pushed. The buffers hold every block, so wait_front never waits.
//   - out: holds every output block; nothing pops it (tt-metal: the writer kernel).
//   - mm_partials: pushed by pack and popped by unpack inside this kernel, so both its counters must be 0 before
//     either thread looks at them, and stale values from an earlier run on the same core must be gone. The pack
//     thread zeroes them and all three threads meet at a barrier before the body starts. tt-metal: the firmware
//     zeroes every CB's stream counters (init_sync_registers, on TRISC0) before the kernel is released.
//   - The buffer constructor stands in for the firmware's per-thread pointer setup from the host's CB config.

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

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "bmm_large_block_zm_fused_bias_activation.h"
#include "test_dataflow_buffer.h" // TEST ONLY: tt-metal's DataflowBuffer, emulated

constexpr DataFormat in_format      = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format     = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat partial_format = llk2::kernel_dest_format; // mm_partials and bias: the intermediate format, as the host picks it

inline constexpr llk2::bmm_fused_config config {
    .rt_dim          = RT_DIM,
    .ct_dim          = CT_DIM,
    .kb_dim          = KB_DIM,
    .k_blocks        = K_BLOCKS,
    .out_blocks      = OUT_BLOCKS,
    .l1_acc          = L1_ACC,
    .fuse_bias       = FUSE_BIAS,
    .pack_relu       = PACK_RELU,
    .activation      = ACTIVATION,
    .activation_p0   = ACTIVATION_P0,
    .activation_p1   = ACTIVATION_P1,
    .in0_format      = in_format,
    .in1_format      = in_format,
    .bias_format     = partial_format,
    .partials_format = partial_format,
    .out_format      = out_format,
};

constexpr std::uint32_t out_block_tiles = RT_DIM * CT_DIM;

/// Bytes of one tile (a page) in format f.
constexpr std::uint32_t page_bytes(DataFormat f)
{
    return llk2::tile_bytes(f, llk2::default_tile_datums);
}

// Ids as the tt-metal host names the CBs: c_0 in0, c_1 in1, c_2 bias, c_16 out, c_24 intermed0.
constexpr std::uint16_t in0_id = 0, in1_id = 1, bias_id = 2, out_id = 16, partials_id = 24;

void run_kernel(RUNTIME_PARAMETERS params)
{
    // buffer_C holds the bias (CT_DIM tiles) and then the partials buffer's pages (one output block).
    llk2_test::DataflowBuffer in0(in0_id, params.buffer_A[0], page_bytes(in_format), OUT_BLOCKS * K_BLOCKS * RT_DIM * KB_DIM);
    llk2_test::DataflowBuffer in1(in1_id, params.buffer_B[0], page_bytes(in_format), OUT_BLOCKS * K_BLOCKS * KB_DIM * CT_DIM);
    llk2_test::DataflowBuffer bias(bias_id, params.buffer_C[0], page_bytes(partial_format), CT_DIM);
    llk2_test::DataflowBuffer partials(partials_id, params.buffer_C[CT_DIM], page_bytes(partial_format), out_block_tiles);
    llk2_test::DataflowBuffer out(out_id, params.buffer_Res[0], page_bytes(out_format), OUT_BLOCKS * out_block_tiles);

    // TEST ONLY: what the reader kernels and the firmware do in tt-metal (see the top of the file).
    in0.preload(OUT_BLOCKS * K_BLOCKS * RT_DIM * KB_DIM);
    in1.preload(OUT_BLOCKS * K_BLOCKS * KB_DIM * CT_DIM);
    bias.preload(CT_DIM);
    partials.zero_counters();
    out.zero_counters();
    llk2_test::harness_barrier(); // every counter written before any thread reads one

    llk2::bmm_large_block_zm_fused_bias_activation<config, llk2::kernel_dest_format>(in0, in1, bias, partials, out);
}
