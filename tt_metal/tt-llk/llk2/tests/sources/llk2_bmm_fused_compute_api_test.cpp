// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the LLK 2.0 body of ttnn's bmm_large_block_zm_fused_bias_activation
// (kernels/bmm_large_block_zm_fused_bias_activation.h), on one core in the tt-llk harness.
//
// run_kernel is two parts:
//   harness_setup_buffers   TEST ONLY. Stands in for what the tt-metal firmware and reader kernels do (below)
//   the kernel body         the same call the tt-metal kernel makes; no LLK, init or per-thread code
//
// What is emulated, and what tt-metal does instead:
//   - in0, in1 and bias: the host wrote every block to L1 in the order the kernel pops them (the reader kernels'
//     job), and preload_dataflow_buffer marks them pushed. The buffers hold every block, so wait_front never waits.
//   - out: holds every output block; nothing pops it (tt-metal: the writer kernel).
//   - mm_partials: pushed by pack and popped by unpack inside this kernel, so both its counters must be 0 before
//     either thread looks at them, and stale values from an earlier run on the same core must be gone. The pack
//     thread zeroes them (setup_dataflow_buffer) and all three threads meet at a barrier before the body starts.
//     tt-metal: the firmware zeroes every CB's stream counters (init_sync_registers, on TRISC0) before the kernel
//     is released, so no barrier is needed there.
//   - The per-thread pointer setup (setup / preload) is what tt-metal's firmware does from the host's CB config,
//     for both pointers on every TRISC (see dataflow_buffer.h).

#include <cstdint>

#include "barrier.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "bmm_large_block_zm_fused_bias_activation.h"

constexpr DataFormat in_format      = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format     = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat dest_format    = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
constexpr DataFormat partial_format = dest_format; // mm_partials and bias: the intermediate format, as the host picks it

inline constexpr llk2::bmm_fused_config config {
    .rt_dim        = RT_DIM,
    .ct_dim        = CT_DIM,
    .kb_dim        = KB_DIM,
    .k_blocks      = K_BLOCKS,
    .out_blocks    = OUT_BLOCKS,
    .l1_acc        = L1_ACC,
    .fuse_bias     = FUSE_BIAS,
    .pack_relu     = PACK_RELU,
    .activation    = static_cast<llk2::VecFunction>(ACTIVATION),
    .activation_p0 = ACTIVATION_P0,
    .activation_p1 = ACTIVATION_P1,
};

constexpr std::uint32_t tile_datums     = llk2::default_tile_dim * llk2::default_tile_dim; // one 32x32 tile
constexpr std::uint32_t out_block_tiles = RT_DIM * CT_DIM;

// Pages of one tile. Ids as the tt-metal host names the CBs: c_0 in0, c_1 in1, c_2 bias, c_16 out, c_24 intermed0.
inline constexpr llk2::DataflowBufferSpec in0_buf {
    .id = 0, .format = in_format, .page_bytes = llk2::tile_bytes(in_format, tile_datums), .pages = OUT_BLOCKS * K_BLOCKS * RT_DIM * KB_DIM};
inline constexpr llk2::DataflowBufferSpec in1_buf {
    .id = 1, .format = in_format, .page_bytes = llk2::tile_bytes(in_format, tile_datums), .pages = OUT_BLOCKS * K_BLOCKS * KB_DIM * CT_DIM};
inline constexpr llk2::DataflowBufferSpec bias_buf {
    .id = 2, .format = partial_format, .page_bytes = llk2::tile_bytes(partial_format, tile_datums), .pages = CT_DIM};
inline constexpr llk2::DataflowBufferSpec out_buf {
    .id = 16, .format = out_format, .page_bytes = llk2::tile_bytes(out_format, tile_datums), .pages = OUT_BLOCKS * out_block_tiles};
inline constexpr llk2::DataflowBufferSpec partials_buf {
    .id = 24, .format = partial_format, .page_bytes = llk2::tile_bytes(partial_format, tile_datums), .pages = out_block_tiles};

/// TEST ONLY: what tt-metal's firmware and reader kernels do before the compute kernel runs (see the top of the file).
/// buffer_C holds the bias (CT_DIM tiles) and then the partials buffer's pages.
inline void harness_setup_buffers(RUNTIME_PARAMETERS params)
{
    llk2::preload_dataflow_buffer<in0_buf>(params.buffer_A[0], in0_buf.pages);
    llk2::preload_dataflow_buffer<in1_buf>(params.buffer_B[0], in1_buf.pages);
    llk2::preload_dataflow_buffer<bias_buf>(params.buffer_C[0], bias_buf.pages);
    llk2::setup_dataflow_buffer<partials_buf>(params.buffer_C[CT_DIM]);
    llk2::setup_dataflow_buffer<out_buf>(params.buffer_Res[0]);
    llk_barrier::rendezvous(llk_barrier::is_action_thread()); // every counter written before any thread reads one
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    harness_setup_buffers(params);

    llk2::bmm_large_block_zm_fused_bias_activation<config, dest_format, in0_buf, in1_buf, bias_buf, partials_buf, out_buf>();
}
