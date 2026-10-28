// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: Tensors from dataflow buffers (tensor/dataflow_buffer.h) feeding llk2::matmul, one kernel body
// for all three threads.
//
// The kernel walks OUT_BLOCKS output blocks of [RT, CT] tiles; each sums K_BLOCKS products of an in0 block [RT, KB]
// and an in1 block [KB, CT], taken from their buffers with wait_front, into Dest, then packs it into a block of the
// output buffer taken with reserve_back. The host laid the blocks out in the order the kernel takes them, as the
// reader kernels would; in1's blocks are repeated for every output block.
//
// What is emulated: no data movement core fills in0_buf and in1_buf or drains out_buf. The harness wrote every
// input block to L1 before the kernel starts, and preload_dataflow_buffer marks them all pushed, so the buffers
// hold every block (capacity = all of them) and wait_front never waits. out_buf holds every output block, so
// reserve_back never waits either. The pointer moves, the wrap at the end of each buffer, and the counter writes
// (STOREREG to the stream registers) are the real ones.
//
// Dest sync and pack use the LLK 2.0 Dest sync API (compute_api/dest.h) and pack (compute_api/pack.h).

#include <cstdint>

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
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "tensor/dataflow_buffer.h"

constexpr DataFormat in0_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format  = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format  = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;

constexpr std::uint32_t tile_datums     = llk2::default_tile_dim * llk2::default_tile_dim; // one 32x32 tile
constexpr std::uint32_t in0_block_tiles = RT_DIM * KB_DIM;
constexpr std::uint32_t in1_block_tiles = KB_DIM * CT_DIM;
constexpr std::uint32_t out_block_tiles = RT_DIM * CT_DIM;

// Pages of one tile, a capacity of every block the kernel takes (see "What is emulated" above).
// Ids as tt-metal names its CBs: c_0 and c_1 for the inputs, c_16 for the output.
inline constexpr llk2::DataflowBufferSpec in0_buf {
    .id = 0, .format = in0_format, .page_bytes = llk2::tile_bytes(in0_format, tile_datums), .pages = OUT_BLOCKS * K_BLOCKS * in0_block_tiles};
inline constexpr llk2::DataflowBufferSpec in1_buf {
    .id = 1, .format = in1_format, .page_bytes = llk2::tile_bytes(in1_format, tile_datums), .pages = OUT_BLOCKS * K_BLOCKS * in1_block_tiles};
inline constexpr llk2::DataflowBufferSpec out_buf {
    .id = 16, .format = out_format, .page_bytes = llk2::tile_bytes(out_format, tile_datums), .pages = OUT_BLOCKS * out_block_tiles};

inline constexpr auto& acc_spec = llk2::dst_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, dest_format>; // [M, N] in Dest

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Test only: the harness has no data movement cores (see "What is emulated" above).
    llk2::preload_dataflow_buffer<in0_buf>(params.buffer_A[0], in0_buf.pages);
    llk2::preload_dataflow_buffer<in1_buf>(params.buffer_B[0], in1_buf.pages);
    llk2::setup_dataflow_buffer<out_buf>(params.buffer_Res[0]);

    for (std::uint32_t block = 0; block < OUT_BLOCKS; ++block)
    {
        const auto acc = llk2::dest_acquire<acc_spec>(); // a zeroed Dest section
        for (std::uint32_t k = 0; k < K_BLOCKS; ++k)
        {
            const auto x = llk2::wait_front<in0_buf, RT_DIM, KB_DIM>(); // [RT, KB] tiles
            const auto w = llk2::wait_front<in1_buf, KB_DIM, CT_DIM>(); // [KB, CT] tiles
            llk2::matmul(x, w, acc);                                    // acc += x * w
            llk2::release(x);
            llk2::release(w);
        }
        llk2::dest_commit(acc);

        const auto out = llk2::reserve_back<out_buf, RT_DIM, CT_DIM>(); // [RT, CT] tiles
        llk2::pack(acc, out);
        llk2::release(out);
        llk2::dest_release(acc);
    }
}
