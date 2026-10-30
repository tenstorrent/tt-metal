// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: Tensors at dataflow buffer pointers feeding llk2::matmul, one kernel body for all three threads.
// The buffers are the harness's stand-in for tt-metal's DataflowBuffer (tests/helpers/test_dataflow_buffer.h). The
// kernel gets Tensors from them with compute_api/dataflow.h (wait_front, reserve_back) and gives the pages back with
// release(). A pointer is valid only on its owner (read: unpack, write: pack), which is where llk2 reads the Tensors'
// addresses.
//
// The kernel walks OUT_BLOCKS output blocks of [RT, CT] tiles; each sums K_BLOCKS products of an in0 block [RT, KB]
// and an in1 block [KB, CT] into Dest, then packs it into a block of the output buffer. The host laid the blocks out
// in the order the kernel takes them, as the reader kernels would; in1's blocks are repeated for every output block.
//
// What is emulated (TEST ONLY): no data movement core fills in0 and in1 or drains out. The harness wrote every
// input block to L1 before the kernel starts, and preload() marks them all pushed, so the buffers hold every block
// (capacity = all of them) and wait_front never waits. out holds every output block, so reserve_back never waits
// either. The pointer moves, the wrap at the end of each buffer, and the counter writes (STOREREG to the stream
// registers) are as tt-metal's.

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
#include "compute_api/dataflow.h"
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "test_dataflow_buffer.h" // TEST ONLY: tt-metal's DataflowBuffer, emulated

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

constexpr std::uint32_t in0_block_tiles = RT_DIM * KB_DIM;
constexpr std::uint32_t in1_block_tiles = KB_DIM * CT_DIM;
constexpr std::uint32_t out_block_tiles = RT_DIM * CT_DIM;

inline constexpr auto& x_spec   = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, KB_DIM * llk2::default_tile_dim, in0_format>;                // [RT, KB] tiles
inline constexpr auto& w_spec   = llk2::l1_spec<KB_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, in1_format>;                // [KB, CT] tiles
inline constexpr auto& out_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, out_format>;                // [RT, CT] tiles
inline constexpr auto& acc_spec = llk2::dst_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, llk2::kernel_dest_format>; // [M, N] in Dest

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Pages of one tile. Ids as tt-metal names its CBs: c_0 and c_1 for the inputs, c_16 for the output.
    llk2_test::DataflowBuffer in0(0, params.buffer_A[0], llk2::tile_bytes(in0_format, llk2::default_tile_datums), OUT_BLOCKS * K_BLOCKS * in0_block_tiles);
    llk2_test::DataflowBuffer in1(1, params.buffer_B[0], llk2::tile_bytes(in1_format, llk2::default_tile_datums), OUT_BLOCKS * K_BLOCKS * in1_block_tiles);
    llk2_test::DataflowBuffer out(16, params.buffer_Res[0], llk2::tile_bytes(out_format, llk2::default_tile_datums), OUT_BLOCKS * out_block_tiles);
    // TEST ONLY: the reader kernels' pushes, and the firmware's counter reset for the output.
    in0.preload(OUT_BLOCKS * K_BLOCKS * in0_block_tiles);
    in1.preload(OUT_BLOCKS * K_BLOCKS * in1_block_tiles);
    out.zero_counters();

    using AccRing = llk2::DestRing<acc_spec>;
    for (std::uint32_t block = 0; block < OUT_BLOCKS; ++block)
    {
        const auto acc = llk2::reserve_back<AccRing>(); // a zeroed Dest section, math's
        for (std::uint32_t k = 0; k < K_BLOCKS; ++k)
        {
            const auto x = llk2::wait_front<x_spec>(in0);
            const auto w = llk2::wait_front<w_spec>(in1);
            llk2::matmul(x, w, acc); // acc += x * w; still math's
            llk2::release(x);
            llk2::release(w);
        }
        llk2::release(llk2::pack(acc, llk2::reserve_back<out_spec>(out))); // pack takes acc from math, then pushes
        llk2::release(acc);                                                 // the section back to the ring
    }
}
