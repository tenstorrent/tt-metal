// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::read_value, a scalar from a dataflow buffer on all three threads (compute_api/sync.h), one
// kernel body for all three threads.
//
// The kernel runs ROUNDS rounds. Each round takes one tile of words from the control buffer (buffer C) and reads
// three of them: word 0, the round's number of tiles n (1..CT_DIM); word 1, whether the round subtracts (else it
// adds); and, as a 16-bit value at 16-bit index WIDE_INDEX, n again. It then takes n tiles of a and of b from their
// buffers, adds or subtracts them and packs n tiles. Every thread uses n: unpack waits for and pops n pages of each
// input and unpacks n tiles, math works on n tiles, pack reserves, packs and pushes n pages. A thread that read
// another value than the others packs other tiles than the host expects, or waits for pages that never come.
//
// The counts are in L1, written by the host: one build serves every sequence of them.
//
// What is emulated (TEST ONLY): as llk2_dfb_compute_api_test.cpp, no data movement core fills the buffers; the
// harness wrote every tile to L1 before the kernel starts and preload() marks them pushed.

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
#include "compute_api/eltwise_binary.h"
#include "compute_api/pack.h"
#include "compute_api/sync.h"
#include "test_dataflow_buffer.h" // TEST ONLY: tt-metal's DataflowBuffer, emulated

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t tile    = llk2::default_tile_dim;

inline constexpr auto& a_spec    = llk2::runtime_cols<llk2::l1_spec<tile, CT_DIM * tile, in_format>>; // up to CT_DIM tiles of a row
inline constexpr auto& out_spec  = llk2::runtime_cols<llk2::l1_spec<tile, CT_DIM * tile, out_format>>;
inline constexpr auto& ctrl_spec = llk2::l1_spec<tile, tile, DataFormat::UInt32>; // one tile of words
inline constexpr auto& acc_spec  = llk2::dst_spec<tile, CT_DIM * tile, llk2::kernel_dest_format>;

void run_kernel(RUNTIME_PARAMETERS params)
{
    llk2_test::DataflowBuffer in0(0, params.buffer_A[0], llk2::tile_bytes(in_format, llk2::default_tile_datums), TOTAL_TILES);
    llk2_test::DataflowBuffer in1(1, params.buffer_B[0], llk2::tile_bytes(in_format, llk2::default_tile_datums), TOTAL_TILES);
    llk2_test::DataflowBuffer ctrl(2, params.buffer_C[0], llk2::tile_bytes(ctrl_spec), ROUNDS);
    llk2_test::DataflowBuffer out(16, params.buffer_Res[0], llk2::tile_bytes(out_format, llk2::default_tile_datums), TOTAL_TILES);
    // TEST ONLY: the reader kernels' pushes, and the firmware's counter reset for the output.
    in0.preload(TOTAL_TILES);
    in1.preload(TOTAL_TILES);
    ctrl.preload(ROUNDS);
    out.zero_counters();

    using AccRing = llk2::DestRing<acc_spec>;
    for (std::uint32_t round = 0; round < ROUNDS; ++round)
    {
        const auto words         = llk2::wait_front<ctrl_spec>(ctrl);
        const std::uint32_t n    = llk2::read_value(words, 0);                         // the same n on unpack, math and pack
        const bool subtract      = llk2::read_value(words, 1) != 0;                    // the branch every thread takes
        const std::uint16_t same = llk2::read_value<std::uint16_t>(words, WIDE_INDEX); // a 16-bit word further into the tile
        llk2::release(words);
        const std::uint32_t count = n == same ? n : CT_DIM + 1; // a wrong word: a count no buffer holds

        const auto a    = llk2::wait_front<a_spec>(in0, count);
        const auto b    = llk2::wait_front<a_spec>(in1, count);
        const auto acc  = llk2::reserve_back<AccRing>();
        const auto slot = llk2::reserve_back<out_spec>(out, count);
        if (subtract)
        {
            llk2::pack(llk2::sub(a, b, acc), slot);
        }
        else
        {
            llk2::pack(llk2::add(a, b, acc), slot);
        }
        llk2::release(a);
        llk2::release(b);
        llk2::release(slot);
        llk2::release(acc);
    }
}
