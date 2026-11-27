// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: two copies with different input formats, issued back to back ROUNDS times, so every op boundary
// on the unpack thread is a real reconfiguration (L1 and register formats, strides, and with a 32-bit Dest the
// switch between unpacking to SrcA and straight to Dest) queued behind the previous copy's unpacks. One kernel body
// for all three threads; only Tensors, the Compute API (copy, pack) and a Dest ring.
//
// Round r copies input A (even r) or input B (odd r) and packs it to its own copy of the output.

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

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/pack.h"

constexpr DataFormat a_format   = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat b_format   = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& a_spec   = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, a_format>;
inline constexpr auto& b_spec   = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, b_format>;
inline constexpr auto& out_spec = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, llk2::kernel_dest_format>;

// The two copies must differ on the unpack thread, or the test checks nothing new.
static_assert(a_format != b_format, "the two inputs must have different formats");
// With a 32-bit Dest the test expects exactly the 32-bit input to go straight to Dest (A_TO_DEST / B_TO_DEST).
static_assert((llk2::resolved<llk2::copy_op, a_spec, a_spec, llk2::copy_modifiers {}>.plan.in0.to == llk2::Where::Dest) == A_TO_DEST);
static_assert((llk2::resolved<llk2::copy_op, b_spec, b_spec, llk2::copy_modifiers {}>.plan.in0.to == llk2::Where::Dest) == B_TO_DEST);

void run_kernel(RUNTIME_PARAMETERS params)
{
    using AccRing = llk2::DestRing<acc_spec>;
    const auto a  = llk2::Tensor<a_spec>(params.buffer_A[0]);
    const auto b  = llk2::Tensor<b_spec>(params.buffer_B[0]);
    for (std::uint32_t round = 0; round < ROUNDS; round++)
    {
        const auto out = llk2::Tensor<out_spec>(params.buffer_Res[round * TILES_Y * TILES_X]);
        if (round % 2 == 0)
        {
            const auto y = llk2::copy(a, llk2::reserve_back<AccRing>());
            llk2::pack(y, out);
            llk2::release(y);
        }
        else
        {
            const auto y = llk2::copy(b, llk2::reserve_back<AccRing>());
            llk2::pack(y, out);
            llk2::release(y);
        }
    }
}
