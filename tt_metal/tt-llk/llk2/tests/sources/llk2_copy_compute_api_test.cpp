// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::copy on Tensors, one kernel body for all three threads. Only Tensors, the Compute API
// (copy, pack) and a Dest ring: no LLK, init or per-thread code.
//
// The input is copied SECTIONS times, each time into the next Dest section (the Dest ring alternates them), and
// each round is packed to its own copy of the output, so both sections are checked.

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

constexpr DataFormat src_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr llk2::copy_modifiers mods {.transpose = TRANSPOSE};

inline constexpr auto& src_spec = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, src_format>;
inline constexpr auto& out_spec = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec =
    llk2::dst_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, llk2::kernel_dest_format>; // every format here is a B format

// The Resolver must make the choice the harness made for the same test: the register format, and whether the
// tiles go straight to Dest.
inline constexpr const llk2::Plan& plan = llk2::resolved<llk2::copy_op, src_spec, src_spec, mods>.plan;
static_assert(plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert((plan.in0.to == llk2::Where::Dest) == unpack_to_dest);

void run_kernel(RUNTIME_PARAMETERS params)
{
    using AccRing  = llk2::DestRing<acc_spec>;
    const auto src = llk2::Tensor<src_spec>(params.buffer_A[0]);
    for (std::uint32_t round = 0; round < SECTIONS; round++)
    {
        const auto out = llk2::Tensor<out_spec>(params.buffer_Res[round * TILES_Y * TILES_X]);
        const auto y   = llk2::copy<mods>(src, llk2::reserve_back<AccRing>()); // y = src, or src with each tile transposed
        llk2::pack(y, out);                                                    // pack takes y from math
        llk2::release(y);
    }
}
