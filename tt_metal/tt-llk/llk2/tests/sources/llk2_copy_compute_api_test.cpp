// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk::copy on Tensors, one kernel body for all three threads. Only Tensors, the Compute API
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
constexpr llk::copy_modifiers mods {.transpose = TRANSPOSE};

inline constexpr auto& src_spec = llk::l1_spec<TILES_Y * llk::default_tile_dim, TILES_X * llk::default_tile_dim, src_format>;
inline constexpr auto& out_spec = llk::l1_spec<TILES_Y * llk::default_tile_dim, TILES_X * llk::default_tile_dim, out_format>;
inline constexpr auto& acc_spec =
    llk::dst_spec<TILES_Y * llk::default_tile_dim, TILES_X * llk::default_tile_dim, llk::kernel_dest_format>; // every format here is a B format

// The Resolver must make the choice the harness made for the same test: the register format, and whether the
// tiles go straight to Dest.
inline constexpr const llk::Plan& plan = llk::resolved<llk::copy_op, llk::layout_of<decltype(src_spec)>, llk::layout_of<decltype(src_spec)>, mods>.plan;
static_assert(plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert((plan.in0.to == llk::Where::Dest) == unpack_to_dest);

void run_kernel(RUNTIME_PARAMETERS params)
{
    using AccRing  = llk::DestRing<decltype(acc_spec)>;
    const auto src = llk::Tensor(src_spec, params.buffer_A[0]);
    for (std::uint32_t round = 0; round < SECTIONS; round++)
    {
        const auto out = llk::Tensor(out_spec, params.buffer_Res[round * TILES_Y * TILES_X]);
        const auto y   = llk::copy<mods>(src, llk::reserve_back<AccRing>()); // y = src, or src with each tile transposed
        llk::pack(y, out);                                                    // pack takes y from math
        llk::release(y);
    }
}
