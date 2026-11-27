// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::pack with .l1_acc behind a copy, one kernel body for all three threads: a tensor added onto
// another in L1, x += y, where y has a format of its own (block-float or Float16_b) and x is in the output format.
//
// x (buffer A, [1, TILES] tiles) is first copied to the output, so the output holds x in the output format.
// Then y (buffer B, the same grid, its own format) is copied to Dest and packed with .l1_acc onto the output:
// out = x + y.

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

constexpr DataFormat x_format   = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat y_format   = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t tile    = llk2::default_tile_dim;

inline constexpr auto& x_spec   = llk2::l1_spec<tile, TILES * tile, x_format>;
inline constexpr auto& y_spec   = llk2::l1_spec<tile, TILES * tile, y_format>;
inline constexpr auto& out_spec = llk2::l1_spec<tile, TILES * tile, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<tile, TILES * tile, llk2::kernel_dest_format>;
constexpr llk2::pack_modifiers added {.l1_acc = llk2::Toggle::On};

void run_kernel(RUNTIME_PARAMETERS params)
{
    using AccRing  = llk2::DestRing<acc_spec>;
    const auto x   = llk2::Tensor<x_spec>(params.buffer_A[0]);
    const auto y   = llk2::Tensor<y_spec>(params.buffer_B[0]);
    const auto out = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    const auto first = llk2::copy(x, llk2::reserve_back<AccRing>());
    llk2::pack(first, out); // out = x
    llk2::release(first);

    const auto second = llk2::copy(y, llk2::reserve_back<AccRing>());
    llk2::pack<added>(second, out); // out += y, added in L1
    llk2::release(second);
}
