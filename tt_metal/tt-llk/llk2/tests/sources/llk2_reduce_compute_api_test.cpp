// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::reduce on Tensors, a row max or row sum, unseeded or seeded with a previous result, one
// kernel body for all three threads. Only Tensors, the Compute API (reduce, pack) and a Dest ring: no LLK, init or
// per-thread code.
//
// s (buffer A) is [RT_DIM, CT_DIM] tiles; every row is reduced over all CT_DIM tiles of its tile row into column 0 of
// that tile row's output tile. The scaler (buffer B) is one tile of 1.0. With SEEDED, buffer C is the previous result
// ([RT_DIM, 1] tiles, its values in column 0) the reduction folds into: SDPA's m_new = max(m_prev, rowmax(S)).
//
// With PARENT_CT > CT_DIM, buffer A is [RT_DIM, PARENT_CT] tiles and s is a view of it: CT_DIM tile columns from
// column WINDOW_COL0 (llk2::cols), a window whose tile rows are PARENT_CT tiles apart. SDPA's narrow last K chunk:
// the row max over the valid columns of a score buffer as wide as a full chunk.

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
#include "compute_api/dest.h"
#include "compute_api/pack.h"
#include "compute_api/reduce.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr llk2::reduce_modifiers mods {.kind = IS_SUM ? llk2::ReduceKind::Sum : llk2::ReduceKind::Max};

inline constexpr auto& parent_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, PARENT_CT * llk2::default_tile_dim, in_format>;
inline constexpr auto& scaler_spec = llk2::l1_spec<llk2::default_tile_dim, llk2::default_tile_dim, in_format>;
inline constexpr auto& seed_spec   = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, llk2::default_tile_dim, in_format>;
inline constexpr auto& out_spec    = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec    = llk2::dst_spec<RT_DIM * llk2::default_tile_dim, llk2::default_tile_dim, llk2::kernel_dest_format>;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto s      = llk2::cols<CT_DIM>(llk2::Tensor<parent_spec>(params.buffer_A[0]), WINDOW_COL0); // the whole tensor when PARENT_CT == CT_DIM
    const auto scaler = llk2::Tensor<scaler_spec>(params.buffer_B[0]);
    const auto out    = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    using AccRing  = llk2::DestRing<acc_spec>;
    const auto acc = llk2::reserve_back<AccRing>();
    const auto y   = [&]
    {
        if constexpr (SEEDED)
        {
            const auto seed = llk2::Tensor<seed_spec>(params.buffer_C[0]);
            return llk2::reduce<mods>(s, scaler, seed, acc); // y = reduce(seed, rows of s)
        }
        else
        {
            return llk2::reduce<mods>(s, scaler, acc); // y = reduce(rows of s)
        }
    }();
    llk2::pack(y, out); // pack takes y from math
    llk2::release(y);
}
