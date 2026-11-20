// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk::add and llk::sub on Tensors, plain and with a row broadcast, one kernel body for all three
// threads. Only Tensors, the Compute API (add, sub, pack) and a Dest ring: no LLK, init or per-thread code.
//
// a is [RT_DIM, CT_DIM] tiles. b has a's shape, or with BCAST_ROW is one row of CT_DIM tiles (a bias) whose row 0 is
// added to every row of a's matching tile column.

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
#include "compute_api/eltwise_binary.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format  = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format  = static_cast<DataFormat>(PACK_OUT);
constexpr llk::Broadcast bcast  = BCAST_ROW ? llk::Broadcast::Row : llk::Broadcast::None;
constexpr std::uint32_t in1_rows = BCAST_ROW ? 1 : RT_DIM; // tile rows of b

inline constexpr auto& in0_spec = llk::l1_spec<RT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec = llk::l1_spec<in1_rows * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, in1_format>;
inline constexpr auto& out_spec = llk::l1_spec<RT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, out_format>;
inline constexpr auto& acc_spec =
    llk::dst_spec<RT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, llk::kernel_dest_format>; // a's shape in Dest

// The Resolver must pick the register formats the harness picked for the same test.
static_assert(llk::resolved<llk::add_op, llk::layout_of<decltype(in0_spec)>, llk::layout_of<decltype(in1_spec)>, llk::add_modifiers {.bcast = bcast}>.plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert(llk::resolved<llk::add_op, llk::layout_of<decltype(in0_spec)>, llk::layout_of<decltype(in1_spec)>, llk::add_modifiers {.bcast = bcast}>.plan.in1.writes == static_cast<DataFormat>(UNPACK_B_OUT));

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto a   = llk::Tensor(in0_spec, params.buffer_A[0]);
    const auto b   = llk::Tensor(in1_spec, params.buffer_B[0]);
    const auto out = llk::Tensor(out_spec, params.buffer_Res[0]);

    using AccRing  = llk::DestRing<decltype(acc_spec)>;
    const auto acc = llk::reserve_back<AccRing>();
    const auto y   = [&]
    {
        if constexpr (IS_SUB)
        {
            return llk::sub<llk::sub_modifiers {.bcast = bcast}>(a, b, acc); // y = a - b
        }
        else
        {
            return llk::add<llk::add_modifiers {.bcast = bcast}>(a, b, acc); // y = a + b
        }
    }();
    llk::pack(y, out); // pack takes y from math
    llk::release(y);
}
