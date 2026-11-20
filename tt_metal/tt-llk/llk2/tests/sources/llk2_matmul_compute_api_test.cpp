// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk::matmul on Tensors, one kernel body for all three threads. Only Tensors, the Compute API
// (matmul, pack) and a Dest ring: no LLK, init or per-thread code.

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
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& in0_spec = llk::l1_spec<RT_DIM * llk::default_tile_dim, KT_DIM * llk::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec = llk::l1_spec<KT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, in1_format>;
inline constexpr auto& out_spec = llk::l1_spec<RT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk::dst_spec<RT_DIM * llk::default_tile_dim, CT_DIM * llk::default_tile_dim, llk::kernel_dest_format>; // [M, N] in Dest

// The Resolver must pick the register formats the harness picked for the same test.
static_assert(llk::resolved<llk::matmul_op, llk::layout_of<decltype(in0_spec)>, llk::layout_of<decltype(in1_spec)>, llk::matmul_modifiers {}>.plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert(llk::resolved<llk::matmul_op, llk::layout_of<decltype(in0_spec)>, llk::layout_of<decltype(in1_spec)>, llk::matmul_modifiers {}>.plan.in1.writes == static_cast<DataFormat>(UNPACK_B_OUT));
// And the packer's input format, except where pack starts from the Dest format and the harness from the unpacker's
// (a block-float input with a 16-bit Dest; see resolver/pack.h).
constexpr DataFormat pack_in = llk::resolved<llk::pack_op, llk::layout_of<decltype(acc_spec)>, llk::layout_of<decltype(out_spec)>, llk::pack_modifiers {}>.plan.pack.in_format;
static_assert(pack_in == static_cast<DataFormat>(PACK_IN) || (llk::is_bfp(static_cast<DataFormat>(PACK_IN)) && pack_in == llk::kernel_dest_format));

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto x   = llk::Tensor(in0_spec, params.buffer_A[0]);
    const auto w   = llk::Tensor(in1_spec, params.buffer_B[0]);
    const auto out = llk::Tensor(out_spec, params.buffer_Res[0]);

    using AccRing  = llk::DestRing<decltype(acc_spec)>;
    const auto y   = llk::matmul(x, w, llk::reserve_back<AccRing>()); // y = 0 + x * w, math's
    llk::pack(y, out);                                                  // pack takes y from math
    llk::release(y);
}
