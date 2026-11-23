// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: one Dest ring, three Dest tensor shapes (compute_api/dest.h). One kernel body for all three
// threads. Each iteration i, as SDPA's inner loop mixes a score row, an output row and a statistics tile:
//   y = matmul(x_i, w)              [1 x MM_CT] tiles   x_i: tile row i of A, [1 x KT_DIM]; w: all of B, [KT_DIM x MM_CT]
//   z = add(x_i[:ADD_CT], B_i)      [1 x ADD_CT] tiles  B_i: the first ADD_CT tiles of B's tile row i
//   c = copy(B[i, MM_CT - 1])       [1 x 1] tile
// each packed to its own place in the output: iteration i's MM_CT + ADD_CT + 1 tiles, in that order.
//
// Every Dest tensor comes from reserve_back on a view of the one ring (DestRing<mm_spec>, DestRing<add_spec>,
// DestRing<one_spec>). Three reserves per iteration, so each shape lands in every section over the iterations. With
// INTERLEAVE, y and z are held at once (two sections, both reserved before either is released); else every tensor is
// reserved, written, packed and released before the next. DEST_SECTIONS sets the kernel's ring (LLK2_DEST_SECTIONS).

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
#define LLK2_DEST_SECTIONS DEST_SECTIONS
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t tile    = llk2::default_tile_dim;

static_assert(ADD_CT_DIM <= KT_DIM && ITERATIONS <= KT_DIM, "the add reads x_i's first tiles and B's tile row i");
static_assert(!INTERLEAVE || DEST_SECTIONS == 2, "holding two Dest tensors at once takes two sections");

inline constexpr auto& x_spec     = llk2::l1_spec<tile, KT_DIM * tile, in0_format>;             // x_i, [1, KT]
inline constexpr auto& w_spec     = llk2::l1_spec<KT_DIM * tile, MM_CT_DIM * tile, in1_format>; // w, [KT, MM_CT]
inline constexpr auto& a_spec     = llk2::l1_spec<tile, ADD_CT_DIM * tile, in0_format>;         // x_i's first ADD_CT tiles
inline constexpr auto& b_spec     = llk2::l1_spec<tile, ADD_CT_DIM * tile, in1_format>;         // B_i
inline constexpr auto& c_spec     = llk2::l1_spec<tile, tile, in1_format>;                      // B[i, MM_CT - 1]
inline constexpr auto& mm_out     = llk2::l1_spec<tile, MM_CT_DIM * tile, out_format>;
inline constexpr auto& add_out    = llk2::l1_spec<tile, ADD_CT_DIM * tile, out_format>;
inline constexpr auto& one_out    = llk2::l1_spec<tile, tile, out_format>;
inline constexpr auto& mm_spec    = llk2::dst_spec<tile, MM_CT_DIM * tile, llk2::kernel_dest_format>;  // [1 x MM_CT]: the score row
inline constexpr auto& add_spec   = llk2::dst_spec<tile, ADD_CT_DIM * tile, llk2::kernel_dest_format>; // [1 x ADD_CT]: the output row
inline constexpr auto& one_spec   = llk2::dst_spec<tile, tile, llk2::kernel_dest_format>;              // [1 x 1]: a statistics tile
constexpr std::uint32_t out_tiles = MM_CT_DIM + ADD_CT_DIM + 1;                                        // per iteration

void run_kernel(RUNTIME_PARAMETERS params)
{
    using MmRing  = llk2::DestRing<mm_spec>; // three views of the kernel's one Dest ring
    using AddRing = llk2::DestRing<add_spec>;
    using OneRing = llk2::DestRing<one_spec>;

    const auto w = llk2::Tensor<w_spec>(params.buffer_B[0]);
    for (std::uint32_t i = 0; i < ITERATIONS; ++i)
    {
        const auto x               = llk2::Tensor<x_spec>(params.buffer_A[i * KT_DIM]);
        const auto a               = llk2::Tensor<a_spec>(params.buffer_A[i * KT_DIM]);
        const auto b               = llk2::Tensor<b_spec>(params.buffer_B[i * MM_CT_DIM]);
        const auto c               = llk2::Tensor<c_spec>(params.buffer_B[i * MM_CT_DIM + MM_CT_DIM - 1]);
        const std::uint32_t out_at = i * out_tiles;

        if constexpr (INTERLEAVE)
        {
            const auto y = llk2::matmul(x, w, llk2::reserve_back<MmRing>()); // one section
            const auto z = llk2::add(a, b, llk2::reserve_back<AddRing>());   // the other, while y is still held
            llk2::pack(y, llk2::Tensor<mm_out>(params.buffer_Res[out_at]));
            llk2::release(y);
            llk2::pack(z, llk2::Tensor<add_out>(params.buffer_Res[out_at + MM_CT_DIM]));
            llk2::release(z);
        }
        else
        {
            const auto y = llk2::matmul(x, w, llk2::reserve_back<MmRing>());
            llk2::pack(y, llk2::Tensor<mm_out>(params.buffer_Res[out_at]));
            llk2::release(y);
            const auto z = llk2::add(a, b, llk2::reserve_back<AddRing>());
            llk2::pack(z, llk2::Tensor<add_out>(params.buffer_Res[out_at + MM_CT_DIM]));
            llk2::release(z);
        }
        const auto s = llk2::copy(c, llk2::reserve_back<OneRing>());
        llk2::pack(s, llk2::Tensor<one_out>(params.buffer_Res[out_at + MM_CT_DIM + ADD_CT_DIM]));
        llk2::release(s);
    }
}
