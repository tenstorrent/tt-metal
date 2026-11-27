// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::add, llk2::sub and llk2::mul on Tensors, plain and with each broadcast, one kernel body for
// all three threads. Only Tensors, the Compute API (add, sub, mul, pack) and a Dest ring: no LLK, init or per-thread
// code.
//
// a is [RT_DIM, CT_DIM] tiles. OP: 0 add, 1 sub, 2 mul. BCAST (llk2::Broadcast): b has a's shape (None), or is one
// row of CT_DIM tiles (Row: a bias, its row 0 down every row of a's matching tile column), one column of RT_DIM tiles
// (Col: its column 0 across every column of a's matching tile row), or one tile (Scalar: its datum [0, 0] everywhere).
// DEST_SECTIONS: the kernel's Dest ring sections, as LLK2_DEST_SECTIONS (1 holds twice the tiles of 2).
// ACCUMULATE (mul only): y = (a + b) + a * b, the mul with .accumulate adding its product into the add's result in
// Dest (SDPA: alpha U_prev added into P V).
// VIEWS: 0 the op once on the whole tensors; 1 once per tile row, on llk2::row views of a, b and the output; 2 once
// per tile, on llk2::cols<1> / llk2::tile views. The same result either way (tensor/tensor.h, views).
// 3: once per half of a's tile columns, on llk2::cols<CT_DIM / 2> of a (and of b where b has a's columns): with
// several tile rows that is a window of a wider tile grid, its tile rows CT_DIM tiles apart. The output holds the left
// half's result, then the right half's, each a contiguous [RT_DIM, CT_DIM / 2] block (pack takes no window).

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
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format  = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format  = static_cast<DataFormat>(PACK_OUT);
constexpr llk2::Broadcast bcast  = static_cast<llk2::Broadcast>(BCAST);
constexpr std::uint32_t in1_rows = bcast == llk2::Broadcast::Row || bcast == llk2::Broadcast::Scalar ? 1 : RT_DIM; // tile rows of b
constexpr std::uint32_t in1_cols = bcast == llk2::Broadcast::Col || bcast == llk2::Broadcast::Scalar ? 1 : CT_DIM; // tile columns of b

inline constexpr auto& in0_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec = llk2::l1_spec<in1_rows * llk2::default_tile_dim, in1_cols * llk2::default_tile_dim, in1_format>;
inline constexpr auto& out_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec =
    llk2::dst_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, llk2::kernel_dest_format>; // a's shape in Dest

// The Resolver must pick the register formats the harness picked for the same test.
static_assert(llk2::resolved<llk2::add_op, in0_spec, in1_spec, llk2::add_modifiers {.bcast = bcast}>.plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert(llk2::resolved<llk2::add_op, in0_spec, in1_spec, llk2::add_modifiers {.bcast = bcast}>.plan.in1.writes == static_cast<DataFormat>(UNPACK_B_OUT));

/// y = a <OP> b into acc, for any a, b of matching shapes (whole tensors or views of them).
template <class A, class B, class Acc>
inline __attribute__((always_inline)) auto compute(const A& a, const B& b, const Acc& acc)
{
    if constexpr (OP == 2 && ACCUMULATE)
    {
        const auto sum = llk2::add<llk2::add_modifiers {.bcast = bcast}>(a, b, acc);           // acc = a + b
        return llk2::mul<llk2::mul_modifiers {.bcast = bcast, .accumulate = true}>(a, b, sum); // y = acc + a * b
    }
    else if constexpr (OP == 2)
    {
        return llk2::mul<llk2::mul_modifiers {.bcast = bcast}>(a, b, acc); // y = a * b
    }
    else if constexpr (OP == 1)
    {
        return llk2::sub<llk2::sub_modifiers {.bcast = bcast}>(a, b, acc); // y = a - b
    }
    else
    {
        return llk2::add<llk2::add_modifiers {.bcast = bcast}>(a, b, acc); // y = a + b
    }
}

/// The part of b that goes with W tile columns of a from column c0: those columns of b, or all of b when b has one
/// tile column (Col, Scalar).
template <std::uint32_t W, class B>
inline __attribute__((always_inline)) auto b_columns(const B& b, std::uint32_t c0)
{
    if constexpr (in1_cols == 1)
    {
        return b;
    }
    else
    {
        return llk2::cols<W>(b, c0);
    }
}

/// VIEWS == 3. Two windows of a, W = CT_DIM / 2 tile columns each: their tile rows step by a's CT_DIM tiles, not by
/// their own width. Window w's result goes to the output's w-th [RT_DIM, W] block, whose first tile is at
/// out_tiles[w * RT_DIM * W]. A template that returns void, so that only a VIEWS == 3 build instantiates its body:
/// run_kernel is not a template, and its discarded branches are still checked.
template <std::uint32_t W, class A, class B, class OutTiles>
inline __attribute__((always_inline)) void compute_windows(const A& a, const B& b, const OutTiles& out_tiles)
{
    static_assert(CT_DIM == 2 * W && RT_DIM > 1, "VIEWS == 3: two windows of CT_DIM / 2 tile columns of a tensor with several tile rows");
    using WindowRing = llk2::DestRing<llk2::dst_spec<RT_DIM * llk2::default_tile_dim, W * llk2::default_tile_dim, llk2::kernel_dest_format>>;
    for (std::uint32_t w = 0; w < 2; ++w)
    {
        const auto block = llk2::Tensor<llk2::l1_spec<RT_DIM * llk2::default_tile_dim, W * llk2::default_tile_dim, out_format>>(out_tiles[w * RT_DIM * W]);
        const auto y     = compute(llk2::cols<W>(a, w * W), b_columns<W>(b, w * W), llk2::reserve_back<WindowRing>());
        llk2::pack(y, block);
        llk2::release(y);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto a   = llk2::Tensor<in0_spec>(params.buffer_A[0]);
    const auto b   = llk2::Tensor<in1_spec>(params.buffer_B[0]);
    const auto out = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    if constexpr (VIEWS == 1)
    {
        // One tile row at a time. b has one tile row (Row, Scalar: every row of a reads it) or a's rows.
        using RowRing = llk2::DestRing<llk2::dst_spec<llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, llk2::kernel_dest_format>>;
        for (std::uint32_t r = 0; r < RT_DIM; ++r)
        {
            const auto y = compute(llk2::row(a, r), llk2::row(b, in1_rows == 1 ? 0 : r), llk2::reserve_back<RowRing>());
            llk2::pack(y, llk2::row(out, r));
            llk2::release(y);
        }
    }
    else if constexpr (VIEWS == 2)
    {
        // One tile at a time: b's tile is the one the broadcast reads for a's tile (r, c).
        using TileRing = llk2::DestRing<llk2::dst_spec<llk2::default_tile_dim, llk2::default_tile_dim, llk2::kernel_dest_format>>;
        for (std::uint32_t r = 0; r < RT_DIM; ++r)
        {
            for (std::uint32_t c = 0; c < CT_DIM; ++c)
            {
                const auto b_row = llk2::row(b, in1_rows == 1 ? 0 : r);
                const auto y     = compute(llk2::cols<1>(llk2::row(a, r), c), llk2::cols<1>(b_row, in1_cols == 1 ? 0 : c), llk2::reserve_back<TileRing>());
                llk2::pack(y, llk2::cols<1>(llk2::row(out, r), c));
                llk2::release(y);
            }
        }
    }
    else if constexpr (VIEWS == 3)
    {
        compute_windows<CT_DIM / 2>(a, b, params.buffer_Res);
    }
    else
    {
        using AccRing = llk2::DestRing<acc_spec>;
        const auto y  = compute(a, b, llk2::reserve_back<AccRing>());
        llk2::pack(y, out); // pack takes y from math
        llk2::release(y);
    }
}
