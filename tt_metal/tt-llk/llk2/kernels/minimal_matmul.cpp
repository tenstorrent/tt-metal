// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's minimal_matmul compute kernel (experimental/minimal_matmul/device/kernels/compute_metal2.cpp) on LLK 2.0:
// Tensors, the Compute API and owners for every sync. One body for all three threads; no LLK calls, no init /
// uninit / reconfig, no tile_regs_* and no per-thread code. The reader and writer are the 1.0 kernel's, unchanged.
//
// Per output block (M_block_tiles x N_block_tiles tiles of the output, this core's rows and columns):
//   for each K block:
//     x = in0's [M_block, K_block] tiles, w = in1's [K_block, N_block] tiles
//     for each tile row of the block, in blocks of up to a Dest section (row_block tiles):
//       y = matmul(x's row, w's columns)                     in a Dest section
//       pack y to its place in the partials, adding to what is there from the second K block on (packer L1 acc)
//     The 1.0 kernel's subblocks are [subblock_h, subblock_w]; here they are one tile row high, because pack writes a
//     Dest tensor to a run of tiles that is contiguous in L1 (a part of one row of the partials), and the same number
//     of tiles wide as a Dest section holds. The host's subblock_h and subblock_w are not used.
//   then, for each tile row of the block, in blocks of up to a Dest section:
//     bias          y = partials + bias's row 0              (add, Row broadcast)
//     activation    y = gelu_tanh(y)                          on the pack thread
//     addcmul       y = a + b * (partials + bias)             as y = (a + b * bias) + b * partials: the first part is
//                                                             an add of a with the row b * bias (made once per block,
//                                                             into its own buffer), the second a mul by b's row
//                                                             added into the same Dest (mul .accumulate)
//     pack y into the output's row and push the row (the writer takes the block a row at a time)
//
// The 1.0 kernel's protocol that this one keeps:
//   - in0's block of the last K block stays in its buffer when another N block of the same rows follows: the reader
//     does not send it again, and the next N block's first K block reads it from there.
//   - The partials buffer holds exactly one output block, reserved before the K loop and pushed after it.
//   - The output is pushed a tile row (N_block_tiles tiles) at a time, M_block_tiles rows per block, the whole block
//     even where it is past the tensor's last row or column (the writer skips those tiles).
//   - The bias and the gate b come once per block (one tile row each), the residual a a tile row at a time.
// A subblock that reaches past the block's rows or columns of the tensor computes on what is in L1 there; the 1.0
// kernel does the same for its last subblock, and the writer drops those tiles.
//
// What this file takes from tt-metal beyond what the 1.0 kernel does (the LLK 2.0 branch of
// minimal_matmul_program_descriptor.cpp, TT_MINIMAL_MATMUL_LLK2):
//   - Each buffer's data format as the named compile-time arg <buffer>_format, the same on all three TRISCs.
//   - The activation as the named compile-time arg llk2_activation (0: none, 1: gelu_tanh).
//   - For addcmul, a buffer of one tile row for b * bias (scaled_bias), and a scalar of 1.0.
//   - The Dest mode and the math fidelity as the defines LLK2_DEST_FP32 and LLK2_MATH_FIDELITY.
//
// What the 1.0 kernel does that this one does not: SwiGLU, no bias, a full (not broadcast) gate b, a scalar other
// than 1.0, activations other than gelu_tanh. The host refuses them on this path.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "compute_api/copy.h"
#include "compute_api/dataflow.h"
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/sync.h"
#include "compute_api/vec.h"
#include "experimental/kernel_args.h"
#include "kernel_args.h"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

namespace
{

using namespace llk2;

constexpr std::uint32_t K_blocks          = get_named_compile_time_arg_val("K_num_blocks");
constexpr std::uint32_t M_block_tiles     = get_named_compile_time_arg_val("M_block_tiles");
constexpr std::uint32_t K_block_tiles     = get_named_compile_time_arg_val("K_block_tiles");
constexpr std::uint32_t N_block_tiles     = get_named_compile_time_arg_val("N_block_tiles");
constexpr std::uint32_t M_cores           = get_named_compile_time_arg_val("M_cores");
constexpr std::uint32_t N_blocks_per_core = get_named_compile_time_arg_val("N_blocks_per_core");
constexpr bool gelu_tanh                  = get_named_compile_time_arg_val("llk2_activation") == 1;
#ifdef FUSE_TERNARY
constexpr bool addcmul = true;
#else
constexpr bool addcmul = false;
#endif

#ifndef LLK2_MM_GELU_THREAD
#define LLK2_MM_GELU_THREAD Thread::Pack
#endif

constexpr std::uint32_t tile = default_tile_dim;

// The blocks along a tile row, for the matmul and the epilogue: the largest divisor of N_block_tiles that fits a Dest
// section.
constexpr std::uint32_t row_block_of(std::uint32_t n, std::uint32_t cap)
{
    std::uint32_t b = n < cap ? n : cap;
    while (n % b != 0)
    {
        --b;
    }
    return b;
}

constexpr std::uint32_t row_block = row_block_of(N_block_tiles, dest_section_tiles);

constexpr const auto& x_spec           = l1_spec<M_block_tiles * tile, K_block_tiles * tile, LLK2_CB_FORMAT("in0")>;
constexpr const auto& w_spec           = l1_spec<K_block_tiles * tile, N_block_tiles * tile, LLK2_CB_FORMAT("in1")>;
constexpr const auto& partials_spec    = l1_spec<M_block_tiles * tile, N_block_tiles * tile, LLK2_CB_FORMAT("intermediate")>;
constexpr const auto& bias_spec        = l1_spec<tile, N_block_tiles * tile, LLK2_CB_FORMAT("in2")>;       // row 0 of each tile
constexpr const auto& out_row_spec     = l1_spec<tile, N_block_tiles * tile, LLK2_CB_FORMAT("out")>;       // one tile row
constexpr const auto& a_row_spec       = l1_spec<tile, N_block_tiles * tile, LLK2_CB_FORMAT("ternary_a")>; // one tile row
constexpr const auto& b_spec           = l1_spec<tile, N_block_tiles * tile, LLK2_CB_FORMAT("ternary_b")>; // row 0 of each tile
constexpr const auto& scaled_bias_spec = l1_spec<tile, N_block_tiles * tile, LLK2_CB_FORMAT("scaled_bias")>;

using RowRing = DestRing<dst_spec<tile, row_block * tile, kernel_dest_format>>; // row_block tiles of one tile row

constexpr add_modifiers plus_row {.bcast = Broadcast::Row};
constexpr mul_modifiers times_row {.bcast = Broadcast::Row};
constexpr mul_modifiers plus_times_row {.bcast = Broadcast::Row, .accumulate = true};
constexpr pack_modifiers accumulated {.l1_acc = Toggle::Runtime};

/// A full block of row_block tile columns: cols_of below takes it as compile-time width.
struct FullBlock
{
};

/// The columns [c0, c0 + n) of t: n is row_block (FullBlock) or a run-time 1..row_block.
template <class T>
inline __attribute__((always_inline)) auto cols_of(const T& t, std::uint32_t c0, FullBlock)
{
    return cols<row_block>(t, c0);
}

template <class T>
inline __attribute__((always_inline)) auto cols_of(const T& t, std::uint32_t c0, std::uint32_t n)
{
    return cols_up_to<row_block>(t, c0, n);
}

/// f(c0, n) for the blocks of row_block tile columns of [0, n_cols): the full blocks with n a FullBlock, then the last,
/// narrower one (where n_cols is not a multiple of row_block) with n its run-time width. The columns past n_cols are
/// past the tensor's (the writer drops them), so nothing computes them.
template <class F>
inline __attribute__((always_inline)) void for_row_blocks(std::uint32_t n_cols, const F& f)
{
    std::uint32_t c0 = 0;
    for (; c0 + row_block <= n_cols; c0 += row_block)
    {
        f(c0, FullBlock {});
    }
    if (c0 < n_cols)
    {
        f(c0, n_cols - c0);
    }
}

/// The activation on y, on the pack thread (math moves on while pack runs it); y itself without one.
template <class Y>
inline __attribute__((always_inline)) auto activation(const Y& y)
{
    if constexpr (gelu_tanh)
    {
        return vec_gelu_tanh<vec_modifiers {.thread = LLK2_MM_GELU_THREAD}>(y);
    }
    else
    {
        return y;
    }
}

/// One K block of the output block: the block's rows [0, m_rows) and columns [0, n_cols), row_block tiles of a row at
/// a time, packed into the partials, added to them from the second K block on.
template <class X, class W, class Slot>
inline __attribute__((always_inline)) void k_block_of(const X& x, const W& w, const Slot& slot, std::uint32_t m_rows, std::uint32_t n_cols, bool add)
{
    for (std::uint32_t r = 0; r < m_rows; ++r)
    {
        for_row_blocks(
            n_cols,
            [&](std::uint32_t c0, auto n) __attribute__((always_inline))
            {
                const auto y = matmul(row(x, r), cols_of(w, c0, n), reserve_back<RowRing>());
                pack<accumulated>(y, cols_of(row(slot, r), c0, n), add);
                release(y);
            });
    }
}

} // namespace

void kernel_main()
{
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);
    DataflowBuffer intermediate(dfb::intermediate);
    DataflowBuffer in2(dfb::in2);
#ifdef FUSE_TERNARY
    DataflowBuffer ternary_a(dfb::ternary_a);
    DataflowBuffer ternary_b(dfb::ternary_b);
    DataflowBuffer scaled_bias(dfb::scaled_bias);
#endif

    const std::uint32_t in0_idx      = get_arg(args::in0_idx);
    const std::uint32_t N_start_tile = get_arg(args::N_start_tile);
    const std::uint32_t N_end_tile   = get_arg(args::N_end_tile);
    const std::uint32_t M_tiles      = get_arg(args::M_tiles);
    const auto split                 = ttnn::runtime_length::uniform_split(M_tiles, M_cores, in0_idx, M_block_tiles);

    for (std::uint32_t m_block = 0; m_block < split.blocks_per_core; ++m_block)
    {
        const std::uint32_t m_tile = split.start + m_block * M_block_tiles;
        const std::uint32_t m_end  = m_tile + M_block_tiles < split.end ? m_tile + M_block_tiles : split.end;
        const std::uint32_t m_rows = m_end - m_tile; // the block's rows of this core's share

        for (std::uint32_t n_block = 0; n_block < N_blocks_per_core; ++n_block)
        {
            const std::uint32_t n_tile = N_start_tile + n_block * N_block_tiles;
            const std::uint32_t n_end  = n_tile + N_block_tiles < N_end_tile ? n_tile + N_block_tiles : N_end_tile;
            const std::uint32_t n_cols = n_end - n_tile;
            const bool keep_in0        = n_block + 1 < N_blocks_per_core; // the next N block reads the last K block's in0

            const auto slot = reserve_back<partials_spec>(intermediate);
            for (std::uint32_t k = 0; k < K_blocks; ++k)
            {
                const auto x = peek_front<x_spec>(in0);
                const auto w = wait_front<w_spec>(in1);
                k_block_of(x, w, slot, m_rows, n_cols, k > 0);
                release(w);
                if (!(k + 1 == K_blocks && keep_in0))
                {
                    pop_front<x_spec>(in0);
                }
            }
            release(slot);

            const auto partials = wait_front<partials_spec>(intermediate);
            const auto bias     = wait_front<bias_spec>(in2);
            if constexpr (addcmul)
            {
#ifdef FUSE_TERNARY
                const auto b = wait_front<b_spec>(ternary_b);
                {
                    // b * bias, once per block: row 0 of each tile is what the Row broadcasts below read.
                    const auto bb_slot = reserve_back<scaled_bias_spec>(scaled_bias);
                    for_row_blocks(
                        n_cols,
                        [&](std::uint32_t c0, auto n) __attribute__((always_inline))
                        {
                            const auto y = mul(cols_of(b, c0, n), cols_of(bias, c0, n), reserve_back<RowRing>());
                            pack(y, cols_of(bb_slot, c0, n));
                            release(y);
                        });
                    release(bb_slot);
                }
                const auto bb = wait_front<scaled_bias_spec>(scaled_bias);
                for (std::uint32_t m = 0; m < M_block_tiles; ++m)
                {
                    const auto a     = wait_front<a_row_spec>(ternary_a);
                    const auto o_row = reserve_back<out_row_spec>(out);
                    if (m < m_rows)
                    {
                        for_row_blocks(
                            n_cols,
                            [&](std::uint32_t c0, auto n) __attribute__((always_inline))
                            {
                                const auto ab = add<plus_row>(cols_of(a, c0, n), cols_of(bb, c0, n), reserve_back<RowRing>());
                                const auto y  = mul<plus_times_row>(cols_of(row(partials, m), c0, n), cols_of(b, c0, n), ab);
                                pack(y, cols_of(o_row, c0, n));
                                release(y);
                            });
                    }
                    release(o_row);
                    release(a);
                }
                release(bb);
                release(b);
#endif
            }
            else
            {
                for (std::uint32_t m = 0; m < M_block_tiles; ++m)
                {
                    const auto o_row = reserve_back<out_row_spec>(out);
                    if (m < m_rows)
                    {
                        for_row_blocks(
                            n_cols,
                            [&](std::uint32_t c0, auto n) __attribute__((always_inline))
                            {
                                const auto y = add<plus_row>(cols_of(row(partials, m), c0, n), cols_of(bias, c0, n), reserve_back<RowRing>());
                                const auto z = activation(y);
                                pack(z, cols_of(o_row, c0, n));
                                release(z);
                            });
                    }
                    release(o_row);
                }
            }
            release(bias);
            release(partials);
        }
    }
}
