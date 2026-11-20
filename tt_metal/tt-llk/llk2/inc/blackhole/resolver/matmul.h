// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's OpApi on Blackhole. in0 [M, K] unpacks to SrcB, in1 [K, N] to SrcA, as tt-llk does; the FPU accumulates
// rt_dim x ct_dim output tiles in Dest over kt_dim K steps.

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/formats.h"
#include "resolver/resolver.h"

namespace llk
{

/// The Dest format matmul accumulates into: Int32 for integer inputs (needs the 32-bit Dest mode), else FP32 or BF16.
constexpr DataFormat matmul_dest_format(DataFormat in, bool dest_fp32)
{
    if (is_integer_format(in))
    {
        return DataFormat::Int32;
    }
    return dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

inline constexpr std::uint32_t matmul_tile_dim = default_tile_dim; ///< matmul is ported for 32x32 tiles only

/// accepts: x [M, K] and w [K, N] in L1 in 32x32 tiles, formats the unpacker can write to Src (both float or both
/// integer; integers need the 32-bit Dest), no transpose yet.
constexpr bool matmul_accepts(const TileLayout& x, const TileLayout& w, const matmul_modifiers& m, const KernelSettings& s)
{
    const bool in_l1      = x.storage() == Storage::L1 && w.storage() == Storage::L1;
    const bool k_matches  = x.size(1) == w.size(0) && (x.size(3) * x.size(5)) == (w.size(2) * w.size(4));
    const bool full_tiles = (x.size(2) * x.size(4)) == matmul_tile_dim && (x.size(3) * x.size(5)) == matmul_tile_dim && (w.size(3) * w.size(5)) == matmul_tile_dim;
    const bool to_src = src_register_format(x.format(), s.dest_fp32) != DataFormat::Invalid && src_register_format(w.format(), s.dest_fp32) != DataFormat::Invalid;
    const bool same_kind   = is_integer_format(x.format()) == is_integer_format(w.format());
    const bool int_dest_ok = !is_integer_format(x.format()) || s.dest_fp32;
    return in_l1 && k_matches && full_tiles && to_src && same_kind && int_dest_ok && !m.transpose_in1;
}

/// options: x to SrcB, w to SrcA (as tt-llk), the kernel's fidelity, the [M, N] result in Dest.
constexpr OpConfig matmul_options(const TileLayout& x, const TileLayout& w, const matmul_modifiers&, const KernelSettings& s)
{
    return {
        OwnerExu::Fpu,
        {Where::SrcB, src_register_format(x.format(), s.dest_fp32)},
        {Where::SrcA, src_register_format(w.format(), s.dest_fp32)},
        s.math_fidelity,
        laid_out({x.size(0), w.size(1), x.size(2), w.size(3), x.size(4), w.size(5)}, matmul_dest_format(x.format(), s.dest_fp32), Storage::Dest), // rows from x, columns from w
    };
}

/// work: the output tile grid, K tiles, and x's tile geometry.
constexpr Work matmul_work(const TileLayout& x, const TileLayout&, const OpConfig& c)
{
    return {
        static_cast<std::uint32_t>(c.result.size(0)),
        static_cast<std::uint32_t>(c.result.size(1)),
        static_cast<std::uint32_t>(x.size(1)),
        static_cast<std::uint32_t>(x.size(4)),
        static_cast<std::uint32_t>(x.size(2) * x.size(3))};
}

/// Whether x is the held (reused) operand. A K step multiplies x's tile column k (rt_dim tiles) by w's tile row k
/// (ct_dim tiles). The longer output dimension is reused: when ct_dim >= rt_dim, each x tile is held in SrcB for one
/// run while the ct_dim w tiles of row k stream through SrcA; otherwise each w tile is held in SrcA while the rt_dim
/// x tiles of column k stream through SrcB. x is [rt_dim, kt_dim] tiles and w [kt_dim, ct_dim], both row-major, so
/// moving down x's column is kt_dim tiles and moving along w's row is 1 tile.
constexpr bool matmul_reuses_x(const Work& k)
{
    return k.ct_dim >= k.rt_dim;
}

/// plan: the unpack walk described at matmul_reuses_x, and math on in1's (w's) Src format.
constexpr Plan matmul_plan(const TileLayout& x, const TileLayout& w, const matmul_modifiers& m, const OpConfig& c, const Work& k)
{
    const bool reuse_x          = matmul_reuses_x(k);
    const std::uint32_t x_down  = k.kt_dim; // tiles from x(r, k) to x(r + 1, k)
    const std::uint32_t x_right = 1;        // tiles from x(r, k) to x(r, k + 1)
    const std::uint32_t w_right = 1;        // tiles from w(k, c) to w(k, c + 1)
    const std::uint32_t w_down  = k.ct_dim; // tiles from w(k, c) to w(k + 1, c)

    const UnpackPlan x_plan {
        Where::SrcB,
        x.format(),
        c.in0.format,
        k.face_r_dim,
        k.num_faces,
        tile_bytes(x.format(), x.tile_elements()),
        Layout::Tilized,
        Broadcast::None,
        false /* x is not transposed */,
        reuse_x ? 0 : x_down, // tile_step: streams down its column when w is the held one
        reuse_x ? x_down : 0, // run_step: next held tile, one row down
        x_right,              // step_step: next K column
    };
    const UnpackPlan w_plan {
        Where::SrcA,
        w.format(),
        c.in1.format,
        w.size(4),
        (w.size(2) * w.size(3)),
        tile_bytes(w.format(), w.tile_elements()),
        Layout::Tilized,
        Broadcast::None,
        m.transpose_in1,
        reuse_x ? w_right : 0, // tile_step: streams along its row when x is the held one
        reuse_x ? 0 : w_right, // run_step: next held tile, one column right
        w_down,                // step_step: next K row
    };
    const UnpackWalk walk {
        k.kt_dim,
        reuse_x ? k.rt_dim : k.ct_dim, // runs: one per held tile
        reuse_x ? k.ct_dim : k.rt_dim, // tiles_per_run: the streamed tiles
        reuse_x ? Where::SrcB : Where::SrcA,
    };
    const MathPlan math {
        OwnerExu::Fpu,
        c.math_fidelity,
        is_32bit_format(c.result.format()),
        c.in1.format,
        m.transpose_in1,
        0, // no function
    };
    return {Op::Matmul, x_plan, w_plan, walk, math, PackPlan {}, k};
}

inline constexpr OpApi<matmul_modifiers> matmul_op {&matmul_accepts, &matmul_options, &matmul_work, &matmul_plan};

} // namespace llk
