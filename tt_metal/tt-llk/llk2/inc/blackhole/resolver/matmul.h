// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's OpApi on Blackhole. in0 [M, K] unpacks to SrcB, in1 [K, N] to SrcA, as tt-llk does; the FPU accumulates
// rt_dim x ct_dim output tiles in Dest over kt_dim K steps.

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/formats.h"
#include "resolver/resolver.h"

namespace llk2
{

// The Dest format matmul accumulates into: Int32 for integer inputs (needs the 32-bit Dest mode), else FP32 or BF16.
constexpr DataFormat matmul_dest_format(DataFormat in, bool dest_fp32)
{
    if (is_integer_format(in))
    {
        return DataFormat::Int32;
    }
    return dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

inline constexpr std::uint32_t matmul_tile_dim = default_tile_dim; ///< matmul is ported for 32x32 tiles only

constexpr bool matmul_accepts(const TensorSpec<tensor_rank>& x, const TensorSpec<tensor_rank>& w, const matmul_modifiers& m, const KernelSettings& s)
{
    const bool shapes = x.storage == Storage::L1 && w.storage == Storage::L1 && x.shape.sizes[TileX] == w.shape.sizes[TileY] &&
                        tile_cols(x.shape) == tile_rows(w.shape) && tile_rows(x.shape) == matmul_tile_dim && tile_cols(x.shape) == matmul_tile_dim && tile_cols(w.shape) == matmul_tile_dim;
    const bool formats = src_register_format(x.format, s.dest_fp32) != DataFormat::Invalid && src_register_format(w.format, s.dest_fp32) != DataFormat::Invalid &&
                         is_integer_format(x.format) == is_integer_format(w.format) && (!is_integer_format(x.format) || s.dest_fp32);
    return shapes && formats && !m.transpose_in1; // 32x32 tiles in L1, no transpose, for now
}

constexpr OpConfig matmul_options(const TensorSpec<tensor_rank>& x, const TensorSpec<tensor_rank>& w, const matmul_modifiers&, const KernelSettings& s)
{
    return {
        Exu::Fpu,
        {Where::SrcB, src_register_format(x.format, s.dest_fp32)},
        {Where::SrcA, src_register_format(w.format, s.dest_fp32)},
        s.math_fidelity,
        laid_out(product_sizes(x.shape.sizes, w.shape.sizes), matmul_dest_format(x.format, s.dest_fp32), Storage::Dest),
    };
}

constexpr Work matmul_work(const TensorSpec<tensor_rank>& x, const TensorSpec<tensor_rank>&, const OpConfig& c)
{
    const Dims& out = c.result.shape.sizes;
    return {out[TileY], out[TileX], x.shape.sizes[TileX], x.shape.sizes[DatumY], faces_per_tile(x.shape)};
}

// The unpack walk of one matmul call. A K step multiplies x's tile column k (rt_dim tiles) by w's tile row k
// (ct_dim tiles). The longer output dimension is reused: when ct_dim >= rt_dim, each x tile is held in SrcB for one
// run while the ct_dim w tiles of row k stream through SrcA; otherwise each w tile is held in SrcA while the rt_dim
// x tiles of column k stream through SrcB. x is [rt_dim, kt_dim] tiles and w [kt_dim, ct_dim], both row-major, so
// moving down x's column is kt_dim tiles and moving along w's row is 1 tile.
constexpr bool matmul_reuses_x(const Work& k)
{
    return k.ct_dim >= k.rt_dim;
}

constexpr Plan matmul_plan(const TensorSpec<tensor_rank>& x, const TensorSpec<tensor_rank>& w, const matmul_modifiers& m, const OpConfig& c, const Work& k)
{
    const bool reuse_x       = matmul_reuses_x(k);
    const std::uint32_t x_down  = k.kt_dim; // tiles from x(r, k) to x(r + 1, k)
    const std::uint32_t x_right = 1;        // tiles from x(r, k) to x(r, k + 1)
    const std::uint32_t w_right = 1;        // tiles from w(k, c) to w(k, c + 1)
    const std::uint32_t w_down  = k.ct_dim; // tiles from w(k, c) to w(k + 1, c)

    const UnpackPlan x_plan {
        Where::SrcB, x.format, c.in0.format, k.face_r_dim, k.num_faces, tile_bytes(x), Layout::Tilized, Broadcast::None, false,
        reuse_x ? 0 : x_down, // tile_step: streams down its column when w is the held one
        reuse_x ? x_down : 0, // run_step: next held tile, one row down
        x_right,              // step_step: next K column
    };
    const UnpackPlan w_plan {
        Where::SrcA, w.format, c.in1.format, w.shape.sizes[DatumY], faces_per_tile(w.shape), tile_bytes(w), Layout::Tilized, Broadcast::None, m.transpose_in1,
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
        Exu::Fpu, c.math_fidelity, c.result.format == DataFormat::Float32 || c.result.format == DataFormat::Int32, c.in1.format, m.transpose_in1, 0,
    };
    return {Op::Matmul, x_plan, w_plan, walk, math, PackPlan {}, k};
}

inline constexpr OpApi<matmul_modifiers> matmul_op {&matmul_accepts, &matmul_options, &matmul_work, &matmul_plan};

} // namespace llk2
