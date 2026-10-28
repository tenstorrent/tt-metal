// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The eltwise binary OpApis on Blackhole: add and sub. in0 unpacks to SrcA, in1 to SrcB, as tt-llk does; the FPU
// writes in0 <function> in1 into Dest, one 8-row block per instruction (ELWADD, ELWSUB).
//
// The ops differ only in BinaryFunction, which the Plan carries to llk-lib's math; accepts, options and work are
// shared. mul is one more BinaryFunction away; it is not wired yet because ELWMUL also needs fidelity phases and
// always accumulates onto Dest.
//
// Unpack walk: every operand streams, nothing is held. A run is one row of output tiles: in0 moves one tile per
// streamed tile and one tile row per run. in1 moves the same way when it has in0's shape; with Broadcast::Row it is a
// single row of tiles (a bias), so it goes back to its first tile for every run and the unpacker reads only each
// tile's top face row (the row that is broadcast).

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/formats.h"
#include "resolver/resolver.h"

namespace llk2
{

inline constexpr std::uint32_t eltwise_binary_tile_dim = default_tile_dim; ///< eltwise binary is ported for 32x32 tiles only
inline constexpr std::uint32_t eltwise_binary_lofi     = 0;                ///< ckernel::MathFidelity::LoFi: one fidelity phase

/// The Dest format an eltwise binary writes: FP32 with a 32-bit Dest, else FP16 or BF16 by the Src style (ELWADD.md).
constexpr DataFormat eltwise_binary_dest_format(DataFormat in, bool dest_fp32)
{
    if (dest_fp32)
    {
        return DataFormat::Float32;
    }
    return has_5bit_exponent(in) ? DataFormat::Float16 : DataFormat::Float16_b;
}

/// in1 is one row of tiles spread over every row of in0's tiles.
constexpr bool is_eltwise_binary_bias_row(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1)
{
    return in1.shape.sizes[TileY] == 1 && in1.shape.sizes[TileX] == in0.shape.sizes[TileX];
}

/// t is in L1 in 32x32 tiles.
constexpr bool is_eltwise_binary_tile(const TensorSpec<tensor_rank>& t)
{
    return t.storage == Storage::L1 && tile_rows(t.shape) == eltwise_binary_tile_dim && tile_cols(t.shape) == eltwise_binary_tile_dim;
}

/// accepts: 32x32 float tiles in L1, in1 shaped as Mods.bcast needs, both operands of one exponent width.
template <class Mods>
constexpr bool eltwise_binary_accepts(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& m, const KernelSettings& s)
{
    const bool tiles = is_eltwise_binary_tile(in0) && is_eltwise_binary_tile(in1);
    // No broadcast: the same tiles. Row: in1 is one row of tiles, its top row broadcast down each tile.
    const bool shapes =
        m.bcast == Broadcast::None ? equal(in0.shape.sizes, in1.shape.sizes) : m.bcast == Broadcast::Row && is_eltwise_binary_bias_row(in0, in1);
    // Both operands are read in SrcA's style (ELWADD.md): same exponent width, floats only for now.
    const bool formats = src_register_format(in0.format, s.dest_fp32) != DataFormat::Invalid &&
                         src_register_format(in1.format, s.dest_fp32) != DataFormat::Invalid && !is_integer_format(in0.format) &&
                         !is_integer_format(in1.format) && has_5bit_exponent(in0.format) == has_5bit_exponent(in1.format);
    return tiles && shapes && formats;
}

/// options: in0 to SrcA, in1 to SrcB, one fidelity phase, the result in Dest with in0's shape.
template <class Mods>
constexpr OpConfig eltwise_binary_options(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods&, const KernelSettings& s)
{
    return {
        Exu::Fpu,
        {Where::SrcA, src_register_format(in0.format, s.dest_fp32)},
        {Where::SrcB, src_register_format(in1.format, s.dest_fp32)},
        eltwise_binary_lofi, // ELWADD and ELWSUB run one fidelity phase, whatever the kernel's fidelity is
        laid_out(in0.shape.sizes, eltwise_binary_dest_format(in0.format, s.dest_fp32), Storage::Dest),
    };
}

/// work: one output tile per in0 tile, rt_dim x ct_dim of them.
constexpr Work eltwise_binary_work(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>&, const OpConfig& c)
{
    const Dims& out = c.result.shape.sizes;
    return {out[TileY], out[TileX], no_k_steps, in0.shape.sizes[DatumY], faces_per_tile(in0.shape)}; // one input tile pair per output tile
}

/// plan: the unpack walk described at the top of this file, and math running F.
template <class Mods, BinaryFunction F>
constexpr Plan eltwise_binary_plan(const TensorSpec<tensor_rank>& in0, const TensorSpec<tensor_rank>& in1, const Mods& m, const OpConfig& c, const Work& k)
{
    const std::uint32_t next_tile     = 1;        // tiles from (r, c) to (r, c + 1)
    const std::uint32_t next_tile_row = k.ct_dim; // tiles from (r, c) to (r + 1, c)
    const std::uint32_t no_step       = 0;        // the walk has one step

    const UnpackPlan in0_plan {
        Where::SrcA,
        in0.format,
        c.in0.format,
        k.face_r_dim,
        k.num_faces,
        tile_bytes(in0),
        Layout::Tilized,
        Broadcast::None,
        false, // no transpose
        next_tile,
        next_tile_row,
        no_step,
    };
    const UnpackPlan in1_plan {
        Where::SrcB,
        in1.format,
        c.in1.format,
        in1.shape.sizes[DatumY],
        faces_per_tile(in1.shape),
        tile_bytes(in1),
        Layout::Tilized,
        m.bcast,
        false, // no transpose
        next_tile,
        m.bcast == Broadcast::Row ? 0 : next_tile_row, // run_step: a broadcast row of tiles starts over for every output row
        no_step,
    };
    const UnpackWalk walk {1 /* steps */, k.rt_dim, k.ct_dim, Where::None};
    const MathPlan math {
        Exu::Fpu,
        c.math_fidelity,
        is_32bit_format(c.result.format),
        c.in0.format,
        false, // no transpose
        static_cast<std::uint32_t>(F),
    };
    return {Op::EltwiseBinary, in0_plan, in1_plan, walk, math, PackPlan {}, k};
}

/// in0 + in1.
inline constexpr OpApi<add_modifiers> add_op {
    &eltwise_binary_accepts<add_modifiers>,
    &eltwise_binary_options<add_modifiers>,
    &eltwise_binary_work,
    &eltwise_binary_plan<add_modifiers, BinaryFunction::Add>,
};

/// in0 - in1.
inline constexpr OpApi<sub_modifiers> sub_op {
    &eltwise_binary_accepts<sub_modifiers>,
    &eltwise_binary_options<sub_modifiers>,
    &eltwise_binary_work,
    &eltwise_binary_plan<sub_modifiers, BinaryFunction::Sub>,
};

} // namespace llk2
