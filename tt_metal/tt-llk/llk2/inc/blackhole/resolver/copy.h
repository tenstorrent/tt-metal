// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Copy's OpApi on Blackhole: an L1 tensor into a Dest tensor, tile by tile, optionally transposed (width <-> height).
// Copy has one input; the Resolver's second operand is the same tensor and is ignored.
//
// Two ways to reach Dest, decided here and nowhere else:
//   through SrcA   unpacker 0 writes the tile to SrcA, math moves it to Dest (MOVA2D). Every format the unpacker
//                  can write to Src, and the only way with transpose
//   straight       unpacker 0 writes the tile to Dest itself; math only waits for it. Only for 32-bit L1 formats
//                  with a 32-bit Dest, where the trip through 19-bit SrcA would lose bits (Float32 -> TF32).
//                  This is the test harness's unpack_to_dest rule: input is 32-bit and dest_acc is on
// With transpose, the unpacker transposes inside each face (Haloize_mode) and math writes the faces back in
// transposed order (face 1 <-> face 2). The unpacker cannot transpose on the way to Dest (ISA: UNPACR to Dest with
// Haloize_mode is undefined), so a transposed 32-bit copy goes through SrcA as TF32.
//
// Not ported: the lossless 32-bit transpose in Dest (llk_math_transpose_dest.h), and broadcasts.

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/matmul.h" // src_register_format, is_integer_format
#include "resolver/resolver.h"

namespace llk2
{

inline constexpr std::uint32_t copy_tile_dim = default_tile_dim; ///< copy is ported for 32x32 tiles only

/// The 32-bit formats the unpacker can write straight to Dest.
constexpr bool is_32bit_datum(DataFormat f)
{
    return datum_bits(f) == 32;
}

/// True when copy unpacks straight to Dest instead of through SrcA.
constexpr bool copy_unpacks_to_dest(DataFormat l1, const copy_modifiers& m, const KernelSettings& s)
{
    return is_32bit_datum(l1) && s.dest_fp32 && !m.transpose;
}

/// Formats with a 5-bit exponent (A formats): in a 16-bit Dest they stay FP16, the rest become BF16.
constexpr bool is_exponent_a(DataFormat f)
{
    return f == DataFormat::Float16 || f == DataFormat::Bfp8 || f == DataFormat::Bfp4 || f == DataFormat::Bfp2 || f == DataFormat::Lf8 ||
           f == DataFormat::Fp8_e4m3;
}

/// The Dest format a copy of l1 produces.
constexpr DataFormat copy_dest_format(DataFormat l1, const copy_modifiers& m, const KernelSettings& s)
{
    if (copy_unpacks_to_dest(l1, m, s))
    {
        return l1; // the unpacker writes FP32 / INT32 / UINT32 unchanged
    }
    if (s.dest_fp32)
    {
        return DataFormat::Float32; // MOVA2D widens BF16 / TF32 to FP32 when Dest is 32-bit
    }
    return is_exponent_a(l1) ? DataFormat::Float16 : DataFormat::Float16_b;
}

constexpr bool copy_accepts(const TensorSpec<6>& src, const TensorSpec<6>&, const copy_modifiers& m, const KernelSettings& s)
{
    const bool shape = src.storage == Storage::L1 && tile_rows(src.shape) == copy_tile_dim && tile_cols(src.shape) == copy_tile_dim;
    // Integer data is not ported yet: through SrcA it needs the integer datapath, to Dest only INT32 / UINT32 work.
    const bool format =
        !is_integer_format(src.format) && (copy_unpacks_to_dest(src.format, m, s) || src_register_format(src.format, s.dest_fp32) != DataFormat::Invalid);
    return shape && format;
}

constexpr OpConfig copy_options(const TensorSpec<6>& src, const TensorSpec<6>&, const copy_modifiers& m, const KernelSettings& s)
{
    const bool to_dest = copy_unpacks_to_dest(src.format, m, s);
    return {
        to_dest ? Exu::Unpacker : Exu::Fpu, // straight to Dest, the unpacker does all the work
        {to_dest ? Where::Dest : Where::SrcA, to_dest ? src.format : src_register_format(src.format, s.dest_fp32)},
        {Where::None, DataFormat::Invalid},
        s.math_fidelity,
        laid_out(src.shape.sizes, copy_dest_format(src.format, m, s), Storage::Dest),
    };
}

constexpr Work copy_work(const TensorSpec<6>& src, const TensorSpec<6>&, const OpConfig&)
{
    return {src.shape.sizes[TileY], src.shape.sizes[TileX], 1, src.shape.sizes[DatumY], faces_per_tile(src.shape)};
}

// The unpack walk of one copy: one step, one run, every tile of src streamed in L1 order, each one tile further on.
// Tile i of src lands in tile i of acc.
constexpr Plan copy_plan(const TensorSpec<6>& src, const TensorSpec<6>&, const copy_modifiers& m, const OpConfig& c, const Work& k)
{
    constexpr std::uint32_t next_tile = 1; // tiles from src tile i to tile i + 1
    const UnpackPlan src_plan {
        c.in0.where,
        src.format,
        c.in0.format,
        k.face_r_dim,
        k.num_faces,
        tile_bytes(src),
        Layout::Tilized,
        Broadcast::None,
        m.transpose,
        next_tile, // tile_step
        0,         // run_step: one run
        0,         // step_step: one step
    };
    const UnpackPlan unused {}; // to = Where::None: unpacker 1 idles
    const UnpackWalk walk {1, 1, k.rt_dim * k.ct_dim, Where::None};
    const MathPlan math {c.unit, c.math_fidelity, is_32bit_datum(c.result.format), c.in0.format, m.transpose, 0};
    return {Op::Copy, src_plan, unused, walk, math, PackPlan {}, k};
}

inline constexpr OpApi<copy_modifiers> copy_op {&copy_accepts, &copy_options, &copy_work, &copy_plan};

} // namespace llk2
