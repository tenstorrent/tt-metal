// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Pack's OpApi on Blackhole: a Dest tensor (in0, src) to an L1 tensor (in1, dst), every tile. The Resolver's second
// operand is the tensor pack writes, not a second input: its format is the caller's choice and decides the Plan.
//
//   accepts   src in Dest and dst in L1, the same tile grid, 32x32 tiles, dst contiguous, a conversion the packer
//             can do, L1 accumulation only in formats L1 can add. With sum_tiles: dst is one tile whatever src's
//             tile grid, in a format L1 can add
//   options   the packer's input format (pack_in_format, below) and dst as the result
//   work      the tile grid and the tile geometry
//   plan      PackPlan: formats, geometry and the modifiers (the tile count is the Work's). Unpack, walk and math stay empty
//
// sum_tiles (SDPA's softmax row sum, as the 1.0 kernel takes it: pack_tile<true> of every exp tile into the row's
// sum tile): every Dest tile goes to the same L1 tile, and from the second one on the packer adds it to what is
// there (L1 accumulation), so the L1 tile ends as the elementwise sum of the Dest tensor's tiles, in the L1 format.
// The Plan only says so (PackPlan::sum_tiles); the tile count stays the Dest tensor's.
//
// The packer converts in two stages (tt-isa-documentation, WormholeB0/.../Packers/FormatConversion.md): an early
// stage from what Dest holds to the packer's input format (In_data_format), then a late stage from that to the L1
// format. pack_in_format picks In_data_format the way the test harness does for Blackhole (infer_pack_in in
// tests/python_tests/helpers/data_format_inference.py), with one difference: the harness starts from the unpacker's
// output format, pack starts from the Dest tensor's format. They differ only for a block-float input with a 16-bit
// Dest: the harness passes the block format (Bfp8_b), pack passes what Dest holds (Float16_b). Both are conversions
// the packer supports.
//
// Not ported: untilize, tiles other than 32x32, a window of a wider tile grid as dst (several tile rows of some of a
// tensor's tile columns: PackPlan has no tile-row step).

#pragma once

#include <cstdint>

#include "resolver/formats.h" // has_8bit_exponent
#include "resolver/modifiers.h"
#include "resolver/resolver.h"

namespace llk2
{

/// The packer's input format (In_data_format) for Dest format `dest` and L1 format `out`: infer_pack_in for Blackhole,
/// with Dest's format in place of the unpacker's output (see the file comment).
constexpr DataFormat pack_in_format(DataFormat dest, DataFormat out, bool dest_fp32)
{
    // Sub-byte block formats cannot be In_data_format: the packer reads what Dest holds and converts late.
    if (out == DataFormat::Bfp4_b || out == DataFormat::Bfp2_b)
    {
        return dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
    }
    // A 32-bit Dest: the early stage converts straight to the L1 format.
    if (dest_fp32)
    {
        return out;
    }
    // FP16 to BFP8_b from a 16-bit Dest: through BFP8 (A), as the harness does.
    if (dest == DataFormat::Float16 && out == DataFormat::Bfp8_b)
    {
        return DataFormat::Bfp8;
    }
    // An 8-bit-exponent Dest to FP16: Blackhole's gasket converts, so the input is FP16 itself.
    if (has_8bit_exponent(dest) && out == DataFormat::Float16)
    {
        return DataFormat::Float16;
    }
    return dest;
}

// The packer's two conversion stages, ported from cpack_common.h. Like the LLK, both tables take In_data_format as
// their input; a pair either stage supports is accepted (is_packer_to_L1_conversion_supported).

/// Early stage pairs: is_packer_to_L1_early_conversion_supported.
constexpr bool pack_early_converts(DataFormat in, DataFormat out)
{
    switch (in)
    {
        case DataFormat::Float32:
            return out == DataFormat::Float32 || out == DataFormat::Tf32 || out == DataFormat::Float16_b || out == DataFormat::Bfp8_b ||
                   out == DataFormat::Int32 || out == DataFormat::Int8 || out == DataFormat::UInt8;
        case DataFormat::Float16_b:
            return out == DataFormat::Tf32 || out == DataFormat::Float16_b || out == DataFormat::Bfp8_b || out == DataFormat::Int8;
        case DataFormat::Float16:
            return out == DataFormat::Float16 || out == DataFormat::Bfp8 || out == DataFormat::Lf8 || out == DataFormat::Int8;
        case DataFormat::Int32:
            return out == DataFormat::Float32 || out == DataFormat::Tf32 || out == DataFormat::Float16_b || out == DataFormat::Int32 ||
                   out == DataFormat::Int8 || out == DataFormat::UInt8;
        case DataFormat::UInt16:
            return out == DataFormat::UInt16;
        default:
            return false;
    }
}

/// Late stage pairs: is_packer_to_L1_late_conversion_supported.
constexpr bool pack_late_converts(DataFormat in, DataFormat out)
{
    const bool fp32_column = out == DataFormat::Float32 || out == DataFormat::Float16_b || out == DataFormat::Float16 || out == DataFormat::Lf8 ||
                             out == DataFormat::Bfp8 || out == DataFormat::Bfp4 || out == DataFormat::Bfp2 || out == DataFormat::Bfp8_b ||
                             out == DataFormat::Bfp4_b || out == DataFormat::Bfp2_b;
    switch (in)
    {
        case DataFormat::Float32:
            return fp32_column;
        case DataFormat::Tf32:
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
            return fp32_column || out == DataFormat::Tf32;
        case DataFormat::Int32:
            return out == DataFormat::Int32;
        case DataFormat::UInt16:
            return out == DataFormat::UInt16;
        case DataFormat::Int8:
        case DataFormat::UInt8:
            return out == DataFormat::Int8 || out == DataFormat::UInt8;
        default:
            return false;
    }
}

/// Whether the packer can convert from In_data_format `in` to L1 format `out`: is_packer_to_L1_conversion_supported.
constexpr bool pack_converts(DataFormat in, DataFormat out)
{
    return pack_early_converts(in, out) || pack_late_converts(in, out);
}

/// The L1 formats the packer can accumulate into (tt-isa-documentation, WormholeB0/.../Packers/Accumulation.md).
constexpr bool pack_l1_accumulates(DataFormat out)
{
    return out == DataFormat::Float32 || out == DataFormat::Float16_b || out == DataFormat::Float16;
}

/// The sizes of the one L1 tile a sum_tiles pack of a Dest tensor with sizes d writes: d's tile, a 1 x 1 tile grid.
constexpr Dims pack_sum_sizes(Dims d)
{
    d[TileY] = 1;
    d[TileX] = 1;
    return d;
}

/// accepts: see the file comment.
constexpr Accepts pack_accepts(const Spec& src, const Spec& dst, const pack_modifiers& mods, const KernelSettings& settings)
{
    const bool storage = src.storage == Storage::Dest && dst.storage == Storage::L1;
    const bool tiles   = has_default_tiles(src.shape);
    // The same tile grid and tile geometry: Dest tile i (row-major) goes to L1 tile i. With sum_tiles dst is one tile
    // of that geometry, and every Dest tile goes to it.
    const bool shapes = mods.sum_tiles ? equal(pack_sum_sizes(src.shape.sizes), dst.shape.sizes) : equal(src.shape.sizes, dst.shape.sizes);
    // Dest tiles in Dest's layout and dst dense, both from offset 0.
    const bool layout   = equal(src.shape.strides, dest_strides(src.shape.sizes)) && is_contiguous(dst.shape) && src.shape.offset == 0 && dst.shape.offset == 0;
    const DataFormat in = pack_in_format(src.format, dst.format, settings.dest_fp32);
    const bool formats  = fits_dest_mode(src.format, settings.dest_fp32) && datum_bits(dst.format) != 0 && pack_converts(in, dst.format);
    // sum_tiles adds every tile but the first into L1 whatever l1_acc says, so it needs a format L1 can add as well.
    // untilize: not ported.
    const bool modifiers = ((mods.l1_acc == Toggle::Off && !mods.sum_tiles) || pack_l1_accumulates(dst.format)) && !mods.untilize;
    // A run-time size (resolver/work_counts.h): tile rows, or the tile columns of one tile row (Dest tile i goes to L1
    // tile i). src, as the op that wrote it returned it, and dst mark the same size; a src that marks none is a whole
    // Dest section of which dst's count is packed. With sum_tiles the count is src's (dst is one tile).
    const Spec& sized                    = has_runtime_size(src) ? src : dst;
    const bool one_row                   = !is_runtime_size(sized, TileX) || sized.shape.sizes[TileY] == 1;
    const bool runtime                   = one_row && (mods.sum_tiles ? !has_runtime_size(dst) : (!has_runtime_size(src) || same_runtime_size(src, dst)));
    return {.storage = storage, .tiles = tiles, .shapes = shapes, .layout = layout, .formats = formats, .modifiers = modifiers, .runtime = runtime};
}

/// options: the packer's input format (pack_in_format) and dst, owned by the packer, as the result.
constexpr OpConfig pack_options(const Spec& src, const Spec& dst, const pack_modifiers&, const KernelSettings& settings)
{
    // in0 is read from Dest and converted to the packer's input format; in1 is the L1 tensor: written, not read.
    return {
        .in0           = {.where = Where::Dest, .format = pack_in_format(src.format, dst.format, settings.dest_fp32)},
        .in1           = {.where = Where::None, .format = dst.format},
        .math_fidelity = math_fidelity_lofi, // no fidelity phases
        .result        = with_owner(dst, Thread::Pack, Exu::Packer),
    };
}

/// work: the tile grid and the tile geometry; no K. A size src marks run time is a run-time count, from src (in0);
/// otherwise one dst marks, from dst (in1).
constexpr Work pack_work(const Spec& src, const Spec& dst, const OpConfig&)
{
    if (has_runtime_size(src))
    {
        return in0_tile_grid_work(src);
    }
    Work work    = tile_grid_work(src);
    work.rows_from = in1_count_source(dst, TileY);
    work.cols_from = in1_count_source(dst, TileX);
    return work;
}

/// plan: PackPlan from the tensors and modifiers; unpack, walk and math stay empty.
constexpr Plan pack_plan(const Spec& src, const Spec& dst, const pack_modifiers& mods, const OpConfig& config, const Work& work)
{
    const PackPlan pack {
        .writes     = dst.format,
        .face_r_dim = work.face_r_dim,
        .num_faces  = work.num_faces,
        .tile_bytes = tile_bytes(dst),
        .relu       = mods.relu,
        .l1_acc     = mods.l1_acc,
        .untilize   = mods.untilize,
        .in_format  = config.in0.format,
        .dest_fp32  = is_32bit_format(src.format),
        .sum_tiles  = mods.sum_tiles,
    };
    return {.in0 = {}, .in1 = {}, .walk = {}, .math = {}, .pack = pack, .work = work};
}

inline constexpr OpApi<pack_modifiers> pack_op {&pack_accepts, &pack_options, &pack_work, &pack_plan};

} // namespace llk2
