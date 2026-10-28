// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Pack's OpApi on Blackhole: a Dest tensor (in0) to an L1 tensor (in1), every tile.
//
//   accepts   Dest -> L1, the same tile grid, 32x32 tiles, a conversion the packer can do, L1 accumulation only in
//             formats L1 can add, no untilize yet
//   options   the packer's input format (pack_in_format, below) and the L1 tensor as the result
//   work      the tile grid and the tile geometry
//   plan      PackPlan: formats, geometry, tile count and the modifiers. Unpack, walk and math stay empty
//
// The packer converts in two stages (tt-isa-documentation, WormholeB0/.../Packers/FormatConversion.md): an early
// stage from what Dest holds to the packer's input format (In_data_format), then a late stage from that to the L1
// format. pack_in_format picks In_data_format the way the test harness does for Blackhole (infer_pack_in in
// tests/python_tests/helpers/data_format_inference.py), with one difference: the harness starts from the unpacker's
// output format, pack starts from the Dest tensor's format. They differ only for a block-float input with a 16-bit
// Dest: the harness passes the block format (Bfp8_b), pack passes what Dest holds (Float16_b). Both are conversions
// the packer supports.

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/formats.h" // is_b_format
#include "resolver/resolver.h"

namespace llk2
{

inline constexpr std::uint32_t pack_tile_dim = default_tile_dim; ///< pack is ported for 32x32 tiles only

/// The Dest formats of each Dest mode: 32-bit datums (FP32, INT32) or 16-bit ones.
constexpr bool fits_dest_mode(DataFormat dest, bool dest_fp32)
{
    if (dest_fp32)
    {
        return dest == DataFormat::Float32 || dest == DataFormat::Int32;
    }
    return dest == DataFormat::Float16 || dest == DataFormat::Float16_b || dest == DataFormat::UInt16;
}

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
    if (is_b_format(dest) && out == DataFormat::Float16)
    {
        return DataFormat::Float16;
    }
    return dest;
}

// The packer's two conversion stages, ported from cpack_common.h. Like the LLK, both tables take In_data_format as
// their input; a pair either stage supports is accepted (is_packer_to_L1_conversion_supported).

/// Early stage pairs: is_packer_to_L1_early_conversion_supported.
constexpr bool packer_early_converts(DataFormat in, DataFormat out)
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
constexpr bool packer_late_converts(DataFormat in, DataFormat out)
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
constexpr bool packer_converts(DataFormat in, DataFormat out)
{
    return packer_early_converts(in, out) || packer_late_converts(in, out);
}

/// The L1 formats the packer can accumulate into (tt-isa-documentation, WormholeB0/.../Packers/Accumulation.md).
constexpr bool l1_accumulates(DataFormat out)
{
    return out == DataFormat::Float32 || out == DataFormat::Float16_b || out == DataFormat::Float16;
}

/// accepts: see the file comment.
constexpr bool pack_accepts(const TensorSpec<tensor_rank>& d, const TensorSpec<tensor_rank>& o, const pack_modifiers& m, const KernelSettings& s)
{
    const bool storage = d.storage == Storage::Dest && o.storage == Storage::L1;
    // The same tile grid and tile geometry; Dest tiles in Dest's layout and the L1 tensor dense, both from offset 0,
    // so Dest tile i (row-major) goes to L1 tile i.
    const bool shapes = equal(d.shape.sizes, o.shape.sizes) && equal(d.shape.strides, dest_strides(d.shape.sizes)) && is_contiguous(o.shape) &&
                        d.shape.offset == 0 && o.shape.offset == 0 && tile_rows(d.shape) == pack_tile_dim && tile_cols(d.shape) == pack_tile_dim;
    const DataFormat in  = pack_in_format(d.format, o.format, s.dest_fp32);
    const bool formats   = fits_dest_mode(d.format, s.dest_fp32) && datum_bits(o.format) != 0 && packer_converts(in, o.format);
    const bool modifiers = (m.l1_acc == Toggle::Off || l1_accumulates(o.format)) && !m.untilize;
    return storage && shapes && formats && modifiers;
}

/// options: the packer's input format (pack_in_format) and the L1 tensor, owned by the packer, as the result.
constexpr OpConfig pack_options(const TensorSpec<tensor_rank>& d, const TensorSpec<tensor_rank>& o, const pack_modifiers&, const KernelSettings& s)
{
    return {
        Exu::Packer,
        {Where::Dest, pack_in_format(d.format, o.format, s.dest_fp32)}, // in0: read from Dest, converted to this
        {Where::None, o.format},                                        // in1: the L1 tensor, written, not read
        0,                                                              // no math fidelity
        with_owner(o, Thread::Pack, Exu::Packer),
    };
}

/// work: the tile grid and the tile geometry; no K.
constexpr Work pack_work(const TensorSpec<tensor_rank>& d, const TensorSpec<tensor_rank>&, const OpConfig&)
{
    return {d.shape.sizes[TileY], d.shape.sizes[TileX], no_k_steps, d.shape.sizes[DatumY], faces_per_tile(d.shape)};
}

/// plan: PackPlan from the tensors and modifiers; unpack, walk and math stay empty.
constexpr Plan pack_plan(const TensorSpec<tensor_rank>& d, const TensorSpec<tensor_rank>& o, const pack_modifiers& m, const OpConfig& c, const Work& k)
{
    const PackPlan pack {
        d.format,
        o.format,
        k.face_r_dim,
        k.num_faces,
        tile_bytes(o),
        tile_count(d.shape),
        m.relu,
        m.l1_acc,
        m.untilize,
        c.in0.format,
        is_32bit_format(d.format),
    };
    return {Op::Pack, UnpackPlan {}, UnpackPlan {}, UnpackWalk {}, MathPlan {}, pack, k};
}

inline constexpr OpApi<pack_modifiers> pack_op {&pack_accepts, &pack_options, &pack_work, &pack_plan};

} // namespace llk2
