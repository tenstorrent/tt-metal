// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's OpApi on Blackhole. in0 [M, K] unpacks to SrcB, in1 [K, N] to SrcA, as tt-llk does; the FPU accumulates
// rt_dim x ct_dim output tiles in Dest over kt_dim K steps.

#pragma once

#include "compute_api/modifiers.h"
#include "resolver/resolver.h"

namespace llk2::blackhole
{

// The register format the unpacker writes for an L1 format, unpacking to Src (data_format_inference.py).
constexpr DataFormat src_register_format(DataFormat l1, bool dest_fp32)
{
    if (l1 == DataFormat::Float32)
    {
        return dest_fp32 ? DataFormat::Tf32 : DataFormat::Float16_b; // Src holds 19 bits
    }
    return l1; // Float16, Float16_b, Bfp8_b, Bfp4_b, Bfp2_b: the unpacker keeps the format id
}

constexpr bool matmul_accepts(const TensorSpec<6>& x, const TensorSpec<6>& w)
{
    return x.storage == Storage::L1 && w.storage == Storage::L1 && x.shape.sizes[TileX] == w.shape.sizes[TileY] &&
           tile_cols(x.shape) == tile_rows(w.shape) && tile_rows(x.shape) == 32 && tile_cols(w.shape) == 32 && datum_bits(x.format) != 0 &&
           datum_bits(w.format) != 0;
}

constexpr OpConfig matmul_options(const TensorSpec<6>& x, const TensorSpec<6>& w, const matmul_modifiers&, const KernelSettings& s)
{
    check(matmul_accepts(x, w)); // 32x32 tiles in L1 with matching K, for now
    const DataFormat dest_format = s.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
    return {
        Exu::Fpu,
        {Where::SrcB, src_register_format(x.format, s.dest_fp32)},
        {Where::SrcA, src_register_format(w.format, s.dest_fp32)},
        s.math_fidelity,
        laid_out(product_sizes(x.shape.sizes, w.shape.sizes), dest_format, Storage::Dest),
    };
}

constexpr Work matmul_work(const TensorSpec<6>& x, const TensorSpec<6>&, const OpConfig& c)
{
    const Dims& out = c.result.shape.sizes;
    return {out[TileY], out[TileX], x.shape.sizes[TileX], x.shape.sizes[DatumY], faces_per_tile(x.shape)};
}

constexpr Plan matmul_plan(const TensorSpec<6>& x, const TensorSpec<6>& w, const matmul_modifiers& m, const OpConfig& c, const Work& k)
{
    return {
        {Where::SrcB, x.format, c.in0.format, k.face_r_dim, k.num_faces, tile_bytes(x), false},
        {Where::SrcA, w.format, c.in1.format, w.shape.sizes[DatumY], faces_per_tile(w.shape), tile_bytes(w), m.transpose_in1},
        {Exu::Fpu, MathOp::Matmul, c.math_fidelity, c.result.format == DataFormat::Float32, c.in1.format, m.transpose_in1},
        k,
    };
}

inline constexpr OpApi<matmul_modifiers> matmul_op {&matmul_options, &matmul_work, &matmul_plan};

} // namespace llk2::blackhole

namespace llk2
{
using blackhole::matmul_op; // the arch's OpApi, found by the arch-independent Compute API
} // namespace llk2
