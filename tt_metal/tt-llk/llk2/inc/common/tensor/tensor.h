// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensor: the operand type every LLK 2.0 layer takes. Arch-independent.
// Ported from tt-llk-lab lpremovic/llk-tensor, report 2 (02_tensor_shape_and_resolution.cpp) and its appendix.
//
//   TensorShape  sizes and strides over 6 dims, outermost first, strides in datums, plus an offset in datums:
//                tile y/x, face y/x, datum y/x
//   TensorSpec   the shape, the data format, the storage (L1 or Dest) and the owner (thread, unit). Compile time
//   Tensor       a spec by reference and a release action, both in its type, and a run-time address
//                (L1: the byte address of datum 0; Dest: the first row)

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensix_types.h"  // DataFormat

namespace llk2
{

enum Dim : std::size_t
{
    TileY,
    TileX,
    FaceY,
    FaceX,
    DatumY,
    DatumX
};

enum class Storage : std::uint8_t
{
    L1,
    Dest
};

enum class Thread : std::uint8_t
{
    Unpack,
    Math,
    Pack,
    None
};

enum class Exu : std::uint8_t
{
    Unpacker,
    Fpu,
    Sfpu,
    Packer,
    None
};

/// Dims of every TensorShape: one past the innermost, DatumX.
inline constexpr std::size_t tensor_rank = DatumX + 1;

/// Sizes or strides of a TensorShape, one entry per Dim.
using Dims = std::array<std::uint32_t, tensor_rank>;

// ---- tile geometry constants (the same on every arch) ---------------------------------------------------------------
inline constexpr std::uint32_t face_cols          = 16; ///< datums in a face row; every face is this wide
inline constexpr std::uint32_t max_face_rows      = 16; ///< rows in a full face; shorter tiles have shorter faces
inline constexpr std::uint32_t default_tile_dim   = 32; ///< tile height and width when the caller names none
inline constexpr std::uint32_t dest_row_datums    = 16; ///< datums in one Dest row
inline constexpr std::uint32_t min_dest_face_rows = 8;  ///< every face takes at least an 8-row slot in Dest (Quasar's tiny_face_stride)
inline constexpr std::uint32_t bfp_block_datums   = 16; ///< block float formats share one exponent byte per 16 datums
inline constexpr std::uint32_t l1_word_bytes      = 16; ///< L1 word: tiles in L1 are padded to whole words, and the unpacker, packer and
                                                         ///< dataflow buffer addresses and tile sizes count words
inline constexpr std::uint32_t bits_per_byte      = 8;  ///< bits in one byte, for datum_bits to bytes
inline constexpr std::uint32_t faces_per_face_row = default_tile_dim / face_cols; ///< faces side by side in a full tile: 2
inline constexpr std::uint32_t faces_per_full_tile =
    faces_per_face_row * (default_tile_dim / max_face_rows); ///< faces in a full 32x32 tile: 2 x 2, stored 0, 1 (top), 2, 3 (bottom)

template <std::size_t Rank>
struct TensorShape
{
    std::array<std::uint32_t, Rank> sizes, strides;
    std::uint32_t offset;
};

template <std::size_t Rank>
struct TensorSpec
{
    TensorShape<Rank> shape;
    DataFormat format;
    Storage storage;
    Thread thread; // the owner: its thread
    Exu exu;       // and its unit
};

struct NoRelease
{
    void operator()() const
    {
    }
};

template <const auto& S, class Release = NoRelease>
struct Tensor : Release
{
    static constexpr const auto& spec = S;
    std::uint32_t address;

    constexpr explicit Tensor(std::uint32_t a) : address(a)
    {
    }
};

template <const auto& S, class R>
inline void release(const Tensor<S, R>& t)
{
    static_cast<const R&>(t)();
}

// ---- helpers over rank-6 shapes and specs ----------------------------------------------------------------------
constexpr std::uint32_t min_u32(std::uint32_t a, std::uint32_t b)
{
    return a < b ? a : b;
}

constexpr std::uint32_t max_u32(std::uint32_t a, std::uint32_t b)
{
    return a > b ? a : b;
}

constexpr bool equal(const Dims& a, const Dims& b)
{
    for (std::size_t d = 0; d < tensor_rank; ++d)
    {
        if (a[d] != b[d])
        {
            return false;
        }
    }
    return true;
}

constexpr Dims dense_strides(const Dims& s) // L1: row-major in dim order
{
    Dims t {};
    t[DatumX] = 1; // datums are contiguous along x
    for (std::size_t d = DatumX; d-- > 0;)
    {
        t[d] = s[d + 1] * t[d + 1];
    }
    return t;
}

constexpr Dims dest_strides(const Dims& s) // Dest: rows of 16 datums, a face in at least an 8-row slot (every arch)
{
    Dims t {};
    t[DatumX] = 1; // datums are contiguous along a row
    t[DatumY] = dest_row_datums;
    t[FaceX]  = dest_row_datums * max_u32(s[DatumY], min_dest_face_rows);
    t[FaceY]  = t[FaceX] * s[FaceX];
    t[TileX]  = t[FaceY] * s[FaceY];
    t[TileY]  = t[TileX] * s[TileX];
    return t;
}

constexpr bool is_bfp(DataFormat f)
{
    return f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2_b || f == DataFormat::Bfp8 || f == DataFormat::Bfp4 ||
           f == DataFormat::Bfp2;
}

constexpr std::uint32_t datum_bits(DataFormat f)
{
    switch (f)
    {
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32:
            return 32;
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return 16;
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b:
        case DataFormat::Int8:
        case DataFormat::UInt8:
            return 8;
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
            return 4;
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
            return 2;
        default:
            return 0; // not supported yet: no Tensor can be made in this format
    }
}

/// 32-bit datums: FP32, INT32, UINT32 (the formats a 32-bit Dest holds and the unpacker can write straight to Dest).
constexpr bool is_32bit_format(DataFormat f)
{
    return datum_bits(f) == 32;
}

constexpr std::uint32_t tile_rows(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceY] * s.sizes[DatumY];
}

constexpr std::uint32_t tile_cols(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceX] * s.sizes[DatumX];
}

constexpr std::uint32_t tile_count(const TensorShape<tensor_rank>& s)
{
    return s.sizes[TileY] * s.sizes[TileX];
}

constexpr std::uint32_t faces_per_tile(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceY] * s.sizes[FaceX];
}

constexpr std::uint32_t round_up(std::uint32_t value, std::uint32_t multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

// Bytes of one tile in L1: mantissas, plus one exponent byte per block for block formats; each part padded to
// whole 16-byte words.
constexpr std::uint32_t tile_bytes(DataFormat f, std::uint32_t datums)
{
    const std::uint32_t exponents = is_bfp(f) ? round_up(datums / bfp_block_datums, l1_word_bytes) : 0;
    return round_up(datums * datum_bits(f) / bits_per_byte + exponents, l1_word_bytes);
}

constexpr std::uint32_t tile_bytes(const TensorSpec<tensor_rank>& t)
{
    return tile_bytes(t.format, tile_rows(t.shape) * tile_cols(t.shape));
}

constexpr bool is_contiguous(const TensorShape<tensor_rank>& s)
{
    return equal(s.strides, dense_strides(s.sizes));
}

constexpr bool same_except_owner(const TensorSpec<tensor_rank>& a, const TensorSpec<tensor_rank>& b)
{
    return equal(a.shape.sizes, b.shape.sizes) && equal(a.shape.strides, b.shape.strides) && a.shape.offset == b.shape.offset && a.format == b.format &&
           a.storage == b.storage;
}

constexpr bool same(const TensorSpec<tensor_rank>& a, const TensorSpec<tensor_rank>& b) // specs are equal by value, not by type
{
    return same_except_owner(a, b) && a.thread == b.thread && a.exu == b.exu;
}

constexpr TensorSpec<tensor_rank> with_owner(TensorSpec<tensor_rank> t, Thread thread, Exu exu)
{
    t.thread = thread;
    t.exu    = exu;
    return t;
}

template <const auto& S, Thread T, Exu E>
inline constexpr auto owned_by = with_owner(S, T, E); // the same tensor under another owner

constexpr bool free_or(const TensorSpec<tensor_rank>& d, Thread t, Exu e) // nobody's, or the writer's already
{
    return d.thread == Thread::None || (d.thread == t && d.exu == e);
}

// ---- constructing: the tile is named, or min(dim, 32) per axis -------------------------------------------------
constexpr bool is_valid_tile(std::uint32_t h, std::uint32_t w)
{
    // Tiny tiles are 1, 2, 4 or 8 rows tall; full tiles a whole number of faces tall.
    const bool tiny_height = h == 1 || h == 2 || h == 4 || h == 8;
    return w > 0 && w % face_cols == 0 && (tiny_height || (h > 0 && h % max_face_rows == 0));
}

constexpr Dims tiled(std::uint32_t tiles_y, std::uint32_t tiles_x, std::uint32_t h, std::uint32_t w)
{
    const std::uint32_t face_rows = min_u32(h, max_face_rows);
    return {tiles_y, tiles_x, h / face_rows, w / face_cols, face_rows, face_cols};
}

constexpr TensorSpec<tensor_rank> laid_out(const Dims& sizes, DataFormat f, Storage storage)
{
    const Dims strides = storage == Storage::Dest ? dest_strides(sizes) : dense_strides(sizes);
    return {{sizes, strides, 0}, f, storage, Thread::None, Exu::None}; // nobody's: nothing has produced it yet
}

template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H, std::uint32_t W>
constexpr TensorSpec<tensor_rank> make_spec(Storage storage)
{
    static_assert(is_valid_tile(H, W), "a tile is 16k wide and 1, 2, 4, 8 or 16k tall: pad the tensor first");
    static_assert(Rows % H == 0 && Cols % W == 0, "the tile must divide the tensor: pad it first");
    return laid_out(tiled(Rows / H, Cols / W, H, W), F, storage);
}

template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H = min_u32(Rows, default_tile_dim), std::uint32_t W = min_u32(Cols, default_tile_dim)>
inline constexpr TensorSpec<tensor_rank> l1_spec = make_spec<Rows, Cols, F, H, W>(Storage::L1);

template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H = min_u32(Rows, default_tile_dim), std::uint32_t W = min_u32(Cols, default_tile_dim)>
inline constexpr TensorSpec<tensor_rank> dst_spec = make_spec<Rows, Cols, F, H, W>(Storage::Dest);

template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H = min_u32(Rows, default_tile_dim), std::uint32_t W = min_u32(Cols, default_tile_dim)>
constexpr auto l1_tensor(std::uint32_t address)
{
    return Tensor<l1_spec<Rows, Cols, F, H, W>>(address);
}

template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H = min_u32(Rows, default_tile_dim), std::uint32_t W = min_u32(Cols, default_tile_dim)>
constexpr auto dst_tensor(std::uint32_t row)
{
    return Tensor<dst_spec<Rows, Cols, F, H, W>>(row);
}

// [M, K] x [K, N]: rows from a, columns from b.
constexpr Dims product_sizes(const Dims& a, const Dims& b)
{
    return {a[TileY], b[TileX], a[FaceY], b[FaceX], a[DatumY], b[DatumX]};
}

} // namespace llk2
