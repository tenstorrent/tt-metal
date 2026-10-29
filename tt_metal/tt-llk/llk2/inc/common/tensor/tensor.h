// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensor: the operand type every LLK 2.0 layer takes. Arch-independent.
// Ported from tt-llk-lab lpremovic/llk-tensor, report 2 (02_tensor_shape_and_resolution.cpp) and its appendix:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor
//
//   TensorShape  sizes and strides over tensor_rank (6) dims, outermost first, strides in datums, plus an offset in
//                datums: tile y/x, face y/x, datum y/x
//   TensorSpec   the shape, the data format, the storage (L1 or Dest) and the owner (thread, unit). Compile time
//   Tensor       a spec by reference and a release action, both in its type, and a run-time address (L1: the byte
//                address of datum 0; Dest: the first row)
//
// The owner is the (thread, unit) that last wrote the tensor, or nobody's (Thread::None) for a tensor nothing in the
// kernel has produced. Every op returns what it wrote as a Tensor of its own unit, so the owner travels with the
// type, and the next op that touches it hands it over (compute_api/sync.h). owned_by names the same tensor under
// another owner; it is one object per spec value, so the same tensor under the same owner is always the same type.
//
// The release action is what release(t) does: pop or push a dataflow buffer's pages (compute_api/dataflow.h), or give
// a Dest section back to its ring (compute_api/dest.h). NoRelease does nothing. An op's result keeps the release of
// the tensor it wrote, so releasing the result releases that storage.
//
// An L1 Tensor's address is valid only on the thread that uses it: the unpack thread for inputs (it reads them), the
// pack thread for outputs (it writes them); on other threads it may be anything (a kernel's buffer pointer is 0 on
// math). So every llk-lib call that takes an L1 address runs on unpack or pack; no math-side call (matmul,
// eltwise_binary, copy, vec, the Dest sync) takes one.
//
// A kernel names a spec with l1_spec / dst_spec (or make_spec), which lay the tile grid out with tiled() and give it
// L1's dense strides or Dest's row strides (laid_out). The Resolver reads specs; llk-lib reads only the Plan.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensix_types.h" // DataFormat

namespace llk2
{

/// The dims of a TensorShape, outermost first: the tile grid, the faces in a tile, the datums in a face.
enum Dim : std::size_t
{
    TileY,  ///< tile row
    TileX,  ///< tile column
    FaceY,  ///< face row in a tile
    FaceX,  ///< face column in a tile
    DatumY, ///< row in a face
    DatumX  ///< column in a face
};

/// Where a tensor lives.
enum class Storage : std::uint8_t
{
    L1,  ///< L1 memory; the address is a byte address
    Dest ///< the Dest register; the address is a Dest row
};

/// A TRISC thread: the owner of a tensor, or the thread an llk-lib call runs on.
enum class Thread : std::uint8_t
{
    Unpack, ///< TRISC0
    Math,   ///< TRISC1
    Pack,   ///< TRISC2
    None    ///< nobody's yet, or a host build
};

/// An execution unit: the unit that produces or consumes a tensor.
enum class Exu : std::uint8_t
{
    Unpacker, ///< the unpackers
    Fpu,      ///< the matrix unit (FPU)
    Sfpu,     ///< the vector unit (SFPU)
    Packer,   ///< the packers
    None      ///< nobody's yet
};

/// Dims of every TensorShape: one past the innermost, DatumX.
inline constexpr std::size_t tensor_rank = DatumX + 1;

/// Sizes or strides of a TensorShape, one entry per Dim.
using Dims = std::array<std::uint32_t, tensor_rank>;

// ---- tile geometry constants (the same on every arch) ---------------------------------------------------------------
inline constexpr std::uint32_t face_cols          = 16;                 ///< datums in a face row; every face is this wide
inline constexpr std::uint32_t max_face_rows      = 16;                 ///< rows in a full face; shorter tiles have shorter faces
inline constexpr std::uint32_t default_tile_dim   = 32;                 ///< tile height and width when the caller names none
inline constexpr std::uint32_t dest_row_datums    = 16;                 ///< datums in one Dest row
inline constexpr std::uint32_t min_dest_face_rows = 8;                  ///< every face takes at least an 8-row slot in Dest (Quasar's tiny_face_stride)
inline constexpr std::uint32_t max_tiny_face_rows = min_dest_face_rows; ///< the tallest tiny tile: tiny tiles are 1, 2, 4 or 8 rows
inline constexpr std::uint32_t bfp_block_datums   = 16;                 ///< block float formats share one exponent byte per 16 datums
inline constexpr std::uint32_t l1_word_bytes      = 16;                 ///< L1 word: tiles in L1 are padded to whole words, and the unpacker's and
                                                                        ///< packer's addresses and tile sizes count words
inline constexpr std::uint32_t bits_per_byte      = 8;                  ///< bits in one byte, for datum_bits to bytes
inline constexpr std::uint32_t faces_per_face_row = default_tile_dim / face_cols; ///< faces side by side in a full tile: 2
inline constexpr std::uint32_t faces_per_full_tile =
    faces_per_face_row * (default_tile_dim / max_face_rows); ///< faces in a full 32x32 tile: 2 x 2, stored 0, 1 (top), 2, 3 (bottom)
inline constexpr std::uint32_t default_tile_datums = default_tile_dim * default_tile_dim; ///< datums in one 32x32 tile

/// Sizes and strides over Rank dims (Dim order), in datums, and the offset of datum 0 in datums.
template <std::size_t Rank>
struct TensorShape
{
    std::array<std::uint32_t, Rank> sizes;   ///< elements along each dim
    std::array<std::uint32_t, Rank> strides; ///< datums from one element to the next along each dim
    std::uint32_t offset;                    ///< datums from the address to datum 0
};

/// Everything about a tensor but its address. Compile time.
template <std::size_t Rank>
struct TensorSpec
{
    TensorShape<Rank> shape; ///< sizes, strides, offset
    DataFormat format;       ///< the datum format
    Storage storage;         ///< L1 or Dest
    Thread thread;           ///< the owner: its thread
    Exu exu;                 ///< the owner: its unit
};

/// The release action of a tensor that owns no storage: release() does nothing.
struct NoRelease
{
    /// Nothing to give back.
    constexpr void operator()() const
    {
    }
};

/// A tensor: spec S and release action R in its type, the address at run time. An L1 address is valid only on the
/// thread that reads or writes the tensor (see the top of this file). R is a base, so an empty one costs nothing.
template <const auto& S, class R = NoRelease>
struct Tensor : R
{
    static constexpr const auto& spec = S; ///< the tensor's spec
    std::uint32_t address;                 ///< L1: byte address of datum 0; Dest: the absolute first row

    /// A tensor at address a, released by r.
    constexpr explicit Tensor(std::uint32_t a, R r = R {}) : R(r), address(a)
    {
    }
};

/// Gives t's storage back: pops or pushes its buffer pages, or frees its Dest section. Call it once per storage, on
/// any of the tensors that share it (an op's result shares its output's release).
template <const auto& S, class R>
inline __attribute__((always_inline)) void release(const Tensor<S, R>& t)
{
    static_cast<const R&>(t)();
}

// ---- helpers over tensor_rank shapes and specs ---------------------------------------------------------------------

/// The smaller of a and b.
constexpr std::uint32_t min_u32(std::uint32_t a, std::uint32_t b)
{
    return a < b ? a : b;
}

/// The larger of a and b.
constexpr std::uint32_t max_u32(std::uint32_t a, std::uint32_t b)
{
    return a > b ? a : b;
}

/// a and b hold the same values (std::array's == is not constexpr in C++17).
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

/// L1 strides for sizes s: row-major in dim order.
constexpr Dims dense_strides(const Dims& s)
{
    Dims t {};
    t[DatumX] = 1; // datums are contiguous along x
    for (std::size_t d = DatumX; d-- > 0;)
    {
        t[d] = s[d + 1] * t[d + 1];
    }
    return t;
}

/// Dest strides for sizes s: rows of 16 datums, a face in at least an 8-row slot (every arch).
constexpr Dims dest_strides(const Dims& s)
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

/// f is a block float format (BFP8/4/2, A or B exponent).
constexpr bool is_bfp(DataFormat f)
{
    return f == DataFormat::Bfp8_b || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2_b || f == DataFormat::Bfp8 || f == DataFormat::Bfp4 ||
           f == DataFormat::Bfp2;
}

/// Bits of one datum's mantissa (and sign) in L1; 0 for formats no Tensor can be made in yet.
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

/// Rows of one tile.
constexpr std::uint32_t tile_rows(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceY] * s.sizes[DatumY];
}

/// Columns of one tile.
constexpr std::uint32_t tile_cols(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceX] * s.sizes[DatumX];
}

/// Tiles in the tensor.
constexpr std::uint32_t tile_count(const TensorShape<tensor_rank>& s)
{
    return s.sizes[TileY] * s.sizes[TileX];
}

/// Faces in one tile.
constexpr std::uint32_t faces_per_tile(const TensorShape<tensor_rank>& s)
{
    return s.sizes[FaceY] * s.sizes[FaceX];
}

/// value rounded up to a multiple of multiple.
constexpr std::uint32_t round_up(std::uint32_t value, std::uint32_t multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

/// Bytes of one tile of `datums` datums in L1: mantissas, plus one exponent byte per block for block formats; each
/// part padded to whole L1 words.
constexpr std::uint32_t tile_bytes(DataFormat f, std::uint32_t datums)
{
    const std::uint32_t exponents = is_bfp(f) ? round_up(datums / bfp_block_datums, l1_word_bytes) : 0;
    return round_up(datums * datum_bits(f) / bits_per_byte + exponents, l1_word_bytes);
}

/// Bytes of one of t's tiles in L1.
constexpr std::uint32_t tile_bytes(const TensorSpec<tensor_rank>& t)
{
    return tile_bytes(t.format, tile_rows(t.shape) * tile_cols(t.shape));
}

/// s has L1's dense strides.
constexpr bool is_contiguous(const TensorShape<tensor_rank>& s)
{
    return equal(s.strides, dense_strides(s.sizes));
}

/// a and b are the same tensor, whoever owns them.
constexpr bool same_except_owner(const TensorSpec<tensor_rank>& a, const TensorSpec<tensor_rank>& b)
{
    return equal(a.shape.sizes, b.shape.sizes) && equal(a.shape.strides, b.shape.strides) && a.shape.offset == b.shape.offset && a.format == b.format &&
           a.storage == b.storage;
}

/// a and b are equal by value, owner included (not by type).
constexpr bool same(const TensorSpec<tensor_rank>& a, const TensorSpec<tensor_rank>& b)
{
    return same_except_owner(a, b) && a.thread == b.thread && a.exu == b.exu;
}

/// t with owner (thread, exu).
constexpr TensorSpec<tensor_rank> with_owner(TensorSpec<tensor_rank> t, Thread thread, Exu exu)
{
    t.thread = thread;
    t.exu    = exu;
    return t;
}

/// The compile-time object for spec value V: one per value, so equal specs are one object and one Tensor type.
template <TensorSpec<tensor_rank> V>
inline constexpr TensorSpec<tensor_rank> spec_object = V;

/// The same tensor as S under owner (T, E), as a compile-time object a Tensor can name. Changing the owner twice
/// gives the same object as changing it once.
template <const auto& S, Thread T, Exu E>
inline constexpr const TensorSpec<tensor_rank>& owned_by = spec_object<with_owner(S, T, E)>;

/// t under owner (T, E): the same address and release, the owner changed in the type.
template <Thread T, Exu E, const auto& S, class R>
constexpr Tensor<owned_by<S, T, E>, R> as_owned(const Tensor<S, R>& t)
{
    return Tensor<owned_by<S, T, E>, R>(t.address, t);
}

// ---- constructing: the tile is named, or min(dim, 32) per axis -------------------------------------------------

/// n is a power of two.
constexpr bool is_power_of_two(std::uint32_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

/// An h x w tile can be laid out: whole faces wide; tiny (a power of two up to max_tiny_face_rows rows) or a whole
/// number of full faces tall.
constexpr bool is_valid_tile(std::uint32_t h, std::uint32_t w)
{
    const bool tiny_height = is_power_of_two(h) && h <= max_tiny_face_rows;
    return w > 0 && w % face_cols == 0 && (tiny_height || (h > 0 && h % max_face_rows == 0));
}

/// Sizes of a tiles_y x tiles_x grid of h x w tiles: faces of min(h, 16) rows and face_cols columns.
constexpr Dims tiled(std::uint32_t tiles_y, std::uint32_t tiles_x, std::uint32_t h, std::uint32_t w)
{
    const std::uint32_t face_rows = min_u32(h, max_face_rows);
    return {tiles_y, tiles_x, h / face_rows, w / face_cols, face_rows, face_cols};
}

/// A spec of sizes in format f, with the strides of `storage`, offset 0 and no owner.
constexpr TensorSpec<tensor_rank> laid_out(const Dims& sizes, DataFormat f, Storage storage)
{
    const Dims strides = storage == Storage::Dest ? dest_strides(sizes) : dense_strides(sizes);
    return {{sizes, strides, 0}, f, storage, Thread::None, Exu::None}; // nobody's: nothing has produced it yet
}

/// The spec of a Rows x Cols tensor in format F, cut into H x W tiles, in `storage`.
template <std::uint32_t Rows, std::uint32_t Cols, DataFormat F, std::uint32_t H, std::uint32_t W>
constexpr TensorSpec<tensor_rank> make_spec(Storage storage)
{
    static_assert(is_valid_tile(H, W), "a tile is 16k wide and 1, 2, 4, 8 or 16k tall: pad the tensor first");
    static_assert(Rows % H == 0 && Cols % W == 0, "the tile must divide the tensor: pad it first");
    return laid_out(tiled(Rows / H, Cols / W, H, W), F, storage);
}

/// The spec of a Rows x Cols L1 tensor in format F, in H x W tiles (default: min(dim, 32)).
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = min_u32(Rows, default_tile_dim),
    std::uint32_t W = min_u32(Cols, default_tile_dim)>
inline constexpr TensorSpec<tensor_rank> l1_spec = make_spec<Rows, Cols, F, H, W>(Storage::L1);

/// The spec of a Rows x Cols Dest tensor in format F, in H x W tiles (default: min(dim, 32)).
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = min_u32(Rows, default_tile_dim),
    std::uint32_t W = min_u32(Cols, default_tile_dim)>
inline constexpr TensorSpec<tensor_rank> dst_spec = make_spec<Rows, Cols, F, H, W>(Storage::Dest);

/// An L1 tensor of l1_spec<Rows, Cols, F, H, W> at byte address `address`.
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = min_u32(Rows, default_tile_dim),
    std::uint32_t W = min_u32(Cols, default_tile_dim)>
constexpr auto l1_tensor(std::uint32_t address)
{
    return Tensor<l1_spec<Rows, Cols, F, H, W>>(address);
}

/// A Dest tensor of dst_spec<Rows, Cols, F, H, W> from Dest row `row`.
template <
    std::uint32_t Rows,
    std::uint32_t Cols,
    DataFormat F,
    std::uint32_t H = min_u32(Rows, default_tile_dim),
    std::uint32_t W = min_u32(Cols, default_tile_dim)>
constexpr auto dst_tensor(std::uint32_t row)
{
    return Tensor<dst_spec<Rows, Cols, F, H, W>>(row);
}

/// Sizes of [M, K] x [K, N]: rows from a, columns from b.
constexpr Dims product_sizes(const Dims& a, const Dims& b)
{
    return {a[TileY], b[TileX], a[FaceY], b[FaceX], a[DatumY], b[DatumX]};
}

} // namespace llk2
