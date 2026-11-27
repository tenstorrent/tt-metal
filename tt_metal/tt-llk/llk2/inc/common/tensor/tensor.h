// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensor: the operand type every LLK 2.0 layer takes. Arch-independent; its outside dependencies are DataFormat
// (tensix_types.h) and LLK_ASSERT (tt-llk's llk_assert.h), for the checks of a view's run-time start and length.
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
//
// In this file, in this order:
//   vocabulary        the dims, where a tensor lives, who owns it, the tile geometry constants
//   shape and spec    TensorShape, TensorSpec (Shape, Spec), and what can be asked of one
//   naming a spec     l1_spec, dst_spec
//   Tensor            the Tensor, its release action, l1_tensor, dst_tensor
//   owner             the same tensor under another owner: owned_by, as_owned
//   run-time sizes    a size that is a bound in the spec and a count in the Tensor: runtime_rows, runtime_cols
//   views             a block of an L1 tensor's tile grid: rows, cols, row, tile, and the general narrow and select

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_assert.h"
#include "tensix_types.h" // DataFormat

namespace llk2
{

// ---- vocabulary -----------------------------------------------------------------------------------------------------

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

// Tile geometry constants (the same on every arch).
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

// ---- shape and spec -------------------------------------------------------------------------------------------------

/// Sizes and strides over Rank dims (Dim order), in datums, and the offset of datum 0 in datums.
template <std::size_t Rank>
struct TensorShape
{
    std::array<std::uint32_t, Rank> sizes;   ///< elements along each dim
    std::array<std::uint32_t, Rank> strides; ///< datums from one element to the next along each dim
    std::uint32_t offset;                    ///< datums from the address to datum 0
    /// Bit d set: sizes[d] is only an upper bound, and each Tensor holds the actual count ("run-time sizes", below).
    /// At most one bit, of a tile-grid dim: a Tensor holds one count. Strides and offset are compile time always.
    std::uint32_t runtime = 0;
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

/// The shape and the spec of every tensor in the stack: tensor_rank dims, the only rank there is.
using Shape = TensorShape<tensor_rank>;
using Spec  = TensorSpec<tensor_rank>;

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

/// Datum rows of one tile.
constexpr std::uint32_t tile_height(const Shape& s)
{
    return s.sizes[FaceY] * s.sizes[DatumY];
}

/// Datum columns of one tile.
constexpr std::uint32_t tile_width(const Shape& s)
{
    return s.sizes[FaceX] * s.sizes[DatumX];
}

/// Tiles in the tensor.
constexpr std::uint32_t tile_count(const Shape& s)
{
    return s.sizes[TileY] * s.sizes[TileX];
}

/// s is cut into 32x32 tiles: the one tile size every op is ported for.
constexpr bool has_default_tiles(const Shape& s)
{
    return tile_height(s) == default_tile_dim && tile_width(s) == default_tile_dim;
}

/// Faces in one tile.
constexpr std::uint32_t faces_per_tile(const Shape& s)
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
    const std::uint32_t exponents = llk2::is_bfp(f) ? round_up(datums / bfp_block_datums, l1_word_bytes) : 0;
    return round_up(datums * datum_bits(f) / bits_per_byte + exponents, l1_word_bytes);
}

/// Bytes of one of t's tiles in L1.
constexpr std::uint32_t tile_bytes(const Spec& t)
{
    return tile_bytes(t.format, tile_height(t.shape) * tile_width(t.shape));
}

/// s has L1's dense strides.
constexpr bool is_contiguous(const Shape& s)
{
    return equal(s.strides, dense_strides(s.sizes));
}

/// s is contiguous, or a window of a wider tile grid: dense inside each tile row, with its tile rows a whole number
/// of tiles apart (tile_row_step). What an op whose Resolver reads the tile-row step accepts.
constexpr bool has_dense_tile_rows(const Shape& s)
{
    Dims dense                   = dense_strides(s.sizes);
    const std::uint32_t row_step = s.strides[TileY];
    const bool rows_apart        = row_step >= dense[TileY] && row_step % dense[TileX] == 0;
    dense[TileY]                 = row_step;
    return rows_apart && equal(s.strides, dense);
}

/// Tiles from one tile row of an L1 tensor to the next: sizes[TileX] when contiguous, more for a window (a view of
/// some tile columns of a wider tensor: cols, below).
constexpr std::uint32_t tile_row_step(const Shape& s)
{
    return s.strides[TileY] / s.strides[TileX];
}

/// Whether size d of t is a run-time count (t then holds its upper bound): "run-time sizes", below.
constexpr bool is_runtime_size(const Spec& t, Dim d)
{
    return ((t.shape.runtime >> d) & 1u) != 0;
}

/// Whether any size of t is a run-time count.
constexpr bool has_runtime_size(const Spec& t)
{
    return t.shape.runtime != 0;
}

/// a and b mark the same size run time, or neither marks one.
constexpr bool same_runtime_size(const Spec& a, const Spec& b)
{
    return a.shape.runtime == b.shape.runtime;
}

/// a and b are the same tensor, whoever owns them.
constexpr bool same_except_owner(const Spec& a, const Spec& b)
{
    return equal(a.shape.sizes, b.shape.sizes) && equal(a.shape.strides, b.shape.strides) && a.shape.offset == b.shape.offset && a.format == b.format &&
           a.storage == b.storage && same_runtime_size(a, b);
}

/// a and b are equal by value, owner included (not by type).
constexpr bool same(const Spec& a, const Spec& b)
{
    return same_except_owner(a, b) && a.thread == b.thread && a.exu == b.exu;
}

/// The compile-time object for spec value V: one per value, so equal specs are one object and one Tensor type.
template <Spec V>
inline constexpr Spec spec_object = V;

// ---- naming a spec: the tile is named, or min(dim, 32) per axis -----------------------------------------------------

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
constexpr Spec laid_out(const Dims& sizes, DataFormat f, Storage storage)
{
    const Dims strides = storage == Storage::Dest ? dest_strides(sizes) : dense_strides(sizes);
    return {{sizes, strides, 0}, f, storage, Thread::None, Exu::None}; // nobody's: nothing has produced it yet
}

/// The spec of a tensor of Height x Width datums in format F, cut into tiles of TileH x TileW datums, in `storage`.
template <std::uint32_t Height, std::uint32_t Width, DataFormat F, std::uint32_t TileH, std::uint32_t TileW>
constexpr Spec make_spec(Storage storage)
{
    static_assert(is_valid_tile(TileH, TileW), "a tile's width is a multiple of 16 and its height is 1, 2, 4, 8 or a multiple of 16: pad the tensor first");
    static_assert(Height % TileH == 0 && Width % TileW == 0, "the tile must divide the tensor: pad it first");
    return laid_out(tiled(Height / TileH, Width / TileW, TileH, TileW), F, storage);
}

/// The spec of an L1 tensor of Height x Width datums in format F, in tiles of TileH x TileW datums (default: min(dim, 32)).
template <
    std::uint32_t Height,
    std::uint32_t Width,
    DataFormat F,
    std::uint32_t TileH = min_u32(Height, default_tile_dim),
    std::uint32_t TileW = min_u32(Width, default_tile_dim)>
inline constexpr Spec l1_spec = make_spec<Height, Width, F, TileH, TileW>(Storage::L1);

/// The spec of a Dest tensor of Height x Width datums in format F, in tiles of TileH x TileW datums (default: min(dim, 32)).
template <
    std::uint32_t Height,
    std::uint32_t Width,
    DataFormat F,
    std::uint32_t TileH = min_u32(Height, default_tile_dim),
    std::uint32_t TileW = min_u32(Width, default_tile_dim)>
inline constexpr Spec dst_spec = make_spec<Height, Width, F, TileH, TileW>(Storage::Dest);

// ---- Tensor ---------------------------------------------------------------------------------------------------------

/// The release action of a tensor that owns no storage: release() does nothing.
struct NoRelease
{
    /// Nothing to give back.
    constexpr void operator()() const
    {
    }
};

/// A tensor's run-time count, or nothing: "run-time sizes", below.
template <bool HasRuntimeSize>
struct RuntimeCount;

/// A tensor: spec S and release action R in its type, the address at run time. An L1 address is valid only on the
/// thread that reads or writes the tensor (see the top of this file). R is a base, so an empty one costs nothing.
/// A spec with a run-time size adds that count as the member `count` (RuntimeCount: "run-time sizes", below).
template <const auto& S, class R = NoRelease>
struct Tensor : R, RuntimeCount<has_runtime_size(S)>
{
    static constexpr const auto& spec = S; ///< the tensor's spec
    std::uint32_t address;                 ///< L1: byte address of datum 0; Dest: the absolute first row

    /// A tensor at address a, released by r.
    constexpr explicit Tensor(std::uint32_t a, R r = R {}) : R(r), address(a)
    {
        static_assert(!has_runtime_size(S), "a tensor whose spec marks a run-time size takes its count: Tensor(address, release, count)");
    }

    /// A tensor with a run-time size at address a, released by r: `count` elements along the dim its spec marks.
    constexpr Tensor(std::uint32_t a, R r, std::uint32_t count) : R(r), RuntimeCount<has_runtime_size(S)> {count}, address(a)
    {
        static_assert(has_runtime_size(S), "the spec marks no run-time size: Tensor(address, release)");
    }
};

/// Gives t's storage back: pops or pushes its buffer pages, or frees its Dest section. Call it once per storage, on
/// any of the tensors that share it (an op's result shares its output's release).
template <const auto& S, class R>
inline __attribute__((always_inline)) void release(const Tensor<S, R>& t)
{
    static_cast<const R&>(t)();
}

/// An L1 tensor of l1_spec<Height, Width, F, TileH, TileW> at byte address `address`.
template <
    std::uint32_t Height,
    std::uint32_t Width,
    DataFormat F,
    std::uint32_t TileH = min_u32(Height, default_tile_dim),
    std::uint32_t TileW = min_u32(Width, default_tile_dim)>
constexpr auto l1_tensor(std::uint32_t address)
{
    return Tensor<l1_spec<Height, Width, F, TileH, TileW>>(address);
}

/// A Dest tensor of dst_spec<Height, Width, F, TileH, TileW> from Dest row `row`.
template <
    std::uint32_t Height,
    std::uint32_t Width,
    DataFormat F,
    std::uint32_t TileH = min_u32(Height, default_tile_dim),
    std::uint32_t TileW = min_u32(Width, default_tile_dim)>
constexpr auto dst_tensor(std::uint32_t row)
{
    return Tensor<dst_spec<Height, Width, F, TileH, TileW>>(row);
}

// ---- owner ----------------------------------------------------------------------------------------------------------

/// t with owner (thread, exu).
constexpr Spec with_owner(Spec t, Thread thread, Exu exu)
{
    t.thread = thread;
    t.exu    = exu;
    return t;
}

/// The same tensor as S under owner (T, E), as a compile-time object a Tensor can name. Changing the owner twice
/// gives the same object as changing it once.
template <const auto& S, Thread T, Exu E>
inline constexpr const Spec& owned_by = spec_object<with_owner(S, T, E)>;

/// t under owner (T, E): the same address and release, the owner changed in the type. The release argument is t
/// itself: a Tensor is its release action (R is its base), so t converts to it.
template <Thread T, Exu E, const auto& S, class R>
constexpr Tensor<owned_by<S, T, E>, R> as_owned(const Tensor<S, R>& t)
{
    if constexpr (has_runtime_size(S))
    {
        return Tensor<owned_by<S, T, E>, R>(t.address, t, t.count); // the count goes with it
    }
    else
    {
        return Tensor<owned_by<S, T, E>, R>(t.address, t);
    }
}

// ---- run-time sizes -------------------------------------------------------------------------------------------------
//
// A spec may mark one of its tile-grid sizes run time (runtime_rows / runtime_cols, or a view with a run-time length,
// below): the spec then holds that size's upper BOUND, which every compile-time decision is made on (does it fit a
// Dest section, a MOP count, an address counter), and each Tensor of the spec holds the actual count, 1..bound.
// Everything else about the tensor, its strides included, stays compile time.
//
// This is the design, not a placeholder: a bound and a mark in the spec, the count in the Tensor. A spec is a
// structural value (a template argument: the Resolver, the Plans and llk-lib's params are computed from it at compile
// time), and a structural value cannot hold a run-time number, so the number travels beside it, in the object. The
// ct / rt types (wrapper/wrapper.h) start where a count is a function argument: llk-lib's execute (resolver/work_counts.h).

/// t with its size d marked run time. One run-time size per tensor, a tile-grid one: marking another replaces it.
constexpr Spec with_runtime_size(Spec t, Dim d)
{
    t.shape.runtime = 1u << d;
    return t;
}

/// The compile-time object of S with its size D marked run time.
template <const auto& S, Dim D>
inline constexpr const Spec& runtime_size_spec = spec_object<with_runtime_size(S, D)>;

/// S with a run-time number of tile rows, at most S's: wait_front<runtime_rows<S>>(buffer, n).
template <const auto& S>
inline constexpr const Spec& runtime_rows = runtime_size_spec<S, TileY>;

/// S with a run-time number of tile columns, at most S's: wait_front<runtime_cols<S>>(buffer, n).
template <const auto& S>
inline constexpr const Spec& runtime_cols = runtime_size_spec<S, TileX>;

/// A tensor's run-time count, a base of Tensor. Nothing, for a spec with no run-time size: an empty base.
template <bool HasRuntimeSize>
struct RuntimeCount
{
};

/// The actual count of the size the spec marks run time.
template <>
struct RuntimeCount<true>
{
    std::uint32_t count; ///< elements along the run-time dim: 1..the spec's size
};

/// t as a tensor holding `count` of its size D, of which t's spec has the bound: the same address and release. What an
/// op returns for a Dest tensor it wrote only `count` tile rows or columns of.
template <Dim D, const auto& S, class R>
constexpr auto with_count(const Tensor<S, R>& t, std::uint32_t count)
{
    return Tensor<runtime_size_spec<S, D>, R>(t.address, t, count);
}

// ---- views: a block of an L1 tensor's tile grid, at a compile-time or run-time place --------------------------------
//
// Two general views, with torch's names, over the tile grid's two dims by name (Rows, Cols):
//
//   llk2::narrow<Rows, 2>(s, r0)                              // 2 tile rows of s from row r0 (run time):    [2, Sk]
//   llk2::narrow<Cols, 8, 0>(s)                               // the first 8 tile columns (compile time):    [G, 8]
//   llk2::select<Rows>(s, i)                                  // tile row i:                                 [1, Sk]
//
// and the names a kernel reads best, each one of the above:
//
//   const auto s = llk2::wait_front<rows_spec>(scores);      // [G, Sk] tiles
//   llk2::row(s, i)                                           // select<Rows>(s, i)
//   llk2::rows<2>(s, r0)                                      // narrow<Rows, 2>(s, r0)
//   llk2::cols<8>(llk2::row(s, i), c0)                        // narrow<Cols, 8>(...): 8 tiles of row i from c0: [1, 8]
//   llk2::tile(m, i)                                          // tile i of a column of tiles
//
// The LENGTH is a template argument: it decides the view's spec, and every op needs its operands' specs at compile
// time. The START is a function argument when it is known at run time only (it moves the address and nothing else),
// or a third template argument when it is a constant: then start + length is checked against the tensor at compile
// time. A run-time start is checked when the kernel runs, in builds with LLK asserts.
//
// A LENGTH known only at run time has its own names, narrow_up_to, rows_up_to and cols_up_to: there the template
// argument is the length's upper BOUND (what the view's spec holds, and every compile-time decision is made on) and
// the length is the last function argument (what the view holds: "run-time sizes", above):
//
//   llk2::cols_up_to<8>(llk2::row(s, i), c0, n)               // n (1..8) tiles of row i from c0: [1, <= 8]
//   llk2::rows_up_to<4>(v, 0, n)                              // the first n (1..4) tile rows
//
// A view is a Tensor of its own spec (the block's size is in its type) at an address inside the tensor it was taken
// from, with that tensor's format, tile and owner. It owns no storage: release the tensor it was taken from. Ops take
// a view wherever they take an L1 tensor: as an input (the unpack thread reads it) or as pack's destination.
//
// Unlike torch's: select keeps its dim, with size 1 (a spec always has the whole tile grid, [1, Sk] and not [Sk]), and
// a view's strides are put in one form (block_of), so that equal blocks are one spec whatever they were cut from.

/// The tile grid's dims by the names a view takes: narrow<Rows, N>, select<Cols>. Rows and Cols always mean tile rows
/// and tile columns; a size in datums is a height or a width.
inline constexpr Dim Rows = TileY; ///< tile rows
inline constexpr Dim Cols = TileX; ///< tile columns

/// The spec of a rows x cols tile block of s's tile grid. With one tile row, or as wide as a contiguous s, the block
/// is contiguous in L1; otherwise its tile rows step by s's tile-row step (a window: only ops whose Resolver reads
/// that step accept it, has_dense_tile_rows).
constexpr Spec block_of(Spec s, std::uint32_t rows, std::uint32_t cols)
{
    const std::uint32_t row_step = s.shape.strides[TileY];
    const bool whole_rows        = cols == s.shape.sizes[TileX] && row_step == s.shape.sizes[TileX] * s.shape.strides[TileX];
    s.shape.sizes[TileY]         = rows;
    s.shape.sizes[TileX]         = cols;
    s.shape.strides              = dense_strides(s.shape.sizes);
    if (rows > 1 && !whole_rows)
    {
        s.shape.strides[TileY] = row_step;
    }
    return s;
}

/// s with `length` tiles along dim (Rows or Cols) and all of its other dim.
constexpr Spec narrowed(const Spec& s, Dim dim, std::uint32_t length)
{
    return dim == Rows ? block_of(s, length, s.shape.sizes[TileX]) : block_of(s, s.shape.sizes[TileY], length);
}

/// The compile-time object of narrowed(S, D, Length): one per value, so equal blocks are one Tensor type.
template <const auto& S, Dim D, std::uint32_t Length>
inline constexpr const Spec& narrowed_spec = spec_object<narrowed(S, D, Length)>;

/// Tiles from one element of s to the next along dim (Rows or Cols): a tile row's step, or one tile.
constexpr std::uint32_t tiles_per_element(const Shape& s, Dim dim)
{
    return dim == Rows ? tile_row_step(s) : 1;
}

// With ENABLE_LLK_ASSERT, LLK_ASSERT is an asm statement, which C++17 flags in a constexpr function (C++20 allows it,
// and it is never reached in a constant evaluation that passes). -Wpragmas: older compilers do not know the option.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wc++20-extensions"

/// Length tile rows (D = Rows) or tile columns (D = Cols) of t, from its row or column `start`.
template <Dim D, std::uint32_t Length, const auto& S, class R>
constexpr auto narrow(const Tensor<S, R>& t, std::uint32_t start)
{
    static_assert(S.storage == Storage::L1, "views are of L1 tensors");
    static_assert(D == Rows || D == Cols, "a view is of the tile grid: narrow<Rows, N> or narrow<Cols, N>");
    static_assert(
        !has_runtime_size(S),
        "a view of a tensor with a run-time size: not ported. Take the tensor under its spec without the run-time size, and give the length to the view "
        "(rows_up_to / cols_up_to)");
    static_assert(Length >= 1 && Length <= S.shape.sizes[D], "narrow: the length is from 1 to the tensor's size along the dim");
    LLK_ASSERT(start + Length <= S.shape.sizes[D], "narrow: start + length is past the tensor's end");
    return Tensor<narrowed_spec<S, D, Length>>(t.address + start * tiles_per_element(S.shape, D) * tile_bytes(S));
}

/// narrow with a run-time length: `length` (1..Bound) elements along D from element `start`. Bound is what every
/// compile-time decision about the view is made on (its spec is narrow<D, Bound>'s with D marked run time); the view
/// holds `length`. Its strides are the parent's, compile time: a view's tile rows step as t's do, whatever the
/// run-time number of tile columns.
template <Dim D, std::uint32_t Bound, const auto& S, class R>
constexpr auto narrow_up_to(const Tensor<S, R>& t, std::uint32_t start, std::uint32_t length)
{
    static_assert(S.storage == Storage::L1, "views are of L1 tensors");
    static_assert(D == Rows || D == Cols, "a view is of the tile grid: narrow_up_to<Rows, N> or narrow_up_to<Cols, N>");
    static_assert(
        !has_runtime_size(S),
        "a view of a tensor with a run-time size: not ported. Take the tensor under its spec without the run-time size, and give the length to the view "
        "(rows_up_to / cols_up_to)");
    static_assert(Bound >= 1 && Bound <= S.shape.sizes[D], "narrow_up_to: the length's bound is from 1 to the tensor's size along the dim");
    LLK_ASSERT(length >= 1 && length <= Bound && start + length <= S.shape.sizes[D], "narrow_up_to: the run-time length is outside the view or the tensor");
    return Tensor<runtime_size_spec<narrowed_spec<S, D, Bound>, D>>(t.address + start * tiles_per_element(S.shape, D) * tile_bytes(S), NoRelease {}, length);
}

#pragma GCC diagnostic pop

/// narrow from a compile-time Start: checked to lie inside t.
template <Dim D, std::uint32_t Length, std::uint32_t Start, const auto& S, class R>
constexpr auto narrow(const Tensor<S, R>& t)
{
    static_assert(D == Rows || D == Cols, "a view is of the tile grid: narrow<Rows, N, Start> or narrow<Cols, N, Start>");
    static_assert(Start + Length <= S.shape.sizes[D], "narrow: start + length is past the tensor's end");
    return narrow<D, Length>(t, Start);
}

/// Tile row (D = Rows) or tile column (D = Cols) `index` of t. The dim stays, with size 1.
template <Dim D, const auto& S, class R>
constexpr auto select(const Tensor<S, R>& t, std::uint32_t index)
{
    static_assert(D == Rows || D == Cols, "a view is of the tile grid: select<Rows> or select<Cols>");
    return narrow<D, 1>(t, index);
}

/// N tile rows of t, from its tile row r0: narrow<Rows, N>.
template <std::uint32_t N, const auto& S, class R>
constexpr auto rows(const Tensor<S, R>& t, std::uint32_t r0)
{
    return narrow<Rows, N>(t, r0);
}

/// n (run time, 1..Bound) tile rows of t, from its tile row r0: narrow_up_to<Rows, Bound>.
template <std::uint32_t Bound, const auto& S, class R>
constexpr auto rows_up_to(const Tensor<S, R>& t, std::uint32_t r0, std::uint32_t n)
{
    return narrow_up_to<Rows, Bound>(t, r0, n);
}

/// Tile row r of t: select<Rows>.
template <const auto& S, class R>
constexpr auto row(const Tensor<S, R>& t, std::uint32_t r)
{
    return select<Rows>(t, r);
}

/// N tile columns of t, from its tile column c0: narrow<Cols, N>.
template <std::uint32_t N, const auto& S, class R>
constexpr auto cols(const Tensor<S, R>& t, std::uint32_t c0)
{
    return narrow<Cols, N>(t, c0);
}

/// n (run time, 1..Bound) tile columns of t, from its tile column c0: narrow_up_to<Cols, Bound>.
template <std::uint32_t Bound, const auto& S, class R>
constexpr auto cols_up_to(const Tensor<S, R>& t, std::uint32_t c0, std::uint32_t n)
{
    return narrow_up_to<Cols, Bound>(t, c0, n);
}

/// Tile i of a column of tiles (a tensor one tile wide).
template <const auto& S, class R>
constexpr auto tile(const Tensor<S, R>& t, std::uint32_t i)
{
    static_assert(S.shape.sizes[TileX] == 1, "tile(t, i) is for a tensor one tile wide: use row and cols");
    return select<Rows>(t, i);
}

} // namespace llk2
