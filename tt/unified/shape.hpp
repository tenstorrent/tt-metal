// SPDX-License-Identifier: Apache-2.0
//
// Compile-time block shapes.
//
// Deliberately op-agnostic, the same way tt/unified/expr.hpp is: this header knows
// the shape ALGEBRA -- rank, extents, the page count, and how to rebuild a shape with
// its innermost two dimensions replaced -- and nothing about what an op does with it.
// tt/unified/math.hpp applies it (reduce_shape, matmul_shape).
//
// Why compile-time: an audit of the 1294 kernel sources under
// ttnn/cpp/ttnn/operations/ found block shape is 89% compile-time (311 declarations
// against 38) while ITERATION COUNT is 81% runtime (335 against 80). The split is a
// convention, stated outright by ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:
//
//     template <uint32_t block_width_tiles, ...>
//     ALWI void tilize(uint32_t num_blocks, ...);
//
// Shape in the template parameter, count in the function argument. So Shape is static
// and the block index, block count and `finish` flags stay runtime.
//
// Shape equality is TYPE identity, which is the point of the exercise: Shape<1, 4> and
// Shape<4> hold the same number of pages and are different types, so the mistake a
// single num_entries count cannot see does not compile.
//
// A future DynamicShape, used in place of Shape, is the escape for the genuinely
// runtime-extent cases (45% of Ht/Wt uses in ttnn are runtime). Nothing here should
// preclude one.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tt {
namespace unified {

// The ELEMENT extent of one dataflow-buffer page, defined below. Declared here because
// Shape names it as a default.
template <uint32_t Rows, uint32_t Cols>
struct Tile;

// Innermost dimension LAST, the usual convention: Shape<4, 4> is a 4x4 block of tiles,
// Shape<2, 4, 4> is two of them. Only the last two dimensions take part in the math
// ops; anything to their left is a leading extent the kernel's own loop walks.
template <uint32_t... Dims>
struct Shape {
    static constexpr uint32_t rank = sizeof...(Dims);
    static_assert(rank > 0, "a Shape needs at least one dimension");

    static constexpr uint32_t dims[rank] = {Dims...};

    // dim(2) is the third dimension; dim(-1) is the last and dim(-rank) the first, the
    // way Python indexes. Negative is the useful half: a shape's meaning is anchored at
    // the END -- dim(-1) is columns and dim(-2) rows whatever the rank -- so counting
    // back is how the code that does not care about rank wants to ask.
    //
    // Out of range is a compile error, not a wrong answer. `dims[i]` outside the array is
    // ill-formed in a constant expression, and a negative index past -rank wraps to a
    // large unsigned value that is equally ill-formed, so both fail wherever this is
    // evaluated at compile time -- which is everywhere a Shape is asked about itself.
    static constexpr uint32_t dim(int32_t i) {
        return dims[i < 0 ? static_cast<uint32_t>(static_cast<int32_t>(rank) + i) : static_cast<uint32_t>(i)];
    }

    // One tile per dataflow-buffer page in v1, so the page count is the product. This
    // is the name the DFB protocol speaks in, and it stays that name -- reading it off
    // an instance (`storage.num_entries`) still compiles now that it is static.
    static constexpr uint32_t num_entries = (uint32_t{1} * ... * Dims);

    static constexpr uint32_t cols = dims[rank - 1];
    // The guard is repeated inside the index because both arms of a ternary are
    // parsed: at rank 1 `rank - 2` would wrap.
    static constexpr uint32_t rows = rank >= 2 ? dims[rank >= 2 ? rank - 2 : 0] : 1;
    static constexpr uint32_t leading = num_entries / (rows * cols);

    // The tile geometry these counts are counts OF. A full 32x32 tile unless the shape is
    // re-tiled with Tiled<> below, which is every shape in this repo and every shape any
    // existing kernel spells -- so this member is what keeps that spelling unchanged.
    using tile = Tile<32, 32>;
};

// One tile: the element extent of a single page.
//
// Reuses Shape for its arithmetic through `extent` rather than BEING a Shape, because the
// two mean different things and the names would collide -- Shape<32, 32> is a 32x32 block
// of TILES, Tile<32, 32> is one tile of 32x32 ELEMENTS. Conflating them reads fine until
// someone writes Shape<1, 32> meaning a row-form tile and gets a 1x32 block of full tiles.
template <uint32_t Rows, uint32_t Cols>
struct Tile {
    using extent = Shape<Rows, Cols>;
    static constexpr uint32_t rows = Rows;
    static constexpr uint32_t cols = Cols;
    static constexpr uint32_t elements = Rows * Cols;
};

// The default, named so a re-tiling back to it can be canonicalised away.
using TileFull = Tile<32, 32>;

// An existing Shape, re-tiled. Inherits every member -- it IS that shape, with a different
// page geometry -- so nothing that reads rows/cols/num_entries/dim needs to know.
//
// A wrapper rather than a template parameter on Shape, and that is forced rather than
// chosen: a defaulted leading TYPE parameter cannot be omitted when a non-type pack
// follows it, so `Shape<4, 4>` could never keep meaning what it means if the tile were
// `Shape<TileShape, Dims...>`. There is no ordering that works -- the pack must be last.
template <typename TileShape, typename S>
struct Tiled : S {
    using tile = TileShape;
};

// The plain Shape underneath, for helpers that must rebuild one.
template <typename S>
struct base_shape_impl {
    using type = S;
};
template <typename TileShape, typename S>
struct base_shape_impl<Tiled<TileShape, S>> {
    using type = S;
};
template <typename S>
using base_shape_t = typename base_shape_impl<S>::type;

// Re-tiling to the DEFAULT yields the plain Shape rather than Tiled<TileFull, Shape<...>>.
// Without this every derived shape would come back wrapped and stop comparing equal to the
// Shape<...> a kernel wrote, which would turn a type-identity check into a false alarm
// everywhere.
template <typename TileShape, typename S>
struct retile_impl {
    using type = Tiled<TileShape, base_shape_t<S>>;
};
template <typename S>
struct retile_impl<TileFull, S> {
    using type = base_shape_t<S>;
};
template <typename TileShape, typename S>
using retile = typename retile_impl<TileShape, S>::type;

// Shape equality, spelled for legibility inside static_asserts. Type identity, so a
// re-tiled shape is NOT equal to the same tile counts at full geometry -- which is the
// point: it makes a geometry mismatch a build error rather than sparse data.
template <typename A, typename B>
inline constexpr bool same_shape_v = std::is_same<A, B>::value;

// ---------------------------------------------------------------------------
// LOGICAL extents: the innermost two dimensions multiplied out by the tile.
//
// Free functions rather than members of Shape, and that is also forced. Tiled<> overrides
// `tile` by inheritance, and a static member function of Shape would read Shape's OWN
// tile, not the derived override -- statics do not dispatch. A free template reads
// `S::tile` at the call site, where S is the type actually being asked about.
// ---------------------------------------------------------------------------

template <typename S>
inline constexpr uint32_t logical_rows_v = S::rows * S::tile::rows;

template <typename S>
inline constexpr uint32_t logical_cols_v = S::cols * S::tile::cols;

// The i-th dimension in ELEMENTS. Indexed like dim(): the innermost two are scaled by the
// tile, and anything to their left is a leading extent that tiles do not touch.
template <typename S>
static constexpr uint32_t logical_dim(int32_t i) {
    const uint32_t idx = i < 0 ? static_cast<uint32_t>(static_cast<int32_t>(S::rank) + i) : static_cast<uint32_t>(i);
    return idx == S::rank - 1 ? logical_cols_v<S>
                              : (idx == S::rank - 2 ? logical_rows_v<S> : S::dim(static_cast<int32_t>(idx)));
}

// Two shapes hold the same ELEMENTS in the same arrangement, whatever their tiling. The
// weaker question than same_shape_v, for the places where a differently-tiled operand is
// legitimate as long as the element extents line up.
template <typename A, typename B>
inline constexpr bool same_logical_hw_v =
    logical_rows_v<A> == logical_rows_v<B> && logical_cols_v<A> == logical_cols_v<B>;

// Rebuild a shape with its innermost two dimensions replaced, preserving everything to
// their left. One helper covers every derived shape the ops need, at any rank -- which
// is what makes variadic rank free rather than a later migration.
template <typename S, uint32_t H, uint32_t W, typename Idx>
struct with_hw_impl;

template <uint32_t... D, uint32_t H, uint32_t W, std::size_t... I>
struct with_hw_impl<Shape<D...>, H, W, std::index_sequence<I...>> {
    using S = Shape<D...>;
    using type = Shape<(I == S::rank - 2 ? H : (I == S::rank - 1 ? W : S::dim(static_cast<int32_t>(I))))...>;
};

// Carries the TILE through: a derived shape of a re-tiled operand is tiled the same way.
// with_hw_impl is specialised on Shape<D...> exactly, so the wrapper has to be stripped
// before rebuilding and put back after.
template <typename S, uint32_t H, uint32_t W>
using with_hw =
    retile<typename S::tile, typename with_hw_impl<base_shape_t<S>, H, W, std::make_index_sequence<S::rank>>::type>;

}  // namespace unified
}  // namespace tt
