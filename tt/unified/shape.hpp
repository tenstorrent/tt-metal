// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tt {
namespace unified {

template <uint32_t Rows, uint32_t Cols>
struct Tile;

template <uint32_t... Dims>
struct Shape {
    static constexpr uint32_t rank = sizeof...(Dims);
    static_assert(rank > 0, "a Shape needs at least one dimension");

    static constexpr uint32_t dims[rank] = {Dims...};

    static constexpr uint32_t dim(int32_t i) {
        return dims[i < 0 ? static_cast<uint32_t>(static_cast<int32_t>(rank) + i) : static_cast<uint32_t>(i)];
    }

    static constexpr uint32_t num_entries = (uint32_t{1} * ... * Dims);

    static constexpr uint32_t cols = dims[rank - 1];
    static constexpr uint32_t rows = rank >= 2 ? dims[rank >= 2 ? rank - 2 : 0] : 1;
    static constexpr uint32_t leading = num_entries / (rows * cols);

    using tile = Tile<32, 32>;
};

template <uint32_t Rows, uint32_t Cols>
struct Tile {
    static_assert(
        Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8 || Rows == 16 || Rows == 32,
        "a tile's HEIGHT must be 1, 2, 4, 8, 16 or 32 -- see TILE_FACE_HW_CHOICES in "
        "tt_metal/impl/data_format/tile.cpp, which is what the host validates against");
    static_assert(
        Cols == 16 || Cols == 32,
        "a tile's WIDTH must be 16 or 32: a face is 16 wide and a tile is one or two faces "
        "across, so there is no narrower tile to ask for -- see TILE_FACE_HW_CHOICES in "
        "tt_metal/impl/data_format/tile.cpp");

    using extent = Shape<Rows, Cols>;
    static constexpr uint32_t rows = Rows;
    static constexpr uint32_t cols = Cols;
    static constexpr uint32_t elements = Rows * Cols;
};

using TileFull = Tile<32, 32>;

template <typename TileShape, typename S>
struct Tiled : S {
    using tile = TileShape;
};

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

template <typename A, typename B>
inline constexpr bool same_shape_v = std::is_same<A, B>::value;

template <typename S>
inline constexpr uint32_t logical_rows_v = S::rows * S::tile::rows;

template <typename S>
inline constexpr uint32_t logical_cols_v = S::cols * S::tile::cols;

template <typename S>
static constexpr uint32_t logical_dim(int32_t i) {
    const uint32_t idx = i < 0 ? static_cast<uint32_t>(static_cast<int32_t>(S::rank) + i) : static_cast<uint32_t>(i);
    return idx == S::rank - 1 ? logical_cols_v<S>
                              : (idx == S::rank - 2 ? logical_rows_v<S> : S::dim(static_cast<int32_t>(idx)));
}

template <typename A, typename B>
inline constexpr bool same_logical_hw_v =
    logical_rows_v<A> == logical_rows_v<B> && logical_cols_v<A> == logical_cols_v<B>;

template <typename S, uint32_t H, uint32_t W, typename Idx>
struct with_hw_impl;

template <uint32_t... D, uint32_t H, uint32_t W, std::size_t... I>
struct with_hw_impl<Shape<D...>, H, W, std::index_sequence<I...>> {
    using S = Shape<D...>;
    using type = Shape<(I == S::rank - 2 ? H : (I == S::rank - 1 ? W : S::dim(static_cast<int32_t>(I))))...>;
};

template <typename S, uint32_t H, uint32_t W>
using with_hw =
    retile<typename S::tile, typename with_hw_impl<base_shape_t<S>, H, W, std::make_index_sequence<S::rank>>::type>;

}  // namespace unified
}  // namespace tt
