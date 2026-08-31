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
};

// Shape equality, spelled for legibility inside static_asserts.
template <typename A, typename B>
inline constexpr bool same_shape_v = std::is_same<A, B>::value;

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

template <typename S, uint32_t H, uint32_t W>
using with_hw = typename with_hw_impl<S, H, W, std::make_index_sequence<S::rank>>::type;

}  // namespace unified
}  // namespace tt
