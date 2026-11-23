// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A tensor size known only at run time, as llk-lib takes it: the counts of a call's Work.
//
// The split. A resolved call is compile time except for the three counts of its Work (resolver/plan.h): tile_rows,
// tile_cols, k_tiles. Each is either Fixed (the number is the value) or comes from a size one of the call's tensors
// holds at run time (CountSource); the Work then holds that size's upper bound, and accepts / options / work / plan
// run on the bound exactly as they run on a compile-time size. Everything configure reads is therefore compile time:
// formats, registers, fidelity, tile geometry, address steps, the replay and MOP bodies. What follows the run-time
// value is only how often execute's loops run, the page counts of wait_front / reserve_back, and the one MOP loop
// count llk-lib's pack has.
//
// Two representations meet here, on purpose. Kernel, Tensor, Resolver and Compute API say "run time" with a VALUE: a
// mark in the spec, a CountSource in the Work, the count in the Tensor object, because specs, Works and Plans are
// template arguments and a template argument cannot hold a run-time number (tensor/tensor.h, "run-time sizes").
// llk-lib says it with a TYPE, ct or rt (wrapper/wrapper.h), because there a count is a function argument, and a
// loop's bound has to be a constant inside the function that loops for a compile-time call's code to stay what it was.
//
//   WorkCounts<W>       the three counts of Work W as execute takes them: a count that is Fixed is a ct (an
//                       empty object, its value in the type), a run-time one an rt<std::uint32_t>. For a Work
//                       with only fixed counts WorkCounts is empty, and a call that passes it costs nothing.
//                       The Compute API makes it from the call's tensors (compute_api/counts.h)
//   plan_count<...>(w)  llk-lib's entry: the count a loop of the Plan runs to (WorkCount), as a ct or an rt,
//                       which it hands to execute. execute's loops are written once for both
//
// An execute takes a count as `template <class N, ctrt<N, std::uint32_t> = 0> ... const N& n` and reads it with
// wrapper::value(n), and checks an rt against its bound (1..bound) with LLK_ASSERT.
//
// Which count a loop runs to. The unpackers' walk is the Plan's own numbers, which may or may not be one of the
// Work's counts: the Plan names it (UnpackWalk::steps_of / runs_of / tiles_of), and unpack and matmul (whose loops
// are the walk's) read the name. An entry whose loop IS a Work count by definition names it itself: eltwise_binary,
// vec and pack run over every output tile (WorkCount::Tiles), reduce folds k_tiles tiles into each (WorkCount::K).

#pragma once

#include <cstdint>
#include <type_traits>

#include "resolver/plan.h"
#include "wrapper/wrapper.h"

namespace llk2
{

/// A count as llk-lib's execute takes it: the constant Number, or a run-time value of at most Number.
template <std::uint32_t Number, bool Runtime>
using CtOrRt = std::conditional_t<Runtime, rt<std::uint32_t>, ct<Number>>;

/// The run-time side of Work W: its three counts, each a constant or a run-time value. Empty when all are Fixed.
template <Work W>
struct WorkCounts
{
    CtOrRt<W.tile_rows, W.rows_from != CountSource::Fixed> tile_rows; ///< output tiles along y
    CtOrRt<W.tile_cols, W.cols_from != CountSource::Fixed> tile_cols; ///< output tiles along x
    CtOrRt<W.k_tiles, W.k_from != CountSource::Fixed> k_tiles;        ///< K tiles
};

/// The count a loop of a Plan runs to: Work count Of of w, or the Plan's own Number when the loop is no Work count.
template <WorkCount Of, std::uint32_t Number, Work W>
constexpr auto plan_count(const WorkCounts<W>& w)
{
    if constexpr (Of == WorkCount::Rows)
    {
        return w.tile_rows;
    }
    else if constexpr (Of == WorkCount::Cols)
    {
        return w.tile_cols;
    }
    else if constexpr (Of == WorkCount::K)
    {
        return w.k_tiles;
    }
    else if constexpr (Of == WorkCount::Tiles)
    {
        if constexpr (W.rows_from == CountSource::Fixed && W.cols_from == CountSource::Fixed)
        {
            return ct<W.tile_rows * W.tile_cols> {};
        }
        else
        {
            return rt<std::uint32_t> {wrapper::value(w.tile_rows) * wrapper::value(w.tile_cols)};
        }
    }
    else
    {
        return ct<Number> {};
    }
}

} // namespace llk2
