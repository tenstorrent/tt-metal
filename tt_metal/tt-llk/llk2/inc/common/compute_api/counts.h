// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API: the run-time counts of a call, between its Tensors and llk-lib (resolver/work_counts.h has the split).
// Every op's Compute API function uses these the same way, as the lines of the form of an op that carry `counts`
// (compute_api/modifiers.h):
//
//   same_runtime_count(in0, in1)       whether two tensors that hold the same run-time size agree on it
//   work_counts<r.work>(in0, in1)      the WorkCounts: each run-time count, read from the tensor the Work names
//   result_of<T, E, r.work>(acc, ...)  the Dest tensor written, holding the count written
//
// An op whose result has the size of its input (the SFPU functions, in place) or of the tensor it writes (pack, into
// dst) returns that tensor as it is, as_owned: the count it holds is already the one the op ran on. reduce's result
// is one tile per tile row, a compile-time size.

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "resolver/work_counts.h"

namespace llk2
{

namespace detail
{
/// One count of a Work: the constant, or the run-time size of the tensor From names.
template <std::uint32_t Number, CountSource From, const auto& A, class RA, const auto& B, class RB>
constexpr CtOrRt<Number, From != CountSource::Fixed> count_from(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1)
{
    if constexpr (From == CountSource::Fixed)
    {
        return {};
    }
    else if constexpr (From == CountSource::In0TileY || From == CountSource::In0TileX)
    {
        static_assert(is_runtime_size(A, From == CountSource::In0TileY ? TileY : TileX), "the Work names a size in0's spec does not mark run time");
        return rt<std::uint32_t> {in0.count};
    }
    else
    {
        static_assert(is_runtime_size(B, From == CountSource::In1TileY ? TileY : TileX), "the Work names a size in1's spec does not mark run time");
        return rt<std::uint32_t> {in1.count};
    }
}
} // namespace detail

/// The run-time side of Work W for a call on in0 and in1: each run-time count from the tensor the Work names.
template <Work W, const auto& A, class RA, const auto& B, class RB>
constexpr WorkCounts<W> work_counts(const Tensor<A, RA>& in0, const Tensor<B, RB>& in1)
{
    return {
        detail::count_from<W.tile_rows, W.rows_from>(in0, in1),
        detail::count_from<W.tile_cols, W.cols_from>(in0, in1),
        detail::count_from<W.k_tiles, W.k_from>(in0, in1)};
}

/// Whether the two tensors of a call agree on a run-time size they both hold (always true when either has none).
template <const auto& A, class RA, const auto& B, class RB>
constexpr bool same_runtime_count(const Tensor<A, RA>& a, const Tensor<B, RB>& b)
{
    if constexpr (has_runtime_size(A) && has_runtime_size(B))
    {
        return a.count == b.count;
    }
    else
    {
        return true;
    }
}

/// acc as the result of an op with Work W, under owner (T, E): when the output tile rows or columns are a run-time
/// count, a tensor that holds it (tensor.h, with_count), so the next op (an SFPU activation, pack) takes the count
/// from what was written and not from the size of the Dest section. An acc that already holds a count (an earlier
/// result, written again in place or added to: mul with .accumulate) keeps it, and this op must have written as many.
template <Thread T, Exu E, Work W, const auto& D, class RD>
inline __attribute__((always_inline)) auto result_of(const Tensor<D, RD>& acc, [[maybe_unused]] const WorkCounts<W>& counts)
{
    constexpr bool rows_at_run_time = W.rows_from != CountSource::Fixed;
    constexpr bool cols_at_run_time = W.cols_from != CountSource::Fixed;
    if constexpr (!rows_at_run_time && !cols_at_run_time)
    {
        return as_owned<T, E>(acc);
    }
    else
    {
        constexpr Dim dim           = cols_at_run_time ? TileX : TileY;
        const std::uint32_t written = cols_at_run_time ? wrapper::value(counts.tile_cols) : wrapper::value(counts.tile_rows);
        if constexpr (has_runtime_size(D))
        {
            static_assert(is_runtime_size(D, dim), "acc holds the count of another size than the op's run-time one");
            LLK_ASSERT(acc.count == written, "acc holds another run-time count than the op wrote");
            return as_owned<T, E>(acc);
        }
        else
        {
            static_assert(!(rows_at_run_time && cols_at_run_time), "one run-time size per tensor");
            return as_owned<T, E>(with_count<dim>(acc, written));
        }
    }
}

} // namespace llk2
