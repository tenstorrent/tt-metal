// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TEST ONLY: compile-time checks of the Tensor views (tensor/tensor.h: narrow, narrow_up_to, select, row, rows,
// rows_up_to, cols, cols_up_to, tile), one static_assert per rule, its message the rule. Nothing here runs: a test
// source includes this file and every build of it checks the static_asserts (the dataflow harness test, and
// tests/host/tensor_views_test.cpp on the host and with the device compiler). What must NOT compile, and what a
// run-time start or length outside the tensor does, is in that host test and its driver,
// tt-llk/tests/hardware_free/kernels/test_llk2_views.py.

#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace llk2::views_selftest
{

inline constexpr const auto& s_spec = l1_spec<3 * 32, 16 * 32, DataFormat::Float16_b>; // [3, 16] tiles
inline constexpr const auto& m_spec = l1_spec<3 * 32, 32, DataFormat::Float16_b>;      // [3, 1]: a column of tiles

/// Never called: the views are taken here so that they are compile-time values and no object of the test exists in a
/// device build.
inline void views_hold()
{
    constexpr std::uint32_t base = 0x1000; // s's address, and m's
    constexpr std::uint32_t tb   = tile_bytes(s_spec);
    constexpr Tensor<s_spec> s(base);
    constexpr Tensor<m_spec> m(base);

    // ---- narrow and select, the start at run time ----
    constexpr auto r  = narrow<Rows, 2>(s, 1);                  // [2, 16] from tile row 1
    constexpr auto c  = narrow<Cols, 8>(s, 4);                  // [3, 8] from tile column 4
    constexpr auto rc = narrow<Cols, 8>(select<Rows>(s, 2), 8); // [1, 8]: tile row 2, from tile column 8
    static_assert(
        r.address == base + 16 * tb && c.address == base + 4 * tb && rc.address == base + (2 * 16 + 8) * tb, "a start moves the address by whole tiles");
    static_assert(r.spec.shape.sizes[TileY] == 2 && r.spec.shape.sizes[TileX] == 16, "narrow<Rows, N> has N tile rows and all the tile columns");
    static_assert(c.spec.shape.sizes[TileY] == 3 && c.spec.shape.sizes[TileX] == 8, "narrow<Cols, N> has N tile columns and all the tile rows");
    static_assert(is_contiguous(r.spec.shape), "whole tile rows are contiguous");
    static_assert(!is_contiguous(c.spec.shape) && has_dense_tile_rows(c.spec.shape), "some tile columns of several tile rows are a window");
    static_assert(tile_row_step(c.spec.shape) == 16, "a window's tile rows step by its parent's width");
    static_assert(is_contiguous(rc.spec.shape) && tile_count(rc.spec.shape) == 8, "some tile columns of one tile row are contiguous");
    static_assert(select<Rows>(c, 2).address == base + (2 * 16 + 4) * tb, "a tile row of a window is at the parent's tile-row step");
    static_assert(narrow<Rows, 2>(c, 1).address == base + (16 + 4) * tb, "tile rows of a window are at the parent's tile-row step");

    // ---- the start at compile time ----
    static_assert(narrow<Cols, 8, 4>(s).address == c.address && &narrow<Cols, 8, 4>(s).spec == &c.spec, "a compile-time start gives the run-time start's view");
    static_assert(narrow<Rows, 1, 2>(s).address == select<Rows>(s, 2).address, "narrow<Rows, 1, i> is at select<Rows>'s address");
    static_assert(&narrow<Rows, 1, 2>(s).spec == &select<Rows>(s, 2).spec, "narrow<Rows, 1, i> has select<Rows>'s spec");

    // ---- the short names ----
    static_assert(&rows<2>(s, 1).spec == &r.spec && rows<2>(s, 1).address == r.address, "rows<N> is narrow<Rows, N>");
    static_assert(&cols<8>(s, 4).spec == &c.spec && cols<8>(s, 4).address == c.address, "cols<N> is narrow<Cols, N>");
    static_assert(&row(s, 2).spec == &select<Rows>(s, 2).spec && row(s, 2).address == base + 2 * 16 * tb, "row is select<Rows>");
    static_assert(
        &tile(m, 1).spec == &select<Rows>(m, 1).spec && tile(m, 2).address == base + 2 * tile_bytes(m_spec), "tile is select<Rows> of a column of tiles");

    // ---- one spec object per block ----
    static_assert(same(narrow<Cols, 16, 0>(s).spec, s_spec) && same(narrow<Rows, 3, 0>(s).spec, s_spec), "all of a tensor has the tensor's own spec value");
    static_assert(
        &narrow<Cols, 8>(narrow<Rows, 1>(s, 0), 0).spec == &narrow<Rows, 1>(narrow<Cols, 8>(s, 0), 0).spec,
        "equal blocks cut two ways are one spec object, so one Tensor type");

    // ---- a run-time length ----
    constexpr auto u = cols_up_to<8>(s, 4, 5); // 5 of at most 8 tile columns of [3, 16], from tile column 4
    constexpr auto v = rows_up_to<2>(s, 1, 1); // 1 of at most 2 tile rows, from tile row 1
    static_assert(u.count == 5 && v.count == 1, "a view with a run-time length holds the length");
    static_assert(u.address == c.address && v.address == r.address, "a run-time length does not move the address");
    static_assert(is_runtime_size(u.spec, TileX) && !is_runtime_size(u.spec, TileY), "cols_up_to marks the tile columns run time, and only them");
    static_assert(&u.spec == &runtime_cols<c.spec> && &v.spec == &runtime_rows<r.spec>, "the spec is the bound's block with the dim marked run time");
    static_assert(
        equal(u.spec.shape.sizes, c.spec.shape.sizes) && equal(u.spec.shape.strides, c.spec.shape.strides), "the sizes and strides are the bound's block's");
    static_assert(&cols_up_to<8>(s, 0, 8).spec == &u.spec, "the spec does not depend on the start or the length");
    static_assert(
        &narrow_up_to<Cols, 8>(s, 4, 5).spec == &u.spec && &narrow_up_to<Rows, 2>(s, 1, 1).spec == &v.spec, "cols_up_to and rows_up_to are narrow_up_to");
}

} // namespace llk2::views_selftest
