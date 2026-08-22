// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <optional>
#include <vector>

struct TensAddr {
    std::vector<std::uint32_t> sh;

    TensAddr(const std::vector<std::uint32_t>& shape);
    std::uint32_t numel() const;
    int offs(int n, int c, int h, int w);
};

std::uint32_t round_up_to_mul16(std::uint32_t val);

std::uint32_t round_up_to_tile(int val, int tile_val);

template <class T>
std::vector<T> convert_layout_tile_swizzled_to_tile_nfaces(
    ttsl::Span<const T> in_tile_swizzled,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

template <class T>
std::vector<T> convert_layout_tile_nfaces_to_tile_swizzled(
    ttsl::Span<const T> in_tile_nfaces,
    std::optional<PhysicalSize> tile_shape = std::nullopt,
    std::optional<PhysicalSize> face_shape = std::nullopt,
    bool transpose_face = false,
    bool transpose_face_order = false);

// Those are specific cases that convert_layout can do, but left for compatibility with existing codebase
// Converts from/to row major
template <typename T>
std::vector<T> tilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> untilize_swizzled(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

template <typename T>
std::vector<T> untilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);
