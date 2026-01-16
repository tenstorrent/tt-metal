// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file is a copy of TTNN's ttnn/api/ttnn/tensor/layout/page_config.hpp
// at commit 9f3856801448f589170defe41b23c8b9b43e33a2, with modifications to
// use experimental tensor types.

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tile.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>  // For Layout, DataType

namespace tt::tt_metal {

class RowMajorPageConfig {
public:
    RowMajorPageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const;

    const Tile& get_tile() const;

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const RowMajorPageConfig&) const = default;
    bool operator!=(const RowMajorPageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    auto attribute_values() const { return std::forward_as_tuple(tile_); }

private:
    // This is currently needed for compatibility reasons.
    // Each time tile is specified, a warning will be issued. This should be removed soon.
    Tile tile_;
};

class TilePageConfig {
public:
    TilePageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const;

    const Tile& get_tile() const;

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const TilePageConfig&) const = default;
    bool operator!=(const TilePageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    auto attribute_values() const { return std::forward_as_tuple(tile_); }

private:
    Tile tile_;
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(const Config& config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const;

    Tile get_tile() const;

    Layout get_layout() const;

    Alignment get_required_shard_shape_alignment() const;

    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const PageConfig&) const = default;
    bool operator!=(const PageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("config");
    auto attribute_values() const { return std::forward_as_tuple(config_); }

private:
    Config config_;
};

}  // namespace tt::tt_metal
