// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/fmt.hpp>

#include <memory>
#include <numeric>
#include <optional>
#include <variant>

#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tile.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

// ------------------------------------------------------------------------------------------------
// RowMajorPageConfig and TilePageConfig: internal implementation types.
// These are NOT part of the public PageConfig API. Access them via PageConfig::impl().
// ------------------------------------------------------------------------------------------------

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

    // Emits a deprecation warning; prefer raw_tile() for internal/reflection use.
    const Tile& get_tile() const;
    const Tile& raw_tile() const { return tile_; }

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const RowMajorPageConfig&) const = default;
    bool operator!=(const RowMajorPageConfig&) const = default;

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
    const Tile& raw_tile() const { return tile_; }

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const TilePageConfig&) const = default;
    bool operator!=(const TilePageConfig&) const = default;

private:
    Tile tile_;
};

using PageConfigVariant = std::variant<RowMajorPageConfig, TilePageConfig>;

// ------------------------------------------------------------------------------------------------
// PageConfigImpl: the internal page-config API, reachable from within tt_metal via impl().
// ------------------------------------------------------------------------------------------------

class PageConfigImpl {
public:
    explicit PageConfigImpl(const PageConfigVariant& config) : config_(config) {}

    PageConfigImpl(Layout layout, const std::optional<Tile>& tile) {
        if (layout == Layout::ROW_MAJOR) {
            // TODO: add TT_FATAL(!tile.has_value(), "Specifying tile shape for a row major layout is not supported")
            config_ = RowMajorPageConfig(tile.value_or(Tile()));
        } else {
            config_ = TilePageConfig(tile.value_or(Tile()));
        }
    }

    bool operator==(const PageConfigImpl&) const = default;
    bool operator!=(const PageConfigImpl&) const = default;

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.create_default_alignment(dtype, memory_config); },
            config_);
    }

    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
        std::visit(
            [&](const auto& config) constexpr { config.validate_alignment(alignment, dtype, memory_config); }, config_);
    }

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const {
        return std::visit(
            [&](const auto& config) constexpr {
                return config.get_page_shape(physical_size, dtype, memory_config, physical_shard_size);
            },
            config_);
    }

    size_t get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dtype); }, config_);
    }

    Layout get_layout() const {
        if (std::holds_alternative<RowMajorPageConfig>(config_)) {
            return Layout::ROW_MAJOR;
        }
        return Layout::TILE;
    }

    Tile get_tile() const {
        return std::visit([&](const auto& config) { return config.get_tile(); }, config_);
    }

    // Returns the stored tile without triggering the RowMajor deprecation warning.
    // Use this for reflection and hashing.
    Tile raw_tile() const {
        return std::visit([](const auto& config) { return config.raw_tile(); }, config_);
    }

    Alignment get_required_shard_shape_alignment() const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_required_shard_shape_alignment(); }, config_);
    }

    Alignment get_recommended_shard_shape_alignment(DataType dtype) const {
        return std::visit(
            [&](const auto& config) constexpr { return config.get_recommended_shard_shape_alignment(dtype); }, config_);
    }

    const PageConfigVariant& config() const { return config_; }

private:
    PageConfigVariant config_;
};

}  // namespace tt::tt_metal
