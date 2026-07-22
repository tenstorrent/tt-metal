// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <utility>
#include <variant>

#include <tt-metalium/tile.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/layout.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

// Empty page-config alternative for row-major tensors.
// User-declared ctor keeps this non-aggregate so reflection uses attribute_names/values only
// (avoids colliding with ttsl::concepts::Reflectable).
class RowMajorPageConfig {
public:
    RowMajorPageConfig() = default;

    bool operator==(const RowMajorPageConfig&) const = default;
    bool operator!=(const RowMajorPageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const { return std::forward_as_tuple(); }
};

class TilePageConfig {
public:
    TilePageConfig() = default;
    explicit TilePageConfig(Tile tile) : tile(std::move(tile)) {}

    Tile tile = Tile{};

    bool operator==(const TilePageConfig&) const = default;
    bool operator!=(const TilePageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    auto attribute_values() const { return std::forward_as_tuple(tile); }
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(Config config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    /// Returns the recommended alignment for the shard shape.
    /// This takes into account device memory alignment requirements trying to optimize memory usage and read/write
    /// performance. The exact device alignment requirements are dependent on device architecture and BufferType, so the
    /// maximum possible alignment is used.
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    Layout get_layout() const;
    Tile get_tile() const;  // Tile{} for ROW_MAJOR
    const Config& get_config() const { return config_; }

    bool operator==(const PageConfig&) const = default;
    bool operator!=(const PageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("config");
    auto attribute_values() const { return std::forward_as_tuple(config_); }

private:
    Config config_;
};

}  // namespace tt::tt_metal
