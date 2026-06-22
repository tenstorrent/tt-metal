// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <tuple>

#include <tt-metalium/tile.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/layout.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

class PageConfigImpl;

class PageConfig {
public:
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    ~PageConfig();
    PageConfig(const PageConfig& other);
    PageConfig& operator=(const PageConfig& other);
    PageConfig(PageConfig&& other) noexcept;
    PageConfig& operator=(PageConfig&& other) noexcept;

    Tile get_tile() const;
    Layout get_layout() const;

    /// Returns the recommended alignment for the shard shape.
    /// This takes into account device memory alignment requirements trying to optimize memory usage and read/write
    /// performance. The exact device alignment requirements are dependent on device architecture and BufferType, so the
    /// maximum possible alignment is used.
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const PageConfig& other) const;
    bool operator!=(const PageConfig& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("layout", "tile");
    std::tuple<Layout, Tile> attribute_values() const;

    // Access to the implementation, which carries the internal page-config API.
    //
    // pre-condition: the PageConfig must not be in a moved-from state.
    PageConfigImpl& impl();
    const PageConfigImpl& impl() const;

private:
    // impl_ may be nullptr if the PageConfig is in a moved-from state.
    // Avoid using impl_ directly; use the impl() accessor instead.
    std::unique_ptr<PageConfigImpl> impl_;
};

}  // namespace tt::tt_metal
