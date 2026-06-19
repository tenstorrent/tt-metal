// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/shape.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

#include <memory>
#include <optional>
#include <string>

namespace tt::tt_metal {

// Validates that the physical shard shape is aligned to tile dimensions for sharded TILE layouts.
// Returns true if the configuration is valid, false otherwise.
bool can_shard_align(const MemoryConfig& memory_config, const Layout& layout, const Tile& tile = Tile{});

class IDevice;

// Implementation details for TensorLayout. Reachable from within tt_metal via impl(); not part of the public API.
class TensorLayoutImpl;

using Strides = std::vector<size_t>;

// TensorLayout describes how a tensor is laid out in memory
// It takes datatype, layout (eg. TILE vs. RM), memory (eg. DRAM vs. L1), sharding (ie. how you want to cut your logical
// shape) And provides information required to physically lay out the tensor in memory
class TensorLayout {
public:
    TensorLayout(
        DataType dtype,
        const PageConfig& page_config,
        const MemoryConfig& memory_config,
        const Alignment& alignment = {});

    ~TensorLayout();
    TensorLayout(const TensorLayout& other);
    TensorLayout& operator=(const TensorLayout& other);
    TensorLayout(TensorLayout&& other) noexcept;
    TensorLayout& operator=(TensorLayout&& other) noexcept;

    // static method makes it easy to find and remove all of its usages in the codebase - that's why it is not a
    // constructor
    [[deprecated("Use of Padded Shape is deprecated")]] static TensorLayout fromPaddedShape(
        DataType dtype,
        const PageConfig& page_config,
        const MemoryConfig& memory_config,
        const tt::tt_metal::Shape& logical_shape,
        const tt::tt_metal::Shape& padded_shape);

    Layout get_layout() const;
    Tile get_tile() const;
    PageConfig get_page_config() const;
    DataType get_data_type() const;
    const MemoryConfig& get_memory_config() const;
    const Alignment& get_alignment() const;

    // This method is deprecated and should be replaced with get_strides() / get_physical_size()
    // It computes padded shape on the fly from shape and alignment
    [[deprecated(
        "Use of LegacyPaddedShape is deprecated. Please use get_physical_size() or get_strides() instead.")]] tt::
        tt_metal::Shape
        compute_padded_shape(const tt::tt_metal::Shape& shape) const;

    TensorLayout with_memory_config(MemoryConfig memory_config) const;

    bool operator==(const TensorLayout& other) const;
    bool operator!=(const TensorLayout& other) const;


    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "page_config", "memory_config", "alignment");
    std::tuple<const DataType&, const PageConfig&, const MemoryConfig&, const Alignment&> attribute_values() const;

    static TensorLayout restore_from_serialized(
        DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

    // Access to the implementation, which carries the internal layout-computation API.
    //
    // pre-condition: the TensorLayout must not be in a moved-from state.
    TensorLayoutImpl& impl();
    const TensorLayoutImpl& impl() const;

private:
    // impl_ may be nullptr if the TensorLayout is in a moved-from state.
    // Avoid using impl_ directly; use the impl() accessor instead.
    std::unique_ptr<TensorLayoutImpl> impl_;
};

}  // namespace tt::tt_metal
