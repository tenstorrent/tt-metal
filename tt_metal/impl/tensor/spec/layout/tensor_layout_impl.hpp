// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

/**
 * Internal implementation header for TensorLayout.
 *
 * This header exposes TensorLayoutImpl so that translation units within tt_metal (e.g. tensor_spec.cpp)
 * can operate directly on the implementation without going through the public TensorLayout accessors.
 * It is private to tt_metal and is not part of the installed public API.
 */

namespace tt::tt_metal {

class TensorLayoutImpl {
public:
    TensorLayoutImpl(
        DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

    TensorLayoutImpl(const TensorLayoutImpl&) = default;
    TensorLayoutImpl(TensorLayoutImpl&&) noexcept = default;
    TensorLayoutImpl& operator=(const TensorLayoutImpl&) = default;
    TensorLayoutImpl& operator=(TensorLayoutImpl&&) noexcept = default;
    ~TensorLayoutImpl() = default;

    bool operator==(const TensorLayoutImpl&) const = default;
    bool operator!=(const TensorLayoutImpl&) const = default;

    Layout get_layout() const { return page_config_.get_layout(); }
    Tile get_tile() const { return page_config_.get_tile(); }
    const PageConfig& get_page_config() const { return page_config_; }
    DataType get_data_type() const { return dtype_; }
    const MemoryConfig& get_memory_config() const { return memory_config_; }
    const Alignment& get_alignment() const { return alignment_; }

    void set_memory_config(MemoryConfig memory_config) { memory_config_ = std::move(memory_config); }

    Strides compute_strides(const tt::tt_metal::Shape& shape) const;

    BufferShardingArgs compute_buffer_sharding_args(const tt::tt_metal::Shape& shape) const;

    size_t compute_packed_buffer_size_bytes(const tt::tt_metal::Shape& shape) const;
    size_t compute_page_size_bytes(const tt::tt_metal::Shape& shape) const;

    size_t compute_consumed_memory_bytes_per_bank(
        const tt::tt_metal::Shape& shape, size_t page_alignment, size_t num_banks) const;

    tt::tt_metal::Shape compute_padded_shape(const tt::tt_metal::Shape& shape) const;

    Shape2D compute_logical_2d_shape(const tt::tt_metal::Shape& shape) const;

    Shape2D compute_physical_shape(const tt::tt_metal::Shape& shape) const;

    Shape2D get_logical_shard_shape() const;

    Shape2D get_physical_shard_shape() const;

    Shape2D compute_page_shape(const Shape2D& physical_size) const;
    size_t compute_page_size_bytes(const Shape2D& page_size) const;

    std::tuple<const DataType&, const PageConfig&, const MemoryConfig&, const Alignment&> attribute_values() const {
        return std::forward_as_tuple(dtype_, page_config_, memory_config_, alignment_);
    }

private:
    void initialize_alignment();

    DataType dtype_ = DataType::BFLOAT16;
    PageConfig page_config_;
    MemoryConfig memory_config_;
    Alignment alignment_;
};

}  // namespace tt::tt_metal
