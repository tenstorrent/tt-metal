// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/buffer_types_from_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

BufferType from_flatbuffer(flatbuffer::BufferType type) {
    switch (type) {
        case flatbuffer::BufferType::DRAM: return BufferType::DRAM;
        case flatbuffer::BufferType::L1: return BufferType::L1;
        case flatbuffer::BufferType::SystemMemory: return BufferType::SYSTEM_MEMORY;
        case flatbuffer::BufferType::L1Small: return BufferType::L1_SMALL;
        case flatbuffer::BufferType::Trace: return BufferType::TRACE;
    }
    TT_THROW("Unsupported BufferType from flatbuffer.");
}

TensorMemoryLayout from_flatbuffer(flatbuffer::TensorMemoryLayout layout) {
    switch (layout) {
        case flatbuffer::TensorMemoryLayout::Interleaved: return TensorMemoryLayout::INTERLEAVED;
        case flatbuffer::TensorMemoryLayout::SingleBank: return TensorMemoryLayout::SINGLE_BANK;
        case flatbuffer::TensorMemoryLayout::HeightSharded: return TensorMemoryLayout::HEIGHT_SHARDED;
        case flatbuffer::TensorMemoryLayout::WidthSharded: return TensorMemoryLayout::WIDTH_SHARDED;
        case flatbuffer::TensorMemoryLayout::BlockSharded: return TensorMemoryLayout::BLOCK_SHARDED;
    }
    TT_THROW("Unsupported TensorMemoryLayout from flatbuffer.");
}

CircularBufferConfig from_flatbuffer(
    const flatbuffer::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer) {
    TT_FATAL(config_fb, "Invalid CircularBufferConfig FlatBuffer object");

    const auto globally_allocated_address =
        config_fb->globally_allocated_address()
            ? std::optional<uint32_t>(config_fb->globally_allocated_address()->value())
            : std::nullopt;

    std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> data_formats = {};
    if (config_fb->data_formats()) {
        for (auto entry : *config_fb->data_formats()) {
            data_formats[entry->index()] = from_flatbuffer(entry->format());
        }
    }

    std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> page_sizes = {};
    if (config_fb->page_sizes()) {
        for (auto entry : *config_fb->page_sizes()) {
            page_sizes[entry->index()] = entry->size();
        }
    }

    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles = {};
    if (config_fb->tiles()) {
        for (auto entry : *config_fb->tiles()) {
            tiles[entry->index()] = from_flatbuffer(entry->tile());
        }
    }

    // Convert FlatBuffer vector to unordered_set of uint8_t
    auto create_uint8_set = [](auto* fb_vector) {
        std::unordered_set<uint8_t> result;
        if (fb_vector) {
            result.insert(fb_vector->begin(), fb_vector->end());
        }
        return result;
    };

    // Constructor supports being able to specify all private members. shadow_global_buffer is public.
    CircularBufferConfig config(
        config_fb->total_size(),
        globally_allocated_address,
        data_formats,
        page_sizes,
        tiles,
        create_uint8_set(config_fb->buffer_indices()),
        create_uint8_set(config_fb->local_buffer_indices()),
        create_uint8_set(config_fb->remote_buffer_indices()),
        config_fb->dynamic_cb(),
        config_fb->max_size(),
        config_fb->buffer_size());

    config.shadow_global_buffer = shadow_global_buffer;

    return config;
}

// TODO: Opportunity to share with TTNN. This was straight up copied from tensor_spec_flatbuffer.cpp

ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation) {
    switch (orientation) {
        case flatbuffer::ShardOrientation::RowMajor: return ShardOrientation::ROW_MAJOR;
        case flatbuffer::ShardOrientation::ColMajor: return ShardOrientation::COL_MAJOR;
    }
    TT_THROW("Unsupported ShardOrientation from flatbuffer.");
}

ShardMode from_flatbuffer(flatbuffer::ShardMode mode) {
    switch (mode) {
        case flatbuffer::ShardMode::Physical: return ShardMode::PHYSICAL;
        case flatbuffer::ShardMode::Logical: return ShardMode::LOGICAL;
    }
    TT_THROW("Unsupported ShardMode from flatbuffer.");
}

ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec) {
    CoreRangeSet grid = from_flatbuffer(spec->grid());
    std::array<uint32_t, 2> shape = {spec->shape_h(), spec->shape_w()};
    ShardOrientation orientation = from_flatbuffer(spec->orientation());
    ShardMode mode = from_flatbuffer(spec->shard_mode());
    if (const auto* fb_shard_shape = spec->physical_shard_shape()) {
        std::array<uint32_t, 2> physical_shard_shape = {fb_shard_shape->height(), fb_shard_shape->width()};
        return ShardSpec(grid, shape, physical_shard_shape, orientation);
    }
    return ShardSpec(grid, shape, orientation, mode);
}

std::optional<ShardSpecBuffer> from_flatbuffer(const flatbuffer::ShardSpecBuffer* fb_shard_spec) {
    if (!fb_shard_spec) {
        return std::nullopt;
    }

    return ShardSpecBuffer{
        from_flatbuffer(fb_shard_spec->tensor_shard_spec()),
        {fb_shard_spec->page_shape_h(), fb_shard_spec->page_shape_w()},
        {fb_shard_spec->tensor2d_shape_in_pages_h(), fb_shard_spec->tensor2d_shape_in_pages_w()}};
}

}  // namespace tt::tt_metal
