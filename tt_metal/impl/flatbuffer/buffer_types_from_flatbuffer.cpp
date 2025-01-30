// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/buffer_types_from_flatbuffer.hpp"

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

CircularBufferConfig from_flatbuffer(
    const flatbuffer::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer) {
    TT_FATAL(config_fb, "Invalid CircularBufferConfig FlatBuffer object");

    std::optional<uint32_t> globally_allocated_address =
        (config_fb->globally_allocated_address() == 0)
            ? std::nullopt
            : std::optional<uint32_t>(config_fb->globally_allocated_address());

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

}  // namespace tt::tt_metal
