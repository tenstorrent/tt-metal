// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer_types_generated.h"

//////////////////////////////////////////////////////////////
// From-flatbuffer helper functions                         //
//////////////////////////////////////////////////////////////

namespace tt::tt_metal {
inline namespace v0 {

inline CircularBufferConfig FromFlatbuffer(
    const tt::tt_metal::flatbuffer::CircularBufferConfig* config_fb, const Buffer* shadow_global_buffer) {
    if (!config_fb) {
        throw std::runtime_error("Invalid CircularBufferConfig FlatBuffer object");
    }

    // Create a CircularBufferConfig. Constructor doesn't matter much, since we serialized all
    // members, will deserialize them here to get fully formed object.
    CircularBufferConfig config(0, {});
    config.total_size_ = config_fb->total_size();

    // Note: std::optional is not supported by FlatBuffers, so nullopt was serialized as value 0 in FlatBuffer.
    config.globally_allocated_address_ = config_fb->globally_allocated_address() == 0
                                             ? std::nullopt
                                             : std::optional<uint32_t>(config_fb->globally_allocated_address());

    if (config_fb->data_formats()) {
        for (auto entry : *config_fb->data_formats()) {
            config.data_formats_[entry->index()] = FromFlatbuffer(entry->format());
        }
    }

    if (config_fb->page_sizes()) {
        for (auto entry : *config_fb->page_sizes()) {
            config.page_sizes_[entry->index()] = entry->size();
        }
    }

    config.tiles_ = FromFlatbuffer(config_fb->tiles());
    config.shadow_global_buffer = shadow_global_buffer;

    if (config_fb->buffer_indices()) {
        config.buffer_indices_.insert(config_fb->buffer_indices()->begin(), config_fb->buffer_indices()->end());
    }

    if (config_fb->local_buffer_indices()) {
        config.local_buffer_indices_.insert(
            config_fb->local_buffer_indices()->begin(), config_fb->local_buffer_indices()->end());
    }

    if (config_fb->remote_buffer_indices()) {
        config.remote_buffer_indices_.insert(
            config_fb->remote_buffer_indices()->begin(), config_fb->remote_buffer_indices()->end());
    }

    config.dynamic_cb_ = config_fb->dynamic_cb();
    config.max_size_ = config_fb->max_size();
    config.buffer_size_ = config_fb->buffer_size();

    return config;
}

}  // namespace v0
}  // namespace tt::tt_metal
