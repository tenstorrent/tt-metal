// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_set>

#include "logger.hpp"
#include "tt_backend_api_types.hpp"
#include "buffer.hpp"
#include "tile.hpp"

#include "circular_buffer_constants.h"

namespace tt::tt_metal {
inline namespace v0 {

using CBHandle = uintptr_t;

class CircularBufferConfig {
public:
    // Static circular buffer spec
    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec);

    // User is expected to use the builder here.
    CircularBufferConfig(uint32_t total_size);

    // Dynamic circular buffer spec
    CircularBufferConfig(
        uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer);

    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);

    CircularBufferConfig& set_total_size(uint32_t total_size);

    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);

    CircularBufferConfig& set_globally_allocated_address_and_total_size(const Buffer& buffer, uint32_t total_size);

    CircularBufferConfig& set_tile_dims(uint8_t buffer_index, const Tile& tile);

    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles() const;

    uint32_t total_size() const;

    std::optional<uint32_t> globally_allocated_address() const;

    const std::unordered_set<uint8_t>& buffer_indices() const;
    const std::unordered_set<uint8_t>& local_buffer_indices() const;
    const std::unordered_set<uint8_t>& remote_buffer_indices() const;

    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats() const;

    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes() const;

    const Buffer* shadow_global_buffer{nullptr};

    class Builder {
    public:
        static Builder LocalBuilder(CircularBufferConfig& parent, uint8_t buffer_index);
        static Builder RemoteBuilder(CircularBufferConfig& parent, uint8_t buffer_index);

        const Builder& set_data_format(tt::DataFormat data_format) const;

        const Builder& set_total_size(uint32_t total_size) const;

        const Builder& set_page_size(uint32_t page_size) const;

        const Builder& set_tile_dims(const Tile& tile) const;

    private:
        Builder(CircularBufferConfig& parent, uint8_t buffer_index);

        CircularBufferConfig& parent_;
        uint8_t buffer_index_;
    };

    Builder index(uint8_t buffer_index);
    Builder remote_index(uint8_t buffer_index);

    friend bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);
    friend bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);

private:
    void set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec);
    void validate_total_size(uint32_t total_size);

    uint32_t total_size_ = 0;
    std::optional<uint32_t> globally_allocated_address_ = std::nullopt;
    std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> data_formats_;
    std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> page_sizes_;
    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles_;
    std::unordered_set<uint8_t> buffer_indices_;
    std::unordered_set<uint8_t> local_buffer_indices_;
    std::unordered_set<uint8_t> remote_buffer_indices_;
    bool dynamic_cb_ = false;
    // `max_size_` is used to ensure that total size does not grow beyond associated buffer size
    // `buffer_size_` is tracked to enforce the old size assertions.
    // Will be removed once tests are updated to respect the correct `max_size_` constraint
    uint32_t max_size_ = 0;
    uint32_t buffer_size_ = 0;
};

bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);
bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);

}  // namespace v0
}  // namespace tt::tt_metal
