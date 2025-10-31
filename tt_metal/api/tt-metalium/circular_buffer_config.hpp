// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
class Buffer;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using CBHandle = uintptr_t;

class CircularBufferConfigImpl;
class CircularBufferConfig {
public:
    // Static circular buffer spec
    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec);

    // User is expected to use the builder here.
    CircularBufferConfig(uint32_t total_size);

    // Dynamic circular buffer spec
    CircularBufferConfig(
        uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer);

    // For flatbuffer deserialization, set all private members.
    CircularBufferConfig(
        uint32_t total_size,
        std::optional<uint32_t> globally_allocated_address,
        const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats,
        const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes,
        const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles,
        const std::unordered_set<uint8_t>& buffer_indices,
        const std::unordered_set<uint8_t>& local_buffer_indices,
        const std::unordered_set<uint8_t>& remote_buffer_indices,
        bool dynamic_cb,
        uint32_t max_size,
        uint32_t buffer_size);

    CircularBufferConfig(const CBDescriptor& descriptor);

    // Copy constructor
    CircularBufferConfig(const CircularBufferConfig& other);

    // Copy assignment operator
    CircularBufferConfig& operator=(const CircularBufferConfig& other);

    // Move constructor
    CircularBufferConfig(CircularBufferConfig&& other) noexcept = default;

    // Move assignment operator
    CircularBufferConfig& operator=(CircularBufferConfig&& other) noexcept = default;

    // Destructor
    ~CircularBufferConfig() = default;

    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);

    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);

    CircularBufferConfig& set_tile_dims(uint8_t buffer_index, const Tile& tile);

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

    CircularBufferConfigImpl* impl() { return impl_.get(); }
    const CircularBufferConfigImpl* impl() const { return impl_.get(); }

private:
    std::unique_ptr<CircularBufferConfigImpl> impl_;
};

class CircularBufferConfigImpl {
public:
    // Static circular buffer spec
    CircularBufferConfigImpl(uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec);

    // User is expected to use the builder here.
    CircularBufferConfigImpl(uint32_t total_size);

    // Dynamic circular buffer spec
    CircularBufferConfigImpl(
        uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer);

    // For flatbuffer deserialization, set all private members.
    CircularBufferConfigImpl(
        uint32_t total_size,
        std::optional<uint32_t> globally_allocated_address,
        const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats,
        const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes,
        const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles,
        const std::unordered_set<uint8_t>& buffer_indices,
        const std::unordered_set<uint8_t>& local_buffer_indices,
        const std::unordered_set<uint8_t>& remote_buffer_indices,
        bool dynamic_cb,
        uint32_t max_size,
        uint32_t buffer_size);

    CircularBufferConfigImpl(const CBDescriptor& descriptor);

    // Copy constructor
    CircularBufferConfigImpl(const CircularBufferConfigImpl& other) = default;

    CircularBufferConfigImpl& set_page_size(uint8_t buffer_index, uint32_t page_size);

    CircularBufferConfigImpl& set_total_size(uint32_t total_size);

    CircularBufferConfigImpl& set_globally_allocated_address(const Buffer& buffer);

    CircularBufferConfigImpl& set_globally_allocated_address_and_total_size(const Buffer& buffer, uint32_t total_size);

    CircularBufferConfigImpl& set_tile_dims(uint8_t buffer_index, const Tile& tile);

    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles() const;

    uint32_t total_size() const;

    std::optional<uint32_t> globally_allocated_address() const;

    const std::unordered_set<uint8_t>& buffer_indices() const;
    const std::unordered_set<uint8_t>& local_buffer_indices() const;
    const std::unordered_set<uint8_t>& remote_buffer_indices() const;

    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats() const;

    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes() const;

    // These 3 getters are not typically used, but needed for flatbuffer serialization
    bool dynamic_cb() const;
    uint32_t max_size() const;
    uint32_t buffer_size() const;

    const Buffer* shadow_global_buffer{nullptr};

    friend bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);
    friend bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);

    void set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec);
    void validate_total_size(uint32_t total_size);

    // Fields are accessible
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

bool operator==(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs);
bool operator!=(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs);

}  // namespace tt::tt_metal
