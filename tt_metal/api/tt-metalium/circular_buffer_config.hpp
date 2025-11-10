// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>

// Forward Declaration
namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
class Buffer;
struct Tile;
class CircularBufferConfigImpl;  // Internal API
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using CBHandle = uintptr_t;

class CircularBufferConfig {
public:
    // Static circular buffer spec
    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec);

    // User is expected to use the builder here.
    CircularBufferConfig(uint32_t total_size);

    // Internal Constructor (takes ownership)
    CircularBufferConfig(CircularBufferConfigImpl&& impl);

    CircularBufferConfig(const CircularBufferConfig& other);
    CircularBufferConfig& operator=(const CircularBufferConfig& other);
    CircularBufferConfig(CircularBufferConfig&& other) noexcept = default;
    CircularBufferConfig& operator=(CircularBufferConfig&& other) noexcept = default;

    ~CircularBufferConfig();

    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);

    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);

    CircularBufferConfig& set_tile_dims(uint8_t buffer_index, const Tile& tile);

    class Builder {
    public:
        static Builder LocalBuilder(CircularBufferConfig& parent, uint8_t buffer_index);
        static Builder RemoteBuilder(CircularBufferConfig& parent, uint8_t buffer_index);

        const Builder& set_data_format(tt::DataFormat data_format) const;

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

bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);
bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs);

}  // namespace tt::tt_metal
