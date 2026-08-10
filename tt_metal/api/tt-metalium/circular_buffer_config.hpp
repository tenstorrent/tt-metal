// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
class Buffer;
class MeshTensor;
class CircularBufferConfigImpl;
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

    CircularBufferConfig(const CircularBufferConfig& other);
    CircularBufferConfig& operator=(const CircularBufferConfig& other);
    CircularBufferConfig(CircularBufferConfig&& other) noexcept;
    CircularBufferConfig& operator=(CircularBufferConfig&& other) noexcept;
    ~CircularBufferConfig();

    CircularBufferConfig& set_page_size(uint8_t buffer_index, uint32_t page_size);

    CircularBufferConfig& set_globally_allocated_address(const Buffer& buffer);

    CircularBufferConfig& set_globally_allocated_address(const MeshTensor& tensor);

    CircularBufferConfig& set_tile_dims(uint8_t buffer_index, const Tile& tile);

    /// Override face row count and logical face count metadata for this buffer index.
    /// This metadata feeds JIT tile-dimension arrays derived from the same descriptor fields, including
    /// unpack_tile_* arrays as well as pack/untilize behavior when applicable.
    /// Use when operand geometry differs from \ref Tile (e.g. pool tilize on compact pages with 2 logical faces).
    CircularBufferConfig& set_unpack_face_geometry(uint8_t buffer_index, uint32_t face_r_dim, uint32_t num_faces);

    class Builder {
    public:
        const Builder& set_data_format(tt::DataFormat data_format) const;

        const Builder& set_page_size(uint32_t page_size) const;

        const Builder& set_tile_dims(const Tile& tile) const;

    private:
        friend class CircularBufferConfig;

        static Builder LocalBuilder(CircularBufferConfig& parent, uint8_t buffer_index);
        static Builder RemoteBuilder(CircularBufferConfig& parent, uint8_t buffer_index);

        const Builder& set_total_size(uint32_t total_size) const;

        Builder(CircularBufferConfig& parent, uint8_t buffer_index);

        CircularBufferConfig& parent_;
        uint8_t buffer_index_;
    };

    Builder index(uint8_t buffer_index);
    Builder remote_index(uint8_t buffer_index);

    // pre-condition: the CircularBufferConfig must not be in a moved-from state.
    CircularBufferConfigImpl& impl();
    const CircularBufferConfigImpl& impl() const;

private:
    // Takes ownership of impl (flatbuffer deserialize / descriptor / tests).
    explicit CircularBufferConfig(std::unique_ptr<CircularBufferConfigImpl> impl);

    // May be nullptr if the CircularBufferConfig is in a moved-from state.
    // Avoid using pimpl_ directly; use the impl() accessor instead.
    std::unique_ptr<CircularBufferConfigImpl> pimpl_;

    friend class CircularBufferImpl;
    friend CircularBufferConfig make_circular_buffer_config(std::unique_ptr<CircularBufferConfigImpl> impl);
};

}  // namespace tt::tt_metal
