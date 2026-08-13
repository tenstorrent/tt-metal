// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_set>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

class Buffer;
class MeshTensor;
enum class DataType;

class CircularBufferConfigImpl {
public:
    CircularBufferConfigImpl() = default;

    CircularBufferConfigImpl(uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec);
    CircularBufferConfigImpl(uint32_t total_size, const std::map<uint8_t, DataType>& data_type_spec);
    explicit CircularBufferConfigImpl(uint32_t total_size);
    CircularBufferConfigImpl(
        uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer);

    // Flatbuffer deserialization: set all members.
    CircularBufferConfigImpl(
        uint32_t total_size,
        std::optional<uint32_t> globally_allocated_address,
        const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats,
        const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes,
        const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles,
        const std::array<std::optional<FaceGeometry>, NUM_CIRCULAR_BUFFERS>& unpack_face_geometry,
        const std::unordered_set<uint8_t>& buffer_indices,
        const std::unordered_set<uint8_t>& local_buffer_indices,
        const std::unordered_set<uint8_t>& remote_buffer_indices,
        bool dynamic_cb,
        uint32_t max_size,
        uint32_t buffer_size);

    explicit CircularBufferConfigImpl(const CBDescriptor& descriptor);

    CircularBufferConfigImpl(const CircularBufferConfigImpl&) = default;
    CircularBufferConfigImpl& operator=(const CircularBufferConfigImpl&) = default;
    CircularBufferConfigImpl(CircularBufferConfigImpl&&) noexcept = default;
    CircularBufferConfigImpl& operator=(CircularBufferConfigImpl&&) noexcept = default;
    ~CircularBufferConfigImpl() = default;

    CircularBufferConfigImpl& set_page_size(uint8_t buffer_index, uint32_t page_size);
    CircularBufferConfigImpl& set_total_size(uint32_t total_size);
    CircularBufferConfigImpl& set_globally_allocated_address(const Buffer& buffer);
    CircularBufferConfigImpl& set_globally_allocated_address(const MeshTensor& tensor);
    CircularBufferConfigImpl& set_globally_allocated_address_and_total_size(const Buffer& buffer, uint32_t total_size);
    CircularBufferConfigImpl& set_globally_allocated_address_and_total_size(
        const MeshTensor& tensor, uint32_t total_size);
    CircularBufferConfigImpl& set_tile_dims(uint8_t buffer_index, const Tile& tile);
    CircularBufferConfigImpl& set_unpack_face_geometry(uint8_t buffer_index, uint32_t face_r_dim, uint32_t num_faces);

    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles() const { return tiles_; }
    const std::array<std::optional<FaceGeometry>, NUM_CIRCULAR_BUFFERS>& unpack_face_geometry() const {
        return unpack_face_geometry_;
    }
    uint32_t total_size() const { return total_size_; }
    std::optional<uint32_t> globally_allocated_address() const { return globally_allocated_address_; }
    const std::unordered_set<uint8_t>& buffer_indices() const { return buffer_indices_; }
    const std::unordered_set<uint8_t>& local_buffer_indices() const { return local_buffer_indices_; }
    const std::unordered_set<uint8_t>& remote_buffer_indices() const { return remote_buffer_indices_; }
    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats() const {
        return data_formats_;
    }
    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes() const { return page_sizes_; }
    bool dynamic_cb() const { return dynamic_cb_; }
    uint32_t max_size() const { return max_size_; }
    uint32_t buffer_size() const { return buffer_size_; }
    uint32_t address_offset() const { return address_offset_; }
    void set_address_offset(uint32_t offset) { address_offset_ = offset; }

    const Buffer* shadow_global_buffer() const { return shadow_global_buffer_; }
    void set_shadow_global_buffer(const Buffer* buffer) { shadow_global_buffer_ = buffer; }

    // Builder helpers (mutate index sets / formats used by CircularBufferConfig::Builder).
    void insert_buffer_index(uint8_t buffer_index) { buffer_indices_.insert(buffer_index); }
    void insert_local_buffer_index(uint8_t buffer_index) { local_buffer_indices_.insert(buffer_index); }
    void insert_remote_buffer_index(uint8_t buffer_index) { remote_buffer_indices_.insert(buffer_index); }
    void set_data_format(uint8_t buffer_index, tt::DataFormat data_format) {
        data_formats_[buffer_index] = data_format;
    }

private:
    void set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec);

    uint32_t total_size_ = 0;
    std::optional<uint32_t> globally_allocated_address_ = std::nullopt;
    std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> data_formats_;
    std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> page_sizes_;
    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles_;
    std::array<std::optional<FaceGeometry>, NUM_CIRCULAR_BUFFERS> unpack_face_geometry_;
    std::unordered_set<uint8_t> buffer_indices_;
    std::unordered_set<uint8_t> local_buffer_indices_;
    std::unordered_set<uint8_t> remote_buffer_indices_;
    bool dynamic_cb_ = false;
    // `max_size_` is used to ensure that total size does not grow beyond associated buffer size
    // `buffer_size_` is tracked to enforce the old size assertions.
    // Will be removed once tests are updated to respect the correct `max_size_` constraint
    uint32_t max_size_ = 0;
    uint32_t buffer_size_ = 0;
    uint32_t address_offset_ = 0;
    const Buffer* shadow_global_buffer_ = nullptr;
};

bool operator==(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs);
bool operator!=(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs);

}  // namespace tt::tt_metal
