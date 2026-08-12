// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer_config.hpp"
#include "impl/buffers/circular_buffer_config_impl.hpp"

#include <unordered_map>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include "buffer.hpp"
#include "hal.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

namespace {

void validate_unpack_face_geometry(uint32_t face_r_dim, uint32_t num_faces) {
    TT_FATAL(face_r_dim > 0, "face_r_dim must be > 0");
    TT_FATAL(
        face_r_dim <= tt::constants::FACE_HEIGHT,
        "face_r_dim ({}) must be <= FACE_HEIGHT ({})",
        face_r_dim,
        tt::constants::FACE_HEIGHT);
    TT_FATAL(num_faces > 0, "num_faces must be > 0");
}

}  // namespace

namespace tt::tt_metal {

// ---------------------------------------------------------------------------
// CircularBufferConfigImpl
// ---------------------------------------------------------------------------

CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec) :
    total_size_(total_size), globally_allocated_address_(std::nullopt) {
    this->set_config(data_format_spec);
}

CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size, const std::map<uint8_t, DataType>& data_type_spec) :
    total_size_(total_size), globally_allocated_address_(std::nullopt) {
    std::map<uint8_t, tt::DataFormat> data_format_spec;
    for (const auto& [idx, dtype] : data_type_spec) {
        data_format_spec[idx] = datatype_to_dataformat_converter(dtype);
    }
    this->set_config(data_format_spec);
}

CircularBufferConfigImpl::CircularBufferConfigImpl(uint32_t total_size) :
    total_size_(total_size), globally_allocated_address_(std::nullopt) {}

CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer) :
    total_size_(total_size) {
    this->set_globally_allocated_address(buffer);
    this->set_config(data_format_spec);
}

CircularBufferConfigImpl::CircularBufferConfigImpl(const CBDescriptor& descriptor) :
    total_size_(descriptor.total_size) {
    TT_FATAL(
        !(descriptor.buffer && descriptor.tensor),
        "CBDescriptor cannot specify both buffer and tensor as the globally-allocated backing storage");

    const Buffer* backing_buffer = descriptor.buffer;
    if (!backing_buffer && descriptor.tensor) {
        backing_buffer = descriptor.tensor->mesh_buffer().get_reference_buffer();
    }
    if (backing_buffer) {
        this->set_globally_allocated_address(*backing_buffer);
        if (descriptor.address_offset != 0) {
            uint32_t l1_alignment = hal::get_l1_alignment();
            TT_FATAL(
                descriptor.address_offset % l1_alignment == 0,
                "address_offset ({}) must be aligned to L1 alignment ({})",
                descriptor.address_offset,
                l1_alignment);
            this->address_offset_ = descriptor.address_offset;
            this->globally_allocated_address_ = this->globally_allocated_address_.value() + descriptor.address_offset;
            this->max_size_ -= descriptor.address_offset;
            TT_FATAL(
                this->total_size_ <= this->max_size_,
                "address_offset ({}) + total_size ({}) exceeds buffer bank size ({})",
                descriptor.address_offset,
                this->total_size_,
                this->max_size_ + descriptor.address_offset);
        }
    }

    auto process_format_descriptor = [this](const CBFormatDescriptor& format_descriptor) {
        uint32_t max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
        if (format_descriptor.buffer_index > max_cbs - 1) {
            TT_THROW(
                "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                format_descriptor.buffer_index,
                max_cbs);
        }
        this->data_formats_[format_descriptor.buffer_index] = format_descriptor.data_format;
        if (this->total_size_ % format_descriptor.page_size != 0) {
            TT_THROW(
                "Total circular buffer size {} B must be divisible by page size {} B",
                this->total_size_,
                format_descriptor.page_size);
        }
        this->page_sizes_[format_descriptor.buffer_index] = format_descriptor.page_size;
        if (format_descriptor.tile) {
            this->tiles_[format_descriptor.buffer_index] = Tile(
                {format_descriptor.tile->height, format_descriptor.tile->width}, format_descriptor.tile->transpose);
        }
        if (format_descriptor.face_geometry) {
            const auto& [face_r_dim, num_faces] = *format_descriptor.face_geometry;
            validate_unpack_face_geometry(face_r_dim, num_faces);
            this->unpack_face_geometry_[format_descriptor.buffer_index] = format_descriptor.face_geometry;
        }
    };
    this->buffer_indices_.reserve(descriptor.format_descriptors.size() + descriptor.remote_format_descriptors.size());
    this->local_buffer_indices_.reserve(descriptor.format_descriptors.size());
    this->remote_buffer_indices_.reserve(descriptor.remote_format_descriptors.size());
    for (const auto& format_descriptor : descriptor.format_descriptors) {
        process_format_descriptor(format_descriptor);
        this->buffer_indices_.insert(format_descriptor.buffer_index);
        this->local_buffer_indices_.insert(format_descriptor.buffer_index);
    }
    for (const auto& format_descriptor : descriptor.remote_format_descriptors) {
        process_format_descriptor(format_descriptor);
        this->buffer_indices_.insert(format_descriptor.buffer_index);
        this->remote_buffer_indices_.insert(format_descriptor.buffer_index);
    }
}

CircularBufferConfigImpl::CircularBufferConfigImpl(
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
    uint32_t buffer_size) :
    total_size_(total_size),
    globally_allocated_address_(globally_allocated_address),
    data_formats_(data_formats),
    page_sizes_(page_sizes),
    tiles_(tiles),
    unpack_face_geometry_(unpack_face_geometry),
    buffer_indices_(buffer_indices),
    local_buffer_indices_(local_buffer_indices),
    remote_buffer_indices_(remote_buffer_indices),
    dynamic_cb_(dynamic_cb),
    max_size_(max_size),
    buffer_size_(buffer_size) {
    for (const auto& geom : unpack_face_geometry) {
        if (geom.has_value()) {
            validate_unpack_face_geometry(geom->face_r_dim, geom->num_faces);
        }
    }
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_page_size(uint8_t buffer_index, uint32_t page_size) {
    uint32_t max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    if (buffer_index > max_cbs - 1) {
        TT_THROW("Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, max_cbs);
    }
    if (!this->buffer_indices_.contains(buffer_index)) {
        TT_THROW(
            "Illegal circular buffer index {}. Page size can only be specified for buffer indices configured "
            "during config creation",
            buffer_index);
    }
    if (this->total_size_ % page_size != 0) {
        TT_THROW(
            "Failed allocation attempt on buffer index {}. Total circular buffer size {} B must be divisible by page "
            "size {} B",
            buffer_index,
            this->total_size_,
            page_size);
    }

    this->page_sizes_[buffer_index] = page_size;
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_total_size(uint32_t total_size) {
    if (dynamic_cb_) {
        TT_FATAL(
            total_size <= this->max_size_,
            "Cannot set circular buffer size to {}. This is larger than the associated dynamically allocated "
            "L1 buffer bank size of {} B",
            total_size,
            this->max_size_);
    }
    this->total_size_ = total_size;
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address(const Buffer& buffer) {
    return this->set_globally_allocated_address_and_total_size(buffer, this->total_size_);
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address(const MeshTensor& tensor) {
    return set_globally_allocated_address(*tensor.mesh_buffer().get_reference_buffer());
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address_and_total_size(
    const MeshTensor& tensor, uint32_t total_size) {
    return set_globally_allocated_address_and_total_size(*tensor.mesh_buffer().get_reference_buffer(), total_size);
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address_and_total_size(
    const Buffer& buffer, uint32_t total_size) {
    if (not buffer.is_l1()) {
        TT_THROW("Only L1 buffers can have an associated circular buffer!");
    }
    this->globally_allocated_address_ = buffer.address();
    this->dynamic_cb_ = true;
    this->max_size_ = buffer.aligned_size_per_bank();
    this->buffer_size_ = buffer.aligned_size();
    this->shadow_global_buffer_ = &buffer;
    this->set_total_size(total_size);
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_tile_dims(uint8_t buffer_index, const Tile& tile) {
    this->tiles_[buffer_index] = tile;
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_unpack_face_geometry(
    uint8_t buffer_index, uint32_t face_r_dim, uint32_t num_faces) {
    uint32_t max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    if (buffer_index > max_cbs - 1) {
        TT_THROW("Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, max_cbs);
    }
    if (!this->buffer_indices_.contains(buffer_index)) {
        TT_THROW(
            "Illegal circular buffer index {}. Unpack face geometry can only be set for buffer indices configured "
            "during config creation",
            buffer_index);
    }
    validate_unpack_face_geometry(face_r_dim, num_faces);
    this->unpack_face_geometry_[buffer_index] = FaceGeometry{face_r_dim, num_faces};
    return *this;
}

void CircularBufferConfigImpl::set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec) {
    uint32_t max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    if (data_format_spec.size() > max_cbs) {
        TT_THROW(
            "Only {} circular buffer slots are available but data formats are specified for {} indices",
            max_cbs,
            data_format_spec.size());
    }

    for (const auto& [buffer_index, data_format] : data_format_spec) {
        if (buffer_index > max_cbs - 1) {
            TT_THROW("Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, max_cbs);
        }
        this->data_formats_[buffer_index] = data_format;
        this->buffer_indices_.insert(buffer_index);
        this->local_buffer_indices_.insert(buffer_index);
    }
}

bool operator==(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs) {
    return lhs.total_size() == rhs.total_size() &&
           lhs.globally_allocated_address() == rhs.globally_allocated_address() &&
           lhs.data_formats() == rhs.data_formats() && lhs.page_sizes() == rhs.page_sizes() &&
           lhs.tiles() == rhs.tiles() && lhs.unpack_face_geometry() == rhs.unpack_face_geometry() &&
           lhs.shadow_global_buffer() == rhs.shadow_global_buffer();
}

bool operator!=(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs) { return !(lhs == rhs); }

// ---------------------------------------------------------------------------
// CircularBufferConfig shell
// ---------------------------------------------------------------------------

CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec) :
    impl_(std::make_unique<CircularBufferConfigImpl>(total_size, data_format_spec)) {}

CircularBufferConfig::CircularBufferConfig(uint32_t total_size) :
    impl_(std::make_unique<CircularBufferConfigImpl>(total_size)) {}

CircularBufferConfig::CircularBufferConfig(CircularBufferConfigImpl impl) :
    impl_(std::make_unique<CircularBufferConfigImpl>(std::move(impl))) {}

CircularBufferConfig make_circular_buffer_config(CircularBufferConfigImpl impl) {
    return CircularBufferConfig(std::move(impl));
}

CircularBufferConfig::~CircularBufferConfig() = default;

CircularBufferConfig::CircularBufferConfig(const CircularBufferConfig& other) :
    impl_(other.impl_ ? std::make_unique<CircularBufferConfigImpl>(*other.impl_) : nullptr) {}

CircularBufferConfig& CircularBufferConfig::operator=(const CircularBufferConfig& other) {
    if (this == &other) {
        return *this;
    }
    impl_ = other.impl_ ? std::make_unique<CircularBufferConfigImpl>(*other.impl_) : nullptr;
    return *this;
}

CircularBufferConfig::CircularBufferConfig(CircularBufferConfig&& other) noexcept = default;
CircularBufferConfig& CircularBufferConfig::operator=(CircularBufferConfig&& other) noexcept = default;

CircularBufferConfigImpl& CircularBufferConfig::impl() {
    TT_FATAL(impl_ != nullptr, "CircularBufferConfig is in a moved-from state.");
    return *impl_;
}

const CircularBufferConfigImpl& CircularBufferConfig::impl() const {
    TT_FATAL(impl_ != nullptr, "CircularBufferConfig is in a moved-from state.");
    return *impl_;
}

CircularBufferConfig& CircularBufferConfig::set_page_size(uint8_t buffer_index, uint32_t page_size) {
    impl().set_page_size(buffer_index, page_size);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_globally_allocated_address(const Buffer& buffer) {
    impl().set_globally_allocated_address(buffer);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_globally_allocated_address(const MeshTensor& tensor) {
    impl().set_globally_allocated_address(tensor);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_tile_dims(uint8_t buffer_index, const Tile& tile) {
    impl().set_tile_dims(buffer_index, tile);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_unpack_face_geometry(
    uint8_t buffer_index, uint32_t face_r_dim, uint32_t num_faces) {
    impl().set_unpack_face_geometry(buffer_index, face_r_dim, num_faces);
    return *this;
}

CircularBufferConfig::Builder CircularBufferConfig::Builder::LocalBuilder(
    CircularBufferConfig& parent, uint8_t buffer_index) {
    auto is_remote_index = parent.impl().remote_buffer_indices().contains(buffer_index);
    if (is_remote_index) {
        TT_THROW("Buffer index {} is already marked as remote", buffer_index);
    }
    auto builder = Builder(parent, buffer_index);
    parent.impl().insert_local_buffer_index(buffer_index);
    return builder;
}

CircularBufferConfig::Builder CircularBufferConfig::Builder::RemoteBuilder(
    CircularBufferConfig& parent, uint8_t buffer_index) {
    auto is_local_index = parent.impl().local_buffer_indices().contains(buffer_index);
    if (is_local_index) {
        TT_THROW("Buffer index {} is already marked as local", buffer_index);
    }
    if (!parent.impl().remote_buffer_indices().contains(buffer_index)) {
        TT_FATAL(parent.impl().remote_buffer_indices().empty(), "Can only specify one remote buffer index per config");
    }
    auto builder = Builder(parent, buffer_index);
    parent.impl().insert_remote_buffer_index(buffer_index);
    return builder;
}

CircularBufferConfig::Builder::Builder(CircularBufferConfig& parent, uint8_t buffer_index) :
    parent_(parent), buffer_index_(buffer_index) {
    uint32_t max_cbs = tt::tt_metal::MetalContext::instance().hal().get_arch_num_circular_buffers();
    if (buffer_index > max_cbs - 1) {
        TT_THROW("Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, max_cbs);
    }
    parent_.impl().insert_buffer_index(buffer_index_);
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_data_format(tt::DataFormat data_format) const {
    parent_.impl().set_data_format(buffer_index_, data_format);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_total_size(uint32_t total_size) const {
    parent_.impl().set_total_size(total_size);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_page_size(uint32_t page_size) const {
    parent_.set_page_size(buffer_index_, page_size);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_tile_dims(const Tile& tile) const {
    parent_.set_tile_dims(buffer_index_, tile);
    return *this;
}

CircularBufferConfig::Builder CircularBufferConfig::index(uint8_t buffer_index) {
    return Builder::LocalBuilder(*this, buffer_index);
}

CircularBufferConfig::Builder CircularBufferConfig::remote_index(uint8_t buffer_index) {
    return Builder::RemoteBuilder(*this, buffer_index);
}

bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) { return lhs.impl() == rhs.impl(); }

bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) { return !(lhs == rhs); }

}  // namespace tt::tt_metal
