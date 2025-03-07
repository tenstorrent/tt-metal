
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_buffer.hpp>
#include <mesh_coord.hpp>
#include <mesh_device_view.hpp>
#include <overloaded.hpp>
#include <tt_metal.hpp>

namespace tt::tt_metal::distributed {
namespace {

void validate_mesh_buffer_config(const MeshBufferConfig& config, const MeshDevice& mesh_device) {
    if (std::holds_alternative<ReplicatedBufferConfig>(config)) {
        // Nothing to validate.
        return;
    }

    const auto& sharded_config = std::get<ShardedBufferConfig>(config);
    const auto [global_buffer_height, global_buffer_width] = sharded_config.global_buffer_shape;
    const auto [shard_height, shard_width] = sharded_config.physical_shard_shape();

    TT_FATAL(
        (global_buffer_height % shard_height == 0) and (global_buffer_width % shard_width == 0),
        "Global buffer shape must be aligned with the shard shape: requested buffer shape: ({}, {}), shard "
        "shape: ({}, {})",
        global_buffer_height,
        global_buffer_width,
        shard_height,
        shard_width);

    const auto num_shard_rows = global_buffer_height / shard_height;
    const auto num_shard_cols = global_buffer_width / shard_width;
    auto num_shards = num_shard_rows * num_shard_cols;

    // The following check needs to account for shard orientation. The scaling factor for
    // replication depends on which orientation we shard/replicate to when writing to device.
    const auto& [height_replicated, width_replicated] = sharded_config.replicated_dims();
    if (height_replicated and width_replicated) {
        // Pure replication
        num_shards *= mesh_device.num_cols() * mesh_device.num_rows();
    } else if (height_replicated or width_replicated) {
        // Replication along row or column dim.
        num_shards *=
            ((sharded_config.shard_orientation == ShardOrientation::ROW_MAJOR) * (mesh_device.num_rows()) +
             (sharded_config.shard_orientation == ShardOrientation::COL_MAJOR) * (mesh_device.num_cols()));
    }
    TT_FATAL(
        num_shards <= mesh_device.num_devices(),
        "The sharded tensor does not fit on the Mesh. Num shards in buffer {}, Num Devices {}",
        num_shards,
        mesh_device.num_devices());
}

size_t generate_unique_mesh_id() {
    static std::atomic<size_t> next_id{0};
    return next_id++;
}

// Helper function to verify all Buffers in the MeshBuffer have the same value, and return it
template <typename F>
decltype(auto) validate_and_get_reference_value(
    const MeshContainer<std::shared_ptr<Buffer>>& buffers,
    F&& func,
    const std::source_location& loc = std::source_location::current()) {
    // Get reference to first device's value
    decltype(auto) reference_value = std::forward<F>(func)(buffers.begin()->value());

    // Validate all other buffers match
    for (auto& [coord, buffer] : buffers) {
        const auto& current_value = std::forward<F>(func)(buffer);
        if (current_value != reference_value) {
            TT_THROW(
                "{} [{}:{}] failed: Buffer {} returned value that differs from reference. "
                "Expected: {}, Actual: {}",
                loc.function_name(),
                loc.file_name(),
                loc.line(),
                buffer->unique_id(),
                reference_value,
                current_value);
        }
    }
    return reference_value;
}

BufferPageMapping generate_buffer_page_mapping(const MeshBuffer& buffer) {
    BufferPageMapping buffer_page_mapping;

    if (buffer.size() == 0) {
        return buffer_page_mapping;
    }
    auto shard_spec = buffer.shard_spec();

    bool row_major = shard_spec.orientation() == ShardOrientation::ROW_MAJOR;
    uint32_t num_cores = buffer.num_cores().value();

    buffer_page_mapping.all_cores_ = corerange_to_cores(shard_spec.grid(), num_cores, row_major);
    TT_FATAL(
        num_cores == buffer_page_mapping.all_cores_.size(),
        "Buffer has {} cores, but page mapping expects {} cores",
        num_cores,
        buffer_page_mapping.all_cores_.size());
    uint32_t core_id = 0;
    for (const auto& core : buffer_page_mapping.all_cores_) {
        buffer_page_mapping.core_to_core_id_.insert({core, core_id});
        core_id++;
    }

    uint32_t num_dev_pages = buffer.num_dev_pages();
    auto [core_host_page_indices, shard_shape] = core_to_host_pages(
        num_dev_pages,
        shard_spec.num_pages(),
        num_cores,
        buffer.buffer_layout(),
        shard_spec.page_shape,
        shard_spec.shape(),
        shard_spec.tensor2d_shape_in_pages);

    buffer_page_mapping.core_host_page_indices_ = std::vector<std::vector<uint32_t>>(num_cores);

    buffer_page_mapping.dev_page_to_host_page_mapping_ =
        std::vector<std::optional<uint32_t>>(num_dev_pages, std::nullopt);
    buffer_page_mapping.dev_page_to_core_mapping_ = std::vector<uint32_t>(num_dev_pages);

    buffer_page_mapping.host_page_to_local_shard_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.host_page_to_dev_page_mapping_ = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.core_shard_shape_ = std::move(shard_shape);
    uint32_t dev_page_index = 0;

    auto shape_in_pages = shard_spec.shape_in_pages();
    for (uint32_t core_index = 0; core_index < core_host_page_indices.size(); core_index++) {
        uint32_t valid_shard_page = 0;
        buffer_page_mapping.core_host_page_indices_[core_index].reserve(shard_spec.num_pages());
        uint32_t shard_page_id = 0;
        for (uint32_t shard_page_x = 0; shard_page_x < shape_in_pages[0]; shard_page_x++) {
            for (uint32_t shard_page_y = 0; shard_page_y < shape_in_pages[1]; shard_page_y++) {
                buffer_page_mapping.dev_page_to_core_mapping_[dev_page_index] = core_index;
                if (shard_page_x < buffer_page_mapping.core_shard_shape_[core_index][0] and
                    shard_page_y < buffer_page_mapping.core_shard_shape_[core_index][1]) {
                    uint32_t host_page = core_host_page_indices[core_index][valid_shard_page];
                    buffer_page_mapping.dev_page_to_host_page_mapping_[dev_page_index] = host_page;
                    buffer_page_mapping.core_host_page_indices_[core_index].push_back(host_page);
                    buffer_page_mapping.host_page_to_local_shard_page_mapping_[host_page] = shard_page_id;
                    buffer_page_mapping.host_page_to_dev_page_mapping_[host_page] = dev_page_index;
                    valid_shard_page++;
                }
                dev_page_index++;
                shard_page_id++;
            }
        }
    }

    return buffer_page_mapping;
}

}  // namespace

uint32_t ShardedBufferConfig::compute_datum_size_bytes() const {
    return global_size / (global_buffer_shape.height() * global_buffer_shape.width());
}

std::pair<bool, bool> ShardedBufferConfig::replicated_dims() const {
    return {shard_shape.height() == 0, shard_shape.width() == 0};
}

Shape2D ShardedBufferConfig::physical_shard_shape() const {
    const auto [shard_height, shard_width] = shard_shape;
    const auto [global_height, global_width] = global_buffer_shape;
    return Shape2D(shard_height == 0 ? global_height : shard_height, shard_width == 0 ? global_width : shard_width);
}

std::shared_ptr<MeshBuffer> MeshBuffer::create(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalBufferConfig& device_local_config,
    MeshDevice* mesh_device,
    std::optional<DeviceAddr> address) {
    validate_mesh_buffer_config(mesh_buffer_config, *mesh_device);

    const DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.size; },
            [mesh_device](const ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.physical_shard_shape();
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    std::shared_ptr<MeshBuffer> mesh_buffer;
    if (!address.has_value()) {
        // Rely on the MeshDevice allocator to provide the address for the entire mesh buffer.
        // The address provided to the backing buffer is used as the address for the MeshBuffer object.
        std::shared_ptr<Buffer> backing_buffer = Buffer::create(
            mesh_device,
            device_local_size,
            device_local_config.page_size,
            device_local_config.buffer_type,
            device_local_config.buffer_layout,
            device_local_config.shard_parameters,
            device_local_config.bottom_up);

        mesh_buffer = std::shared_ptr<MeshBuffer>(new MeshBuffer(
            mesh_buffer_config,
            device_local_config,
            device_local_size,
            mesh_device,
            generate_unique_mesh_id(),
            std::move(backing_buffer)));
    } else {
        mesh_buffer = std::shared_ptr<MeshBuffer>(new MeshBuffer(
            mesh_buffer_config,
            device_local_config,
            address.value(),
            device_local_size,
            mesh_device,
            generate_unique_mesh_id()));
    }

    mesh_buffer->initialize_device_buffers();

    return mesh_buffer;
}

void MeshBuffer::initialize_device_buffers() {
    auto init_device_buffer_at_address = [this](const MeshCoordinate& coord) {
        std::shared_ptr<Buffer> buffer = Buffer::create(
            device()->get_device(coord),
            address_,
            device_local_size_,
            device_local_config_.page_size,
            device_local_config_.buffer_type,
            device_local_config_.buffer_layout,
            device_local_config_.shard_parameters,
            device_local_config_.bottom_up);
        return buffer;
    };

    for (auto& [coord, device_buffer] : buffers_) {
        device_buffer = init_device_buffer_at_address(coord);
    }

    auto mesh_device = mesh_device_.lock();
    if (sub_device_id().has_value()) {
        allocator_ = mesh_device->allocator(sub_device_id().value()).get();
    } else {
        allocator_ = mesh_device->allocator().get();
    }
}

bool MeshBuffer::is_allocated() const { return not std::holds_alternative<DeallocatedState>(state_); }

MeshBuffer::~MeshBuffer() { deallocate(); }

void MeshBuffer::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        state_ = DeallocatedState{};
        return;
    }

    // Special handling is required if MeshDevice is already deallocated
    if (std::holds_alternative<OwnedBufferState>(state_)) {
        auto& owned_state = std::get<OwnedBufferState>(state_);
        owned_state.backing_buffer->mark_as_deallocated();
    }
    state_ = DeallocatedState{};
}

MeshDevice* MeshBuffer::device() const {
    auto device = mesh_device_.lock();
    TT_FATAL(device, "Can't get device from mesh buffer, already deallocated");
    return device.get();
}

std::shared_ptr<Buffer> MeshBuffer::get_device_buffer(const MeshCoordinate& device_coord) const {
    return buffers_.at(device_coord);
}

DeviceAddr MeshBuffer::size() const {
    return std::visit(
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.size; },
            [&](const ShardedBufferConfig& config) { return config.global_size; }},
        config_);
}

MeshBufferLayout MeshBuffer::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(config_) ? MeshBufferLayout::REPLICATED
                                                                   : MeshBufferLayout::SHARDED;
}

const ShardedBufferConfig& MeshBuffer::global_shard_spec() const {
    TT_FATAL(is_sharded(), "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

uint32_t MeshBuffer::datum_size_bytes() const {
    // Limitation for now.
    TT_FATAL(is_sharded(), "Can only query datum size for buffers sharded across the Mesh");
    return this->global_shard_spec().compute_datum_size_bytes();
}

Shape2D MeshBuffer::physical_shard_shape() const {
    TT_FATAL(is_sharded(), "Can only query physical shard shape for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    return sharded_config.physical_shard_shape();
}

std::pair<bool, bool> MeshBuffer::replicated_dims() const {
    TT_FATAL(is_sharded(), "Can only query replicated dims for buffers sharded across the Mesh");
    return this->global_shard_spec().replicated_dims();
}

void MeshBuffer::set_page_size(DeviceAddr page_size) {
    TT_FATAL(page_size == 0 ? size() == 0 : size() % page_size == 0, "buffer size must be divisible by new page size");
    device_local_config_.page_size = page_size;
    for (auto& [coord, device_buffer] : buffers_) {
        device_buffer->set_page_size(page_size);
    }
}

uint32_t MeshBuffer::num_dev_pages() const {
    if (!is_sharded()) {
        return this->num_pages();
    }

    return this->shard_spec().num_pages() * this->num_cores().value();
}

CoreType MeshBuffer::core_type() const {
    switch (buffer_type()) {
        case BufferType::DRAM: return CoreType::DRAM;
        case BufferType::L1:
        case BufferType::L1_SMALL: return CoreType::WORKER;
        default: TT_THROW("Unknown CoreType {} for buffer", buffer_type());
    }
}

bool MeshBuffer::is_l1() const { return buffer_type() == BufferType::L1 or buffer_type() == BufferType::L1_SMALL; }
bool MeshBuffer::is_dram() const { return buffer_type() == BufferType::DRAM || buffer_type() == BufferType::TRACE; }
bool MeshBuffer::is_trace() const { return buffer_type() == BufferType::TRACE; }

bool MeshBuffer::is_valid_region(const BufferRegion& region) const {
    return region.offset + region.size <= this->size();
}

bool MeshBuffer::is_valid_partial_region(const BufferRegion& region) const {
    return this->is_valid_region(region) && (region.offset > 0 || region.size != this->size());
}

DeviceAddr MeshBuffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    return validate_and_get_reference_value(this->buffers_, [bank_id, page_index](const auto& buffer) {
        return buffer->page_address(bank_id, page_index);
    });
}

DeviceAddr MeshBuffer::bank_local_page_address(uint32_t bank_id, uint32_t page_index) const {
    return validate_and_get_reference_value(this->buffers_, [bank_id, page_index](const auto& buffer) {
        return buffer->bank_local_page_address(bank_id, page_index);
    });
}

uint32_t MeshBuffer::alignment() const {
    return validate_and_get_reference_value(this->buffers_, [](const auto& buffer) { return buffer->alignment(); });
}

DeviceAddr MeshBuffer::aligned_page_size() const {
    return validate_and_get_reference_value(
        this->buffers_, [](const auto& buffer) { return buffer->aligned_page_size(); });
}

DeviceAddr MeshBuffer::aligned_size() const {
    return validate_and_get_reference_value(this->buffers_, [](const auto& buffer) { return buffer->aligned_size(); });
}

DeviceAddr MeshBuffer::aligned_size_per_bank() const {
    return validate_and_get_reference_value(
        this->buffers_, [](const auto& buffer) { return buffer->aligned_size_per_bank(); });
}

DeviceAddr MeshBuffer::sharded_page_address(uint32_t bank_id, uint32_t page_index) const {
    TT_FATAL(is_sharded(), "Can only query shard spec for buffers sharded across the Mesh");
    return validate_and_get_reference_value(this->buffers_, [bank_id, page_index](const auto& buffer) {
        return buffer->sharded_page_address(bank_id, page_index);
    });
}

ShardSpecBuffer MeshBuffer::shard_spec() const {
    TT_FATAL(is_sharded(), "Can only query shard spec for buffers sharded across the Mesh");
    return device_local_config_.shard_parameters.value();
}

void MeshBuffer::set_shard_spec(const ShardSpecBuffer& shard_spec) {
    TT_FATAL(is_sharded(), "Can only set shard spec for buffers sharded across the Mesh");
    device_local_config_.shard_parameters = shard_spec;
    for (auto& [coord, device_buffer] : buffers_) {
        device_buffer->set_shard_spec(shard_spec);
    }
}

std::optional<uint32_t> MeshBuffer::num_cores() const {
    if (!is_sharded()) {
        return std::nullopt;
    }

    return this->shard_spec().tensor_shard_spec.grid.num_cores();
}

bool MeshBuffer::is_sharded() const { return this->global_layout() == MeshBufferLayout::SHARDED; }

const std::shared_ptr<const BufferPageMapping>& MeshBuffer::get_buffer_page_mapping() {
    TT_FATAL(is_sharded(), "Can only get page mapping for buffers sharded across the Mesh");
    if (!this->buffer_page_mapping_) {
        this->buffer_page_mapping_ = std::make_shared<const BufferPageMapping>(generate_buffer_page_mapping(*this));
    }
    return this->buffer_page_mapping_;
}

std::optional<SubDeviceId> MeshBuffer::sub_device_id() const {
    return validate_and_get_reference_value(this->buffers_, [](const auto& buffer) { return buffer->sub_device_id(); });
}

}  // namespace tt::tt_metal::distributed
