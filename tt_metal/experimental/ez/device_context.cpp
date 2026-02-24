// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/device_context.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::experimental::ez {

DeviceContext::DeviceContext(int device_id) :
    mesh_device_(distributed::MeshDevice::create_unit_mesh(device_id)), owns_device_(true) {}

DeviceContext::DeviceContext(size_t rows, size_t cols) :
    mesh_device_(distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::MeshShape(rows, cols)))),
    owns_device_(true) {}

DeviceContext::DeviceContext(std::shared_ptr<distributed::MeshDevice> device) :
    mesh_device_(std::move(device)), owns_device_(false) {}

DeviceContext::~DeviceContext() {
    if (owns_device_ && mesh_device_) {
        try {
            mesh_device_->close();
        } catch (...) {
        }
    }
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept :
    mesh_device_(std::move(other.mesh_device_)), owns_device_(other.owns_device_) {
    other.owns_device_ = false;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this != &other) {
        if (owns_device_ && mesh_device_) {
            mesh_device_->close();
        }
        mesh_device_ = std::move(other.mesh_device_);
        owns_device_ = other.owns_device_;
        other.owns_device_ = false;
    }
    return *this;
}

std::shared_ptr<distributed::MeshBuffer> DeviceContext::dram_buffer(
    DeviceAddr size, DeviceAddr page_size) const {
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig replicated_config{.size = size};
    return distributed::MeshBuffer::create(replicated_config, local_config, mesh_device_.get());
}

std::shared_ptr<distributed::MeshBuffer> DeviceContext::l1_buffer(
    DeviceAddr size, DeviceAddr page_size) const {
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig replicated_config{.size = size};
    return distributed::MeshBuffer::create(replicated_config, local_config, mesh_device_.get());
}

std::shared_ptr<distributed::MeshBuffer> DeviceContext::sharded_l1_buffer(const ShardConfig& config) const {
    auto ts = tt::tile_size(config.fmt);
    auto total_pages =
        static_cast<DeviceAddr>(config.tensor2d_shape_in_pages[0]) * config.tensor2d_shape_in_pages[1];
    auto total_size = total_pages * ts;

    ShardSpecBuffer shard_spec(
        config.cores, config.shard_shape, config.orientation, config.page_shape, config.tensor2d_shape_in_pages);
    BufferShardingArgs sharding_args(shard_spec, config.layout);

    distributed::DeviceLocalBufferConfig local_config{
        .page_size = ts,
        .buffer_type = BufferType::L1,
        .sharding_args = sharding_args,
    };
    distributed::ReplicatedBufferConfig replicated_config{.size = total_size};
    return distributed::MeshBuffer::create(replicated_config, local_config, mesh_device_.get());
}

std::shared_ptr<distributed::MeshBuffer> DeviceContext::dram_tile_buffer(
    uint32_t n_tiles, tt::DataFormat fmt) const {
    auto ts = tt::tile_size(fmt);
    return dram_buffer(n_tiles * ts, ts);
}

void DeviceContext::run(Program&& program) {
    launch(std::move(program));
    finish();
}

void DeviceContext::launch(Program&& program) {
    distributed::MeshWorkload workload;
    workload.add_program(full_range(), std::move(program));
    distributed::EnqueueMeshWorkload(cq(), workload, false);
}

void DeviceContext::finish() { distributed::Finish(cq()); }

CoreCoord DeviceContext::physical_core(const CoreCoord& logical) const {
    return mesh_device_->worker_core_from_logical_core(logical);
}

distributed::MeshDevice& DeviceContext::device() { return *mesh_device_; }

const distributed::MeshDevice& DeviceContext::device() const { return *mesh_device_; }

distributed::MeshCommandQueue& DeviceContext::cq() { return mesh_device_->mesh_command_queue(); }

const distributed::MeshCommandQueue& DeviceContext::cq() const { return mesh_device_->mesh_command_queue(); }

distributed::MeshCoordinateRange DeviceContext::full_range() const {
    return distributed::MeshCoordinateRange(mesh_device_->shape());
}

}  // namespace tt::tt_metal::experimental::ez
