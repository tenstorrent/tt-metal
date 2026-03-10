// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hd_socket_test_utils.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"

namespace tt::tt_metal::distributed {

bool is_device_coord_mmio_mapped(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto device_id = mesh_device->get_device(device_coord)->id();
    return cluster.get_associated_mmio_device(device_id) == device_id;
}

PhysicalSystemDescriptor make_physical_system_descriptor() {
    auto& driver = const_cast<tt::umd::Cluster&>(*MetalContext::instance().get_cluster().get_driver());
    return run_physical_system_discovery(
        driver,
        MetalContext::instance().get_distributed_context_ptr(),
        MetalContext::instance().rtoptions().get_target_device());
}

std::shared_ptr<MeshBuffer> make_l1_mesh_buffer(MeshDevice* mesh_device, const CoreCoord& core, DeviceAddr size) {
    auto shard_params = ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig local_config{
        .page_size = size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return MeshBuffer::create(ReplicatedBufferConfig{.size = size}, local_config, mesh_device);
}

void execute_program_on_device(MeshDevice& device, const MeshCoordinate& device_coord, Program program) {
    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(device_coord), std::move(program));
    EnqueueMeshWorkload(device.mesh_command_queue(), workload, false);
}

uint64_t read_l1_uint64(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr) {
    uint64_t val = 0;
    MetalContext::instance().get_cluster().read_core(
        &val,
        sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
    return val;
}

void read_l1_uint64s(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr, std::vector<uint64_t>& out) {
    MetalContext::instance().get_cluster().read_core(
        out.data(),
        out.size() * sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
}

double get_cycles_per_us(const MeshDevice& mesh_device) {
    auto chip_id = mesh_device.get_device(MeshCoordinate(0, 0))->id();
    return static_cast<double>(MetalContext::instance().get_cluster().get_device_aiclk(chip_id));
}

}  // namespace tt::tt_metal::distributed
