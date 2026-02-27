// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include "gmock/gmock.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <cstring>
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_metal::distributed {

inline bool is_device_coord_mmio_mapped(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto device_id = mesh_device->get_device(device_coord)->id();
    return cluster.get_associated_mmio_device(device_id) == device_id;
}

inline PhysicalSystemDescriptor make_physical_system_descriptor() {
    return PhysicalSystemDescriptor(
        MetalContext::instance().get_cluster().get_driver(),
        MetalContext::instance().get_distributed_context_ptr(),
        &MetalContext::instance().hal(),
        MetalContext::instance().rtoptions(),
        true);
}

// Create an L1 mesh buffer sharded to a single logical core.
inline std::shared_ptr<MeshBuffer> make_l1_mesh_buffer(
    MeshDevice* mesh_device, const CoreCoord& core, DeviceAddr size) {
    auto shard_params = ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig local_config{
        .page_size = size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return MeshBuffer::create(ReplicatedBufferConfig{.size = size}, local_config, mesh_device);
}

// Dispatch a single-core program to the given device coordinate (non-blocking).
inline void execute_program_on_device(MeshDevice& device, const MeshCoordinate& device_coord, Program program) {
    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(device_coord), std::move(program));
    EnqueueMeshWorkload(device.mesh_command_queue(), workload, false);
}

// Read a single uint64 from device L1.
inline uint64_t read_l1_uint64(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr) {
    uint64_t val = 0;
    MetalContext::instance().get_cluster().read_core(
        &val,
        sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
    return val;
}

// Read an array of uint64 from device L1 into a pre-sized vector.
inline void read_l1_uint64s(
    const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr, std::vector<uint64_t>& out) {
    MetalContext::instance().get_cluster().read_core(
        out.data(),
        out.size() * sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
}

// Returns the AI clock frequency in MHz, which equals cycles-per-microsecond.
// Queried from the cluster so the value is correct regardless of the device's
// actual operating frequency rather than being hardcoded.
inline double get_cycles_per_us(const MeshDevice& mesh_device) {
    auto chip_id = mesh_device.get_device(MeshCoordinate(0, 0))->id();
    return static_cast<double>(MetalContext::instance().get_cluster().get_device_aiclk(chip_id));
}

using HDSocketFixture = MeshDevice1x2Fixture;

}  // namespace tt::tt_metal::distributed
