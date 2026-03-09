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

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}

namespace tt::tt_metal::distributed {

bool is_device_coord_mmio_mapped(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord);

tt::tt_metal::PhysicalSystemDescriptor make_physical_system_descriptor();

// Create an L1 mesh buffer sharded to a single logical core.
std::shared_ptr<MeshBuffer> make_l1_mesh_buffer(MeshDevice* mesh_device, const CoreCoord& core, DeviceAddr size);

// Dispatch a single-core program to the given device coordinate (non-blocking).
void execute_program_on_device(MeshDevice& device, const MeshCoordinate& device_coord, Program program);

// Read a single uint64 from device L1.
uint64_t read_l1_uint64(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr);

// Read an array of uint64 from device L1 into a pre-sized vector.
void read_l1_uint64s(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr, std::vector<uint64_t>& out);

// Returns the AI clock frequency in MHz, which equals cycles-per-microsecond.
// Queried from the cluster so the value is correct regardless of the device's
// actual operating frequency rather than being hardcoded.
double get_cycles_per_us(const MeshDevice& mesh_device);

using HDSocketFixture = MeshDevice1x2Fixture;

}  // namespace tt::tt_metal::distributed
