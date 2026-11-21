// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

// Stand-alone example demonstrating usage of native multi-device TT-Metalium APIs
// for issuing Read and Write commands to a memory buffer spanning multiple devices in a mesh.
//
// The example demonstrates how to:
// 1. Perform a lock-step allocation of an L1 MeshBuffer: the same buffer, at the same address, on every device
// 2. Distribute data across the mesh by writing each device its own part of it
// 3. Read each device's part back to a local buffer
// 4. Verify that the data read back matches the original data
//
// TT-NN distributed tensors split data across a mesh and gather it back for you; see "Distributing Tensor to
// MeshDevice" in tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md.
int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& cq = mesh_device->mesh_command_queue();

    // Each device holds one 32x32 tile of the distributed data.
    uint32_t tile_size_bytes = tt::tile_size(tt::DataFormat::UInt32);

    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::L1, .bottom_up = false};
    auto buffer_config = ReplicatedBufferConfig{.size = tile_size_bytes};

    // Allocate the buffer in L1 memory on every device in the mesh.
    auto mesh_buffer = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    // Write random data to each device's part of the buffer, then read it back to a local buffer. The data read back
    // from every device should match what was written to it.
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(tile_size_bytes, 1, seed++);
        WriteShard(cq, mesh_buffer, src_data, coord, true /* blocking */);

        std::vector<uint32_t> read_back_data;
        ReadShard(cq, read_back_data, mesh_buffer, coord);
        assert(src_data == read_back_data);
    }

    return 0;
}
