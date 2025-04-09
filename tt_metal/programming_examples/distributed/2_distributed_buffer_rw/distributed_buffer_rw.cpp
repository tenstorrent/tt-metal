// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

// Stand-alone example demonstrating usage of native multi-device TT-Metalium APIs
// for issuing Read and Write commands to a distributed memory buffer spanning
// multiple devices in a mesh.
//
// The example demonstrates how to:
// 1. Perform a lock-step allocation of a distributed L1 MeshBuffer
//    containing data scattered across multiple devices in a mesh
// 2. Enqueue a Write command to the MeshBuffer with random data
// 3. Enqueue a Read command to the MeshBuffer and read back the data to a local buffer
// 4. Verify that the data read back matches the original data
int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;
    using tt::tt_metal::distributed::ShardedBufferConfig;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& cq = mesh_device->mesh_command_queue();

    // Define the shape of the shard and the distributed buffer.
    // We will create a distributed buffer with 8 shards of {32, 32} and distribute it across the devices in the mesh.
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape = Shape2D{32 * mesh_device->num_rows(), 32 * mesh_device->num_cols()};
    uint32_t tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::UInt32);
    uint32_t distributed_buffer_size_bytes = 64 * 128 * tile_size_bytes;

    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    auto distributed_buffer_config = ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape};

    // Allocate a distributed buffer in L1 memory, spanning devices in the mesh.
    auto mesh_buffer = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Enqueue a write to the distributed buffer (L1 banks across devices) with random data.
    std::vector<uint32_t> src_data = create_random_vector_of_bfloat16(
        distributed_buffer_size_bytes, 1, std::chrono::system_clock::now().time_since_epoch().count());
    EnqueueWriteMeshBuffer(cq, mesh_buffer, src_data);

    // Enqueue a read from the distributed buffer (L1 banks across devices) to a local buffer.
    std::vector<uint32_t> read_back_data{};
    EnqueueReadMeshBuffer(cq, read_back_data, mesh_buffer, true /* blocking */);

    // Data read back across all devices in the mesh should match the original data.
    assert(src_data == read_back_data);

    return 0;
}
