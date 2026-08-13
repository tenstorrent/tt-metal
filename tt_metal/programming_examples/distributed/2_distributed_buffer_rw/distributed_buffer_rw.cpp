// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>

// Stand-alone example demonstrating usage of native multi-device TT-Metalium APIs
// for issuing Read and Write commands to a MeshBuffer spanning multiple devices.
//
// Cross-device placement is owned by the caller via DistributedHostBuffer: each
// device gets its own host shard, which MeshCommandQueue writes/reads independently.
// MeshBuffer itself is allocated with ReplicatedBufferConfig (same local size on
// every device).
//
// The example demonstrates how to:
// 1. Lock-step allocate an L1 MeshBuffer across a mesh of devices
// 2. Build a DistributedHostBuffer with distinct data on each device
// 3. Enqueue a write of those host shards to the MeshBuffer
// 4. Enqueue a read back into a DistributedHostBuffer
// 5. Verify that each device's data matches what was written
int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& cq = mesh_device->mesh_command_queue();

    // Each device holds a 32x32-tile shard. The caller, not MeshBuffer, decides
    // what data lives on which device.
    constexpr uint32_t shard_height = 32;
    constexpr uint32_t shard_width = 32;
    const uint32_t tile_size_bytes = tt::tile_size(tt::DataFormat::UInt32);
    const uint32_t per_device_size_bytes = shard_height * shard_width * tile_size_bytes;
    const uint32_t num_uint32s = per_device_size_bytes / sizeof(uint32_t);

    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::L1, .bottom_up = false};
    auto mesh_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{.size = per_device_size_bytes}, local_buffer_config, mesh_device.get());

    auto src = DistributedHostBuffer::create(mesh_device->shape());
    uint32_t seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        const uint32_t shard_seed = seed++;
        src.emplace_shard(coord, [per_device_size_bytes, shard_seed]() {
            return HostBuffer(create_random_vector_of_bfloat16(per_device_size_bytes, 1, shard_seed));
        });
    }

    cq.enqueue_write(mesh_buffer, src, /*blocking=*/false);

    auto dst = DistributedHostBuffer::create(mesh_device->shape());
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        dst.emplace_shard(coord, [num_uint32s]() { return HostBuffer(std::vector<uint32_t>(num_uint32s, 0)); });
    }
    cq.enqueue_read(mesh_buffer, dst, /*shards=*/std::nullopt, /*blocking=*/true);

    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        const auto src_shard = src.get_shard(coord);
        const auto dst_shard = dst.get_shard(coord);
        assert(src_shard.has_value() && dst_shard.has_value());
        assert(*src_shard == *dst_shard);
    }

    return 0;
}
