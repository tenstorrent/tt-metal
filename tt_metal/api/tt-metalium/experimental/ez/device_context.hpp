// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal::experimental::ez {

// Configuration for creating a sharded L1 buffer.
struct ShardConfig {
    CoreRangeSet cores;
    std::array<uint32_t, 2> shard_shape;  // {height, width} in elements
    std::array<uint32_t, 2> tensor2d_shape_in_pages;  // full tensor shape in tiles {n_tiles_y, n_tiles_x}
    TensorMemoryLayout layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    std::array<uint32_t, 2> page_shape = {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    tt::DataFormat fmt = tt::DataFormat::Float16_b;
};

// RAII wrapper around MeshDevice that simplifies common device operations.
// Provides one-line buffer creation, data transfer, and program execution.
class DeviceContext {
public:
    // Create a single-device context on the given device ID.
    explicit DeviceContext(int device_id = 0);

    // Create a multi-device mesh context.
    DeviceContext(size_t rows, size_t cols);

    // Wrap an existing MeshDevice without taking ownership (device is NOT closed on destruction).
    explicit DeviceContext(std::shared_ptr<distributed::MeshDevice> device);

    ~DeviceContext();

    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    // Buffer creation helpers.
    std::shared_ptr<distributed::MeshBuffer> dram_buffer(DeviceAddr size, DeviceAddr page_size) const;
    std::shared_ptr<distributed::MeshBuffer> l1_buffer(DeviceAddr size, DeviceAddr page_size) const;
    std::shared_ptr<distributed::MeshBuffer> dram_tile_buffer(
        uint32_t n_tiles, tt::DataFormat fmt = tt::DataFormat::Float16_b) const;
    std::shared_ptr<distributed::MeshBuffer> sharded_l1_buffer(const ShardConfig& config) const;

    // Data transfer: write host data to a mesh buffer.
    template <typename T>
    void write(const std::shared_ptr<distributed::MeshBuffer>& buf, const std::vector<T>& data, bool blocking = false);

    // Data transfer: read mesh buffer contents back to host.
    template <typename T>
    std::vector<T> read(const std::shared_ptr<distributed::MeshBuffer>& buf, bool blocking = true);

    // Execute a program synchronously (enqueue + finish).
    void run(Program&& program);

    // Enqueue a program without waiting for completion.
    void launch(Program&& program);

    // Block until all enqueued work completes.
    void finish();

    // Convert a logical core coordinate to a physical NoC coordinate.
    CoreCoord physical_core(const CoreCoord& logical) const;

    // Escape hatches to the underlying API.
    distributed::MeshDevice& device();
    const distributed::MeshDevice& device() const;
    distributed::MeshCommandQueue& cq();
    const distributed::MeshCommandQueue& cq() const;
    distributed::MeshCoordinateRange full_range() const;

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    bool owns_device_ = true;
};

// Template implementations.
template <typename T>
void DeviceContext::write(
    const std::shared_ptr<distributed::MeshBuffer>& buf, const std::vector<T>& data, bool blocking) {
    cq().enqueue_write_mesh_buffer(buf, data.data(), blocking);
}

template <typename T>
std::vector<T> DeviceContext::read(const std::shared_ptr<distributed::MeshBuffer>& buf, bool blocking) {
    std::vector<T> result;
    if (buf->global_layout() == distributed::MeshBufferLayout::SHARDED) {
        result.resize(buf->global_shard_spec().global_size / sizeof(T));
    } else {
        result.resize(buf->size() / sizeof(T));
    }
    cq().enqueue_read_mesh_buffer(result.data(), buf, blocking);
    return result;
}

}  // namespace tt::tt_metal::experimental::ez
