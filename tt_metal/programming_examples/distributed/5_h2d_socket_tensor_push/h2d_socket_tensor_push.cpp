// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Push a tensor from host to device using the same high-level socket API as device-device
// (issue #34274, PR #36909): config + create + write/barrier/readback. No kernel creation
// or custom session types in user code.
//
// Flow: SocketMemoryConfig + recv core → create H2D receiver socket → write() → barrier()
// → get_data_buffer() for readback. Requires vIOMMU. Run from repo root for kernel paths.

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_align.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace tt::tt_metal::distributed {

// High-level H2D API (same model as MeshSocket: config + create, then ops).
// When the runtime provides this (PR #36909 / issue #34274), use the library instead;
// the implementation below is a stand-in that uses H2DSocket + standard receiver kernel.
struct H2DReceiverSocket {
    void write(void* data, uint32_t num_pages);
    void barrier();
    std::shared_ptr<MeshBuffer> get_data_buffer() const { return recv_buffer_; }

    static std::shared_ptr<H2DReceiverSocket> create(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        const SocketMemoryConfig& socket_mem_config,
        uint32_t page_size,
        uint32_t data_size) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
            return nullptr;
        }
        TT_FATAL(data_size % page_size == 0, "data_size must be a multiple of page_size");
        auto socket = std::shared_ptr<H2DReceiverSocket>(new H2DReceiverSocket());
        socket->init(mesh_device, recv_core, socket_mem_config, page_size, data_size);
        return socket;
    }

private:
    H2DReceiverSocket() = default;
    void init(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        const SocketMemoryConfig& socket_mem_config,
        uint32_t page_size,
        uint32_t data_size);

    std::shared_ptr<MeshDevice> mesh_device_;
    MeshCoreCoord recv_core_;
    uint32_t page_size_ = 0;
    uint32_t data_size_ = 0;
    std::unique_ptr<H2DSocket> socket_;
    std::shared_ptr<MeshBuffer> recv_buffer_;
};

void H2DReceiverSocket::init(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    const SocketMemoryConfig& socket_mem_config,
    uint32_t page_size,
    uint32_t data_size) {
    mesh_device_ = mesh_device;
    recv_core_ = recv_core;
    page_size_ = page_size;
    data_size_ = data_size;

    socket_ = std::make_unique<H2DSocket>(
        mesh_device_, recv_core_, socket_mem_config.socket_storage_type,
        socket_mem_config.fifo_size, H2DMode::HOST_PUSH);
    socket_->set_page_size(page_size);

    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto recv_data_shard_params = ShardSpecBuffer(
        CoreRangeSet(recv_core_.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    recv_buffer_ = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device_.get());

    constexpr uint32_t num_iterations = 1;
    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_core_.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(socket_->get_config_buffer_address()),
                static_cast<uint32_t>(recv_buffer_->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
            }});

    MeshWorkload mesh_workload;
    mesh_workload.add_program(MeshCoordinateRange(recv_core_.device_coord), std::move(recv_program));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());
}

void H2DReceiverSocket::write(void* data, uint32_t num_pages) {
    const uint32_t page_size_words = page_size_ / sizeof(uint32_t);
    auto* ptr = static_cast<uint32_t*>(data);
    for (uint32_t p = 0; p < num_pages; p++) {
        socket_->write(ptr + p * page_size_words, 1);
    }
}

void H2DReceiverSocket::barrier() {
    socket_->barrier();
}

}  // namespace tt::tt_metal::distributed

int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const MeshShape mesh_shape(1, 1);
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(mesh_shape));

    // Same config style as device-device sockets (SocketMemoryConfig + receiver core).
    const MeshCoreCoord recv_core(MeshCoordinate(0, 0), CoreCoord(0, 0));
    const SocketMemoryConfig socket_mem_config(BufferType::L1, 1024);
    constexpr uint32_t page_size = 64;
    constexpr uint32_t data_size = 1024;

    auto socket = H2DReceiverSocket::create(
        mesh_device, recv_core, socket_mem_config, page_size, data_size);
    if (!socket) {
        std::cerr << "H2D socket requires NOC-mappable host memory (vIOMMU). Skipping.\n";
        return 0;
    }

    const uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0u);

    socket->write(src_vec.data(), num_pages);
    socket->barrier();

    // Readback via get_data_buffer() (same as MeshSocket::get_data_buffer() on receiver).
    std::vector<uint32_t> readback(data_size / sizeof(uint32_t));
    auto recv_core_virtual = mesh_device->worker_core_from_logical_core(recv_core.core_coord);
    MetalContext::instance().get_cluster().read_core(
        readback.data(),
        data_size,
        tt_cxy_pair(mesh_device->get_device(recv_core.device_coord)->id(), recv_core_virtual),
        socket->get_data_buffer()->address());

    if (readback != src_vec) {
        std::cerr << "Mismatch: tensor read back from device does not match sent data.\n";
        return 1;
    }
    std::cout << "OK: tensor pushed via high-level H2D socket API and verified by readback.\n";
    return 0;
}
