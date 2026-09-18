// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Same body and kernels over a real D2D MeshSocket and over a HostMeshSocket.
// Only the socket type and SOCKET_MODE differ; if both pass, the host socket is
// a drop-in at the call site.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/host_mesh_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>

#include "host_socket_test_utils.hpp"

#include <numeric>
#include <vector>

namespace tt::tt_metal::distributed::host_socket_test {
namespace {

constexpr uint32_t kPageSize = 2048;
constexpr uint32_t kNumPages = 16;
constexpr uint32_t kFifoSize = kPageSize * kNumPages;
constexpr uint32_t kDataSize = kFifoSize;

enum class Role { Both, Sender, Receiver };

void run_case(
    const std::shared_ptr<MeshDevice>& device,
    const CoreCoord& sender_core,
    const CoreCoord& recv_core,
    uint32_t sender_config_addr,
    uint32_t recv_config_addr,
    uint32_t mode,
    Role role) {
    const MeshCoordinate coord(0, 0);
    auto src = make_core_l1_buffer(device.get(), sender_core, kDataSize);
    auto dst = make_core_l1_buffer(device.get(), recv_core, kDataSize);

    std::vector<uint32_t> payload(kDataSize / sizeof(uint32_t));
    std::iota(payload.begin(), payload.end(), 0x5a5a0000u);
    std::vector<uint32_t> inverted(payload.size());
    for (size_t i = 0; i < payload.size(); i++) {
        inverted[i] = ~payload[i];
    }
    WriteShard(device->mesh_command_queue(), src, payload, coord, true);
    WriteShard(device->mesh_command_queue(), dst, inverted, coord, true);

    const std::vector<uint32_t> sender_args{
        sender_config_addr, static_cast<uint32_t>(src->address()), kPageSize, kDataSize, mode, kDataSize};
    const std::vector<uint32_t> recv_args{
        recv_config_addr, static_cast<uint32_t>(dst->address()), kPageSize, kDataSize, mode, kDataSize};

    auto program = CreateProgram();
    if (role == Role::Both || role == Role::Sender) {
        CreateKernel(
            program,
            kSenderKernel,
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = sender_args});
    }
    if (role == Role::Both || role == Role::Receiver) {
        // Sharing a program: receiver takes the second DM processor.
        const bool second = role == Role::Both;
        CreateKernel(
            program,
            kReceiverKernel,
            recv_core,
            DataMovementConfig{
                .processor = second ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                .noc = second ? NOC::RISCV_1_default : NOC::RISCV_0_default,
                .compile_args = recv_args});
    }

    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(coord), std::move(program));
    EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
    Finish(device->mesh_command_queue());

    // Only the end that actually lands the data can check it.
    if (role == Role::Both || role == Role::Receiver) {
        std::vector<uint32_t> got(payload.size());
        ReadShard(device->mesh_command_queue(), got, dst, coord, true);
        ASSERT_EQ(got, payload);
    }
}

// Baseline: the same kernels over a real D2D MeshSocket, sender and receiver on
// two cores of one device (NOC transport, no fabric needed). Each rank runs this
// independently against its own device.
TEST(SocketHotSwapTest, D2DMeshSocket) {
    Params params = params_from_env();
    auto device = MeshDevice::create_unit_mesh(static_cast<int>(params.device_id));

    const CoreCoord sender_core(0, 0);
    const CoreCoord recv_core(1, 0);
    const MeshCoordinate coord(0, 0);

    SocketConnection connection{MeshCoreCoord(coord, sender_core), MeshCoreCoord(coord, recv_core)};
    SocketMemoryConfig mem_config(BufferType::L1, kFifoSize);
    SocketConfig socket_config({connection}, mem_config);
    auto [send_socket, recv_socket] = MeshSocket::create_socket_pair(device, device, socket_config);

    run_case(
        device,
        sender_core,
        recv_core,
        static_cast<uint32_t>(send_socket.get_config_buffer_address()),
        static_cast<uint32_t>(recv_socket.get_config_buffer_address()),
        kModeD2D,
        Role::Both);
}

// The swap: identical body and kernels, a HostMeshSocket instead, and
// SOCKET_MODE flipped to the host transport. Two ranks, one endpoint each.
TEST(SocketHotSwapTest, HostMeshSocketDropIn) {
    const auto context = multihost::DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2) << "needs exactly 2 ranks";

    Params params = params_from_env();
    auto device = MeshDevice::create_unit_mesh(static_cast<int>(params.device_id));
    if (!host_sockets_supported(device)) {
        GTEST_SKIP() << "pinned host memory cannot be mapped to the NOC; enable vIOMMU";
    }

    const CoreCoord sender_core(0, 0);
    const CoreCoord recv_core(0, 0);
    const MeshCoordinate coord(0, 0);

    // The very same SocketConfig shape a rank-scoped MeshSocket takes.
    SocketConnection connection{MeshCoreCoord(coord, sender_core), MeshCoreCoord(coord, recv_core)};
    SocketMemoryConfig mem_config(BufferType::L1, kFifoSize);
    SocketConfig socket_config({connection}, mem_config, kSenderRank, kReceiverRank, context);

    HostMeshSocket::TransportConfig transport;
    transport.page_size = kPageSize;
    HostMeshSocket socket(device, socket_config, transport);

    const bool is_sender = context->rank() == kSenderRank;

    // Parity with MeshSocket: a receiver hands out a real fifo_size buffer, a
    // sender raises. A caller branching between the two socket types compiles
    // and behaves the same either way.
    EXPECT_TRUE(socket.is_rank_scoped_socket());
    if (is_sender) {
        EXPECT_ANY_THROW(socket.get_data_buffer());
    } else {
        auto data_buffer = socket.get_data_buffer();
        ASSERT_NE(data_buffer, nullptr);
        // One fifo_size shard per data core, as D2D allocates it.
        EXPECT_GE(data_buffer->device_local_size(), kFifoSize);
        EXPECT_EQ(data_buffer->device_local_size() % kFifoSize, 0u);
    }

    const uint32_t config_addr = static_cast<uint32_t>(socket.get_config_buffer_address());
    run_case(
        device,
        sender_core,
        recv_core,
        config_addr,
        config_addr,
        kModeHostTransport,
        is_sender ? Role::Sender : Role::Receiver);
    socket.barrier();
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_socket_test
