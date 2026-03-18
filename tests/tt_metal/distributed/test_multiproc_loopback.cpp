// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// MPI-based cross-process loopback socket test.
// Rank 0 (launcher): opens device, creates H2D+D2H sockets, launches loopback kernel.
// Rank 1 (connector): connects to sockets via PCIeCoreWriter (no MetalContext), drives data.
//
// Run with: mpirun -np 2 ./multiproc_loopback_test
//

#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "impl/context/metal_context.hpp"

static int g_world_rank = -1;
static int g_world_size = -1;
static int test_counter = 0;

namespace tt::tt_metal::distributed {
namespace {

using namespace tt::tt_metal::distributed::multihost;

constexpr uint32_t NUM_ITERATIONS = 1000;

struct LoopbackConfig {
    std::size_t fifo_size;
    std::size_t page_size;
    std::size_t data_size;
};

constexpr LoopbackConfig loopback_configs[] = {
    {1024, 64, 1024},       // No wrap
    {1024, 64, 32768},      // Even wrap
    {4096, 1088, 78336},    // Uneven wrap
    {16512, 1088, 156672},  // Uneven wrap, multiple host pages
};

const MeshCoreCoord SOCKET_CORE = {MeshCoordinate(0, 0), CoreCoord(0, 0)};

void run_launcher(const std::shared_ptr<MeshDevice>& mesh_device, H2DMode h2d_mode, const LoopbackConfig& cfg) {
    const auto& socket_core = SOCKET_CORE;
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", test_counter);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", test_counter);

    auto h2d_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, cfg.fifo_size, h2d_mode);
    h2d_socket.export_descriptor(h2d_socket_id);

    auto d2h_socket = D2HSocket(mesh_device, socket_core, cfg.fifo_size);
    d2h_socket.export_descriptor(d2h_socket_id);

    auto program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(h2d_socket.get_config_buffer_address()),
                static_cast<uint32_t>(d2h_socket.get_config_buffer_address()),
                static_cast<uint32_t>(cfg.page_size),
                static_cast<uint32_t>(cfg.data_size),
                static_cast<uint32_t>(NUM_ITERATIONS),
                static_cast<uint32_t>(h2d_mode == H2DMode::DEVICE_PULL),
            }});

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device->mesh_command_queue());
}

void run_connector(const LoopbackConfig& cfg) {
    std::string h2d_socket_id = fmt::format("test_h2d_xproc_{}", test_counter);
    std::string d2h_socket_id = fmt::format("test_d2h_xproc_{}", test_counter);

    auto h2d_socket = H2DSocket::connect(h2d_socket_id, 30000);
    auto d2h_socket = D2HSocket::connect(d2h_socket_id, 30000);

    h2d_socket->set_page_size(cfg.page_size);
    d2h_socket->set_page_size(cfg.page_size);

    uint32_t page_size_words = cfg.page_size / sizeof(uint32_t);
    uint32_t data_size_words = cfg.data_size / sizeof(uint32_t);
    uint32_t num_txns = cfg.data_size / cfg.page_size;

    std::vector<uint32_t> src_vec(data_size_words * NUM_ITERATIONS);
    std::vector<uint32_t> dst_vec(data_size_words * NUM_ITERATIONS, 0);
    std::iota(src_vec.begin(), src_vec.end(), 0);

    std::thread write_thread([&]() {
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                h2d_socket->write(src_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    std::thread read_thread([&]() {
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                d2h_socket->read(dst_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    write_thread.join();
    read_thread.join();

    h2d_socket->barrier();
    d2h_socket->barrier();

    EXPECT_EQ(src_vec, dst_vec) << "Loopback verification FAILED (fifo=" << cfg.fifo_size << " page=" << cfg.page_size
                                << " data=" << cfg.data_size << ")";
}

class MultiProcLoopbackFixture : public MeshDeviceFixtureBase {
protected:
    MultiProcLoopbackFixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 1}}) {}

    void SetUp() override {
        ASSERT_EQ(g_world_size, 2) << "This test requires exactly 2 MPI ranks";
        rank_ = g_world_rank;
        if (rank_ == 0) {
            MeshDeviceFixtureBase::SetUp();
        }
    }

    void TearDown() override {
        if (rank_ == 0) {
            MeshDeviceFixtureBase::TearDown();
        }
    }

    int rank_ = -1;
};

TEST_F(MultiProcLoopbackFixture, CrossProcessLoopback) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
    }

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        for (const auto& cfg : loopback_configs) {
            if (rank_ == 0) {
                run_launcher(mesh_device_, h2d_mode, cfg);
            } else {
                run_connector(cfg);
            }
            test_counter++;
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed

int main(int argc, char** argv) {
    using namespace tt::tt_metal::distributed::multihost;

    DistributedContext::create(argc, argv);
    const auto& world = DistributedContext::get_current_world();
    g_world_rank = *world->rank();
    g_world_size = *world->size();

    auto local_ctx = world->split(Color(g_world_rank), Key(0));
    DistributedContext::set_current_world(local_ctx);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
