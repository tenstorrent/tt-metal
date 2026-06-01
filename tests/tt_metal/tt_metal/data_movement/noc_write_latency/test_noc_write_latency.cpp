// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/host_api/temp_quasar_api.hpp"

namespace tt::tt_metal {

using namespace std;

namespace unit_tests::dm::noc_write_latency {

struct NocWriteLatencyConfig {
    CoreCoord src_core;
    CoreCoord dst_core;
    uint32_t num_iterations = 100;
    uint32_t transaction_size_bytes = 32;
};

bool run_noc_write_latency(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocWriteLatencyConfig& cfg) {
    IDevice* device = mesh_device->get_device(0);

    if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
        log_info(LogTest, "Skipping: not a Quasar device");
        return true;
    }
    auto grid = device->compute_with_storage_grid_size();
    if (grid.x < 9 || grid.y < 4) {
        log_info(LogTest, "Skipping: grid {}x{} smaller than 9x4", grid.x, grid.y);
        return true;
    }

    CoreCoord phys_dst = device->worker_core_from_logical_core(cfg.dst_core);

    L1AddressInfo src_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.src_core);
    uint32_t src_l1_addr = src_l1.base_address;
    uint32_t flag_local_addr = src_l1.base_address + 64;

    L1AddressInfo dst_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.dst_core);
    uint32_t dst_l1_data_addr = dst_l1.base_address;
    uint32_t dst_l1_flag_addr = dst_l1.base_address + 64;

    if (src_l1.size < cfg.transaction_size_bytes || dst_l1.size < cfg.transaction_size_bytes) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    TT_FATAL(
        cfg.transaction_size_bytes <= 64,
        "transaction_size_bytes {} exceeds 64 bytes; data and flag buffers would overlap",
        cfg.transaction_size_bytes);

    vector<uint32_t> zero{0};
    detail::WriteToDeviceL1(device, cfg.dst_core, dst_l1_flag_addr, zero);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp",
        cfg.src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {
                src_l1_addr,
                flag_local_addr,
                dst_l1_data_addr,
                dst_l1_flag_addr,
                (uint32_t)phys_dst.x,
                (uint32_t)phys_dst.y,
                cfg.num_iterations,
                cfg.transaction_size_bytes,
            }});

    experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp",
        cfg.dst_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {
                dst_l1_flag_addr,
                cfg.num_iterations,
                (uint32_t)phys_dst.x,
                (uint32_t)phys_dst.y,
            }});

    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    return true;
}

}  // namespace unit_tests::dm::noc_write_latency

TEST_F(QuasarMeshDeviceSingleCardFixture, NocWriteLatencyFarCorners) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {8, 3},
        .num_iterations = 100,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, NocWriteLatencyAdjacentCores) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {1, 0},
        .num_iterations = 100,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

}  // namespace tt::tt_metal
