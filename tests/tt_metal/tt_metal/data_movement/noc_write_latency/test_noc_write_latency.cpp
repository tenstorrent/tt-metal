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
    // When true, dst_core is ignored and the bottom-right compute core is used instead.
    bool dst_far_corner = false;
    // When true, every core on the X-first route (except sender and dst) blasts posted writes
    // toward the dst to saturate the path and measure latency under congestion.
    bool enable_congestion = false;
    uint32_t num_iterations = 100;
    uint32_t transaction_size_bytes = 32;
    uint32_t noise_writes_per_core = 100000;
};

bool run_noc_write_latency(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocWriteLatencyConfig& cfg) {
    IDevice* device = mesh_device->get_device(0);

    const ARCH arch = MetalContext::instance().get_cluster().arch();
    const bool is_quasar = (arch == ARCH::QUASAR);
    if (arch != ARCH::QUASAR && arch != ARCH::BLACKHOLE) {
        log_info(LogTest, "Skipping: unsupported arch (only Quasar and Blackhole)");
        return true;
    }
    auto grid = device->compute_with_storage_grid_size();
    log_info(LogTest, "Compute grid: {}x{}", grid.x, grid.y);

    // The bottom-right compute core is (grid.x - 1, grid.y - 1).
    CoreCoord dst_core = cfg.dst_far_corner ? CoreCoord{grid.x - 1, grid.y - 1} : cfg.dst_core;

    if (cfg.src_core.x >= grid.x || cfg.src_core.y >= grid.y || dst_core.x >= grid.x || dst_core.y >= grid.y) {
        log_info(
            LogTest,
            "Skipping: src ({},{}) or dst ({},{}) outside compute grid {}x{}",
            cfg.src_core.x,
            cfg.src_core.y,
            dst_core.x,
            dst_core.y,
            grid.x,
            grid.y);
        return true;
    }
    log_info(LogTest, "src ({},{}) -> dst ({},{})", cfg.src_core.x, cfg.src_core.y, dst_core.x, dst_core.y);

    CoreCoord phys_src = device->worker_core_from_logical_core(cfg.src_core);
    CoreCoord phys_dst = device->worker_core_from_logical_core(dst_core);

    L1AddressInfo src_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.src_core);
    uint32_t src_l1_addr = src_l1.base_address;

    L1AddressInfo dst_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, dst_core);
    uint32_t dst_l1_data_addr = dst_l1.base_address;

    if (src_l1.size < cfg.transaction_size_bytes || dst_l1.size < cfg.transaction_size_bytes) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // The payload's first word doubles as the receiver's poll signal; zero it before starting.
    vector<uint32_t> zero{0};
    detail::WriteToDeviceL1(device, dst_core, dst_l1_data_addr, zero);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    Program program = CreateProgram();

    // Quasar and Blackhole use different kernel-creation APIs and different kernel sources
    // (the _bh variants read the wall-clock register and skip the uncached-L1 alias). The
    // compile-arg layouts are identical, so only the create call and source path differ.
    const std::string kdir = "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/";
    const std::string sender_kernel = kdir + (is_quasar ? "sender.cpp" : "sender_bh.cpp");
    const std::string receiver_kernel = kdir + (is_quasar ? "receiver.cpp" : "receiver_bh.cpp");

    auto create_dm_kernel = [&](const std::string& path, const CoreCoord& core, const std::vector<uint32_t>& args) {
        if (is_quasar) {
            experimental::quasar::CreateKernel(
                program,
                path,
                core,
                experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .compile_args = args});
        } else {
            CreateKernel(
                program,
                path,
                core,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = args});
        }
    };

    create_dm_kernel(
        sender_kernel,
        cfg.src_core,
        {
            src_l1_addr,
            dst_l1_data_addr,
            (uint32_t)phys_dst.x,
            (uint32_t)phys_dst.y,
            cfg.num_iterations,
            cfg.transaction_size_bytes,
            (uint32_t)phys_src.x,
            (uint32_t)phys_src.y,
        });

    create_dm_kernel(
        receiver_kernel,
        dst_core,
        {
            dst_l1_data_addr,
            cfg.num_iterations,
            (uint32_t)phys_dst.x,
            (uint32_t)phys_dst.y,
        });

    // Optional background traffic: place noise kernels on every core along the X-first route
    // (row 0 east leg, then dst column south leg), excluding the sender and dst. They blast
    // posted writes to a scratch L1 region on dst, contending for the same NOC links.
    if (cfg.enable_congestion && !is_quasar) {
        log_warning(LogTest, "Congestion mode is only implemented for Quasar; running without noise");
    }
    if (cfg.enable_congestion && is_quasar) {
        const uint32_t noise_dst_addr = dst_l1_data_addr + 1024;  // clear of the measured payload/poll word

        std::vector<CoreCoord> noise_cores;
        for (uint32_t x = 1; x <= dst_core.x; ++x) {
            noise_cores.push_back(CoreCoord{x, 0});  // east leg along row 0, including (dst.x, 0)
        }
        for (uint32_t y = 1; y < dst_core.y; ++y) {
            noise_cores.push_back(CoreCoord{dst_core.x, y});  // south leg down dst column, excluding dst
        }

        for (const auto& nc : noise_cores) {
            L1AddressInfo nc_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, nc);
            experimental::quasar::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/noise.cpp",
                nc,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = 1,
                    .compile_args = {
                        (uint32_t)nc_l1.base_address,
                        noise_dst_addr,
                        (uint32_t)phys_dst.x,
                        (uint32_t)phys_dst.y,
                        cfg.noise_writes_per_core,
                        cfg.transaction_size_bytes,
                    }});
        }
        log_info(LogTest, "Congestion enabled: {} noise cores blasting toward dst", noise_cores.size());
    }

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
        .dst_core = {0, 0},
        .dst_far_corner = true,
        .num_iterations = 10,
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
        .dst_far_corner = false,
        .num_iterations = 10,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, NocWriteLatencyFarCornersCongested) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {0, 0},
        .dst_far_corner = true,
        .enable_congestion = true,
        .num_iterations = 10,
        .transaction_size_bytes = 32,
        .noise_writes_per_core = 100000,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

TEST_F(BlackholeSingleCardFixture, NocWriteLatencyFarCornersBH) {
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {0, 0},
        .dst_far_corner = true,
        .num_iterations = 10,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

TEST_F(BlackholeSingleCardFixture, NocWriteLatencyAdjacentCoresBH) {
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {1, 0},
        .dst_far_corner = false,
        .num_iterations = 10,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->devices_[0], cfg));
}

}  // namespace tt::tt_metal
