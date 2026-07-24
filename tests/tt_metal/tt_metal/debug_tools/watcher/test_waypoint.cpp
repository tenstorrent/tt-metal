// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.hpp>
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher waypoints.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

const std::string waypoint = "AAAA";

// Build a comma-separated waypoint string for n processors (e.g., "AAAA,AAAA,AAAA")
std::string build_waypoint_string(uint32_t n) {
    if (n == 0) {
        return "";
    }
    std::string result = waypoint;
    for (uint32_t i = 1; i < n; i++) {
        result += "," + waypoint;
    }
    return result;
}

void RunTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;
    // Watcher waypoint kernels use Metal 2.0 on all architectures.
    const std::string kernel_path_metal2 = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints_2_0.cpp";
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = is_quasar ? CoreCoord{0, 0} : CoreCoord{4, 4};

    // Allocate and zero-init L1 sync flag for host-device handshake
    uint32_t tensix_sync_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::vector<uint32_t> zero_data = {0};

    // Zero-init sync flag on all tensix cores
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            tt::tt_metal::detail::WriteToDeviceL1(device, CoreCoord{x, y}, tensix_sync_addr, zero_data);
        }
    }

    std::vector<experimental::KernelSpec> kernel_specs;
    std::vector<experimental::KernelSpecName> kernel_names;
    experimental::ProgramRunArgs params;
    auto add_dm_kernel =
        [&](const char* name, uint32_t num_threads, std::optional<tt::tt_metal::DataMovementProcessor> gen1_processor) {
            auto gen1_proc = gen1_processor.value_or(tt::tt_metal::DataMovementProcessor::RISCV_0);
            auto gen1_noc = (gen1_proc == tt::tt_metal::DataMovementProcessor::RISCV_1)
                                ? tt::tt_metal::NOC::RISCV_1_default
                                : tt::tt_metal::NOC::RISCV_0_default;
            experimental::DataMovementHardwareConfig dm_cfg;
            if (is_quasar) {
                dm_cfg = experimental::DataMovementGen2Config{};
            } else {
                dm_cfg = experimental::DataMovementGen1Config{.processor = gen1_proc, .noc = gen1_noc};
            }
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = experimental::KernelSpecName{name},
                .source = kernel_path_metal2,
                .num_threads = num_threads,
                .runtime_arg_schema = {.common_runtime_arg_names = {"sync_flag_addr"}},
                .hw_config = dm_cfg,
            });
            kernel_names.emplace_back(name);
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = experimental::KernelSpecName{name},
                .common_runtime_arg_values = {{"sync_flag_addr", tensix_sync_addr}},
            });
        };

    const experimental::KernelSpecName compute_kernel_name{"wp_compute"};
    experimental::ComputeHardwareConfig compute_config;
    if (is_quasar) {
        constexpr uint32_t kQuasarUserDmCores = 6;
        add_dm_kernel("wp_dm", kQuasarUserDmCores, std::nullopt);
        compute_config = experimental::ComputeGen2Config{};
    } else {
        add_dm_kernel("wp_brisc", 1, tt::tt_metal::DataMovementProcessor::RISCV_0);
        add_dm_kernel("wp_ncrisc", 1, tt::tt_metal::DataMovementProcessor::RISCV_1);
        compute_config = experimental::ComputeGen1Config{};
    }
    kernel_specs.push_back(experimental::KernelSpec{
        .unique_id = compute_kernel_name,
        .source = kernel_path_metal2,
        .num_threads = is_quasar ? 4u : 1u,
        .runtime_arg_schema = {.common_runtime_arg_names = {"sync_flag_addr"}},
        .hw_config = compute_config,
    });
    kernel_names.emplace_back(compute_kernel_name);
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = compute_kernel_name,
        .common_runtime_arg_values = {{"sync_flag_addr", tensix_sync_addr}},
    });
    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = kernel_names,
        .target_nodes = experimental::NodeRange{CoreRange(xy_start, xy_end)},
    };
    experimental::ProgramSpec spec{
        .name = "watcher_waypoints",
        .kernels = kernel_specs,
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));

    // Dispatch non-blocking: kernels post waypoint then spin on sync flag
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    // Get processor counts from HAL and build expected waypoint strings.
    uint32_t num_tensix = hal.get_num_risc_processors(HalProgrammableCoreType::TENSIX);
    // On Quasar, DM0/DM1 are reserved for internal use and don't post a user waypoint,
    // so the expected per-tensix waypoint count drops by 2.
    if (is_quasar) {
        num_tensix -= 2;
    }
    std::string tensix_waypoints = build_waypoint_string(num_tensix);

    // Build poll patterns and wait for waypoints to appear.
    // On Quasar, slots 0/1 (reserved DM0/DM1) post a non-AAAA status (e.g. " NTW,  W1,") before
    // the first user AAAA. Glob between ": " and the waypoint absorbs that prefix; it matches
    // empty on non-Quasar where AAAA is the very first status field.
    std::vector<std::string> poll_strings;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            poll_strings.push_back(fmt::format("worker core(x={:2},y={:2})*: *{}", x, y, tensix_waypoints));
        }
    }
    log_info(tt::LogTest, "Polling for {} patterns", poll_strings.size());

    constexpr int timeout_ms = 30000;
    auto start = std::chrono::steady_clock::now();
    while (!FileContainsAllStrings(fixture->log_file_name, poll_strings)) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        ASSERT_LT(elapsed, timeout_ms) << "Timed out waiting for watcher to log " << waypoint << " waypoints";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    log_info(tt::LogTest, "All patterns found!");

    // Release all cores
    std::vector<uint32_t> release_data = {1};
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            tt::tt_metal::detail::WriteToDeviceL1(device, CoreCoord{x, y}, tensix_sync_addr, release_data);
        }
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    log_info(tt::LogTest, "Verifying: {} waypoints/TENSIX", num_tensix);

    // Verify waypoints with device ID and virtual coordinates.
    // On Quasar, the first two 4-wide status fields belong to the reserved DM0/DM1 (e.g. " NTW",
    // "  W1") and do not show the AAAA waypoint. Use four '?' wildcards per reserved slot to
    // skip them and anchor the literal AAAA waypoints right after.
    const std::string tensix_status_prefix = is_quasar ? "????,????," : "";
    auto check_core = [&](const CoreCoord& logical_core,
                          const CoreCoord& virtual_core,
                          const std::string& type,
                          const std::string& wp,
                          const std::string& status_prefix) {
        std::string expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {}{}*rmsg:*",
            device->id(),
            type,
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            status_prefix,
            wp);
        bool found = FileContainsAllStringsInOrder(fixture->log_file_name, {expected});
        EXPECT_TRUE(found) << "Missing waypoint log for " << type << " core (" << logical_core.x << ","
                           << logical_core.y << ")\n"
                           << "Expected: " << expected;
    };
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            CoreCoord logical_core = {x, y};
            CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
            check_core(logical_core, virtual_core, "worker", tensix_waypoints, tensix_status_prefix);
        }
    }
}

void RunEthTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = MetalContext::instance().hal();
    if (hal.get_arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Metal 2.0 does not yet support ETH KernelSpecs";
    }
    if (fixture->IsSlowDispatch() && device->get_inactive_ethernet_cores().empty()) {
        GTEST_SKIP() << "Slow Dispatch ETH waypoint tests require IDLE_ETH cores";
    }

    bool has_active_eth_cores = !device->get_active_ethernet_cores(true).empty();
    bool has_idle_eth_cores = fixture->IsSlowDispatch() && !device->get_inactive_ethernet_cores().empty();
    if (!has_active_eth_cores && !has_idle_eth_cores) {
        GTEST_SKIP() << "No supported ETH cores available";
    }

    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp";
    uint32_t idle_eth_sync_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    std::vector<uint32_t> zero_data = {0};
    for (const auto& core : device->get_inactive_ethernet_cores()) {
        tt::tt_metal::detail::WriteToDeviceL1(device, core, idle_eth_sync_addr, zero_data, CoreType::ETH);
    }

    uint32_t clk_mhz = MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    std::vector<uint32_t> active_eth_args = {clk_mhz * 3000000};  // 3 seconds
    std::vector<uint32_t> idle_eth_args = {idle_eth_sync_addr};
    Program program = CreateProgram();

    if (has_active_eth_cores) {
        std::set<CoreRange> ranges;
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            ranges.insert(CoreRange(core, core));
        }
        auto kid = CreateKernel(program, kernel_path, ranges, EthernetConfig{.noc = NOC::NOC_0});
        SetCommonRuntimeArgs(program, kid, active_eth_args);
    }

    if (has_idle_eth_cores) {
        std::set<CoreRange> ranges;
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            ranges.insert(CoreRange(core, core));
        }
        uint32_t num_idle_eth_processors = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);
        for (uint32_t proc_id = 0; proc_id < num_idle_eth_processors; proc_id++) {
            auto processor = proc_id == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1;
            auto kid = CreateKernel(
                program,
                kernel_path,
                ranges,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = processor});
            SetCommonRuntimeArgs(program, kid, idle_eth_args);
        }
    }

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    uint32_t num_idle_eth = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);
    std::string idle_eth_waypoints = build_waypoint_string(num_idle_eth);
    std::vector<std::string> poll_strings;
    if (has_active_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            poll_strings.push_back(fmt::format("acteth core(x={:2},y={:2})*: {}", core.x, core.y, waypoint));
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            poll_strings.push_back(fmt::format("idleth core(x={:2},y={:2})*: {}", core.x, core.y, idle_eth_waypoints));
        }
    }

    constexpr int timeout_ms = 30000;
    auto start = std::chrono::steady_clock::now();
    while (!FileContainsAllStrings(fixture->log_file_name, poll_strings)) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        ASSERT_LT(elapsed, timeout_ms) << "Timed out waiting for watcher to log ETH waypoints";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (has_idle_eth_cores) {
        std::vector<uint32_t> release_data = {1};
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, idle_eth_sync_addr, release_data, CoreType::ETH);
        }
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    auto check_core = [&](const CoreCoord& logical_core,
                          const CoreCoord& virtual_core,
                          const std::string& type,
                          const std::string& wp) {
        std::string expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {}*rmsg:*",
            device->id(),
            type,
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            wp);
        EXPECT_TRUE(FileContainsAllStringsInOrder(fixture->log_file_name, {expected}))
            << "Missing waypoint log for " << type << " core (" << logical_core.x << "," << logical_core.y << ")\n"
            << "Expected: " << expected;
    };
    if (has_active_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            check_core(core, device->ethernet_core_from_logical_core(core), "acteth", waypoint);
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            check_core(core, device->ethernet_core_from_logical_core(core), "idleth", idle_eth_waypoints);
        }
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(MeshWatcherFixture, TestWatcherWaypoints) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherWaypointsEth) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunEthTest, mesh_device);
    }
}
