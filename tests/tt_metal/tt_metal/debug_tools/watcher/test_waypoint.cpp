// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
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
#include <tt-metalium/experimental/host_api.hpp>

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
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;
    // TENSIX/ACTIVE_ETH cores: SD only used for Quasar watcher tests (match test_assert, test_sanitize)
    if (fixture->IsSlowDispatch() && !is_quasar && device->get_inactive_ethernet_cores().empty()) {
        GTEST_SKIP() << "Slow Dispatch tests only run on Quasar or IDLE_ETH cores";
    }
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp";
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = is_quasar ? CoreCoord{0, 0} : CoreCoord{4, 4};

    // Allocate and zero-init L1 sync flag for host-device handshake
    uint32_t tensix_sync_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t idle_eth_sync_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    std::vector<uint32_t> zero_data = {0};

    // Zero-init sync flag on all tensix cores
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            tt::tt_metal::detail::WriteToDeviceL1(device, CoreCoord{x, y}, tensix_sync_addr, zero_data);
        }
    }

    // Runtime args differ by core type:
    // - TENSIX / idle ERISC: tensix_sync_addr/idle_eth_sync_addr (arg 0) - blocks until host writes 1
    // - Active ERISC: delay_cycles (arg 0) - timed wait, can't block due to tunneling
    const std::vector<uint32_t> tensix_args = {tensix_sync_addr};
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 3000000;  // 3 seconds - enough for watcher to capture waypoint
    const std::vector<uint32_t> active_eth_args = {delay_cycles};

    // Create kernels for TENSIX cores
    if (is_quasar) {
        auto num_dms = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        auto dm_kid = tt::tt_metal::experimental::quasar::CreateKernel(
            program_,
            kernel_path,
            CoreRange(xy_start, xy_end),
            tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = num_dms});
        auto compute_kid = tt::tt_metal::experimental::quasar::CreateKernel(
            program_,
            kernel_path,
            CoreRange(xy_start, xy_end),
            tt::tt_metal::experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 4});
        SetCommonRuntimeArgs(program_, dm_kid, tensix_args);
        SetCommonRuntimeArgs(program_, compute_kid, tensix_args);
    } else {
        auto brisc_kid = CreateKernel(
            program_,
            kernel_path,
            CoreRange(xy_start, xy_end),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto ncrisc_kid = CreateKernel(
            program_,
            kernel_path,
            CoreRange(xy_start, xy_end),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto trisc_kid = CreateKernel(program_, kernel_path, CoreRange(xy_start, xy_end), ComputeConfig{});
        SetCommonRuntimeArgs(program_, brisc_kid, tensix_args);
        SetCommonRuntimeArgs(program_, ncrisc_kid, tensix_args);
        SetCommonRuntimeArgs(program_, trisc_kid, tensix_args);
    }

    // Create kernels for ethernet cores
    bool has_eth_cores = !device->get_active_ethernet_cores(true).empty();
    bool has_idle_eth_cores = fixture->IsSlowDispatch() && !device->get_inactive_ethernet_cores().empty();

    if (has_eth_cores) {
        std::set<CoreRange> ranges;
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            ranges.insert(CoreRange(core, core));
        }
        // Active ERISC: pass delay_cycles for timed wait (can't block forever due to tunneling)
        auto kid = CreateKernel(program_, kernel_path, ranges, tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});
        SetCommonRuntimeArgs(program_, kid, active_eth_args);
    }

    if (has_idle_eth_cores) {
        std::set<CoreRange> ranges;
        const std::vector<uint32_t> idle_eth_args = {idle_eth_sync_addr};
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            ranges.insert(CoreRange(core, core));
            tt::tt_metal::detail::WriteToDeviceL1(device, core, idle_eth_sync_addr, zero_data, CoreType::ETH);
        }
        // Create kernel for each idle ETH processor (use HAL to get count)
        uint32_t num_idle_eth_processors = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);
        for (uint32_t proc_id = 0; proc_id < num_idle_eth_processors; proc_id++) {
            auto processor = (proc_id == 0) ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1;
            auto kid = CreateKernel(
                program_,
                kernel_path,
                ranges,
                tt_metal::EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = processor});
            SetCommonRuntimeArgs(program_, kid, idle_eth_args);
        }
    }

    // Dispatch non-blocking: kernels post waypoint then spin on sync flag
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    // Build poll patterns and wait for waypoints to appear
    std::vector<std::string> poll_strings;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            poll_strings.push_back(fmt::format("worker core(x={:2},y={:2})*: {}", x, y, waypoint));
        }
    }
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            poll_strings.push_back(fmt::format("acteth core(x={:2},y={:2})*: {}", core.x, core.y, waypoint));
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            poll_strings.push_back(fmt::format("idleth core(x={:2},y={:2})*: {}", core.x, core.y, waypoint));
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
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, idle_eth_sync_addr, release_data, CoreType::ETH);
        }
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    // Get processor counts from HAL and build expected waypoint strings
    uint32_t num_tensix = hal.get_num_risc_processors(HalProgrammableCoreType::TENSIX);
    uint32_t num_idle_eth = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);
    std::string tensix_waypoints = build_waypoint_string(num_tensix);
    std::string idle_eth_waypoints = build_waypoint_string(num_idle_eth);

    log_info(tt::LogTest, "Verifying: {} waypoints/TENSIX, 1/active ETH, {}/idle ETH", num_tensix, num_idle_eth);

    // Verify waypoints with device ID and virtual coordinates
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
        bool found = FileContainsAllStringsInOrder(fixture->log_file_name, {expected});
        EXPECT_TRUE(found) << "Missing waypoint log for " << type << " core (" << logical_core.x << ","
                           << logical_core.y << ")\n"
                           << "Expected: " << expected;
    };
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            CoreCoord logical_core = {x, y};
            CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
            check_core(logical_core, virtual_core, "worker", tensix_waypoints);
        }
    }
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, "acteth", waypoint);
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, "idleth", idle_eth_waypoints);
        }
    }
}
}
}

TEST_F(MeshWatcherFixture, TestWatcherWaypoints) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, mesh_device);
    }
}
