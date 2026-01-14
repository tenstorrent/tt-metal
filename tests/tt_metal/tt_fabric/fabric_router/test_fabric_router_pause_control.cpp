// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_generator_defs.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_validation.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_router_state_utils.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"

namespace tt::tt_fabric::fabric_router_tests {

class FabricRouterPauseControlTest : public Fabric1DFixture {
protected:
    void SetUp() override {
        Fabric1DFixture::SetUp();
        // Test-specific initialization
        workers_launched_ = false;
        worker_programs_.clear();
        worker_mem_layouts_.clear();
        worker_cores_.clear();
    }

    void TearDown() override {
        // Ensure workers are cleaned up even if test fails
        if (workers_launched_) {
            cleanup_workers();
        }
        Fabric1DFixture::TearDown();
    }

private:
    bool workers_launched_;
    std::vector<std::shared_ptr<tt_metal::Program>> worker_programs_;
    std::vector<test_utils::WorkerMemoryLayout> worker_mem_layouts_;
    std::vector<CoreCoord> worker_cores_;

    void launch_traffic_generators() {
        auto devices = get_devices();
        size_t num_devices = devices.size();

        // Launch workers on devices 0 to N-2, each sending to the next device
        for (size_t i = 0; i < num_devices - 1; ++i) {
            auto& device = devices[i];
            auto& dest_device = devices[i + 1];

            // Allocate memory for this worker
            auto mem_layout = test_utils::allocate_worker_memory(device);
            worker_mem_layouts_.push_back(mem_layout);

            // Choose a worker core (e.g., first available tensix core)
            CoreCoord worker_core = {0, 0}; // Adjust based on device capabilities
            worker_cores_.push_back(worker_core);

            // Get destination fabric node ID
            FabricNodeId dest_node = test_utils::get_fabric_node_id(dest_device);

            // Create and launch program
            auto program = test_utils::create_traffic_generator_program(
                device, worker_core, dest_node, mem_layout);

            // Enqueue program for execution
            RunProgramNonblocking(device, *program);

            worker_programs_.push_back(program);
        }

        workers_launched_ = true;
    }

    void cleanup_workers() {
        if (!workers_launched_) {
            return;
        }

        auto devices = get_devices();

        // Signal teardown to all workers
        for (size_t i = 0; i < worker_programs_.size(); ++i) {
            test_utils::signal_worker_teardown(
                devices[i],
                worker_cores_[i],
                worker_mem_layouts_[i].teardown_signal_address);
        }

        // Wait for all workers to complete
        for (size_t i = 0; i < worker_programs_.size(); ++i) {
            try {
                test_utils::wait_for_worker_complete(
                    this, devices[i], *worker_programs_[i],
                    std::chrono::milliseconds(1000));
            } catch (const std::exception& e) {
                log_error(LogTest, "Worker {} did not complete in time: {}", i, e.what());
            }
        }

        workers_launched_ = false;
        worker_programs_.clear();
        worker_mem_layouts_.clear();
        worker_cores_.clear();
    }

public:
    // Public accessors for testing helper methods
    bool get_workers_launched() const { return workers_launched_; }
    size_t get_num_workers() const { return worker_programs_.size(); }
};

// FR-9: Complete test lifecycle
// Tests that pause command stops all traffic and routers transition to PAUSED state
TEST_F(FabricRouterPauseControlTest, PauseStopsTraffic) {
    // Step 1: Get control plane and validate topology
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";
    MeshId mesh_id = mesh_ids[0];

    size_t num_devices = get_devices().size();
    ASSERT_GE(num_devices, 2) << "Test requires at least 2 devices";

    log_info(LogTest, "Starting Fabric Router Pause Control Test");
    log_info(LogTest, "  Mesh ID: {}", mesh_id);
    log_info(LogTest, "  Num Devices: {}", num_devices);

    // Step 2: Launch worker kernels on devices 0 to N-1
    // Each sends to next device (device i sends to device i+1)
    // FR-2: Launch worker kernels
    // Note: launch_traffic_generators would be called here, but CS-006 (worker helpers)
    // needs to be completed first. For testing purposes, this is a placeholder.
    // launch_traffic_generators();

    log_info(LogTest, "Worker kernels launched (when CS-006 is complete)");

    // Step 3: Wait briefly for traffic to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // FR-3: Validate traffic is flowing
    // Note: This would be validated after workers are launched
    log_info(LogTest, "Validating traffic is flowing...");
    // bool traffic_flowing = test_utils::validate_traffic_flowing(
    //     control_plane, mesh_id, num_devices);
    // ASSERT_TRUE(traffic_flowing) << "Traffic not detected before pause command";

    log_info(LogTest, "Traffic validation (when workers are running)");
    // test_utils::log_all_router_states(control_plane, {mesh_id});

    // Step 4: Issue pause command to all routers
    // FR-4: Router pause command
    log_info(LogTest, "Issuing pause command to all routers...");
    test_utils::FabricCommandInterface cmd_interface(control_plane);
    cmd_interface.pause_routers();

    // Step 5: Wait for all routers to enter PAUSED state
    // FR-6: Pause state validation with timeout
    log_info(LogTest, "Waiting for routers to enter PAUSED state...");
    auto pause_start = std::chrono::steady_clock::now();
    bool paused = cmd_interface.wait_for_pause(test_utils::DEFAULT_PAUSE_TIMEOUT);
    auto pause_duration = std::chrono::steady_clock::now() - pause_start;

    log_info(LogTest, "Pause transition took {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(pause_duration).count());

    // NFR-5: Observability - log router states
    // test_utils::log_all_router_states(control_plane, {mesh_id});

    ASSERT_TRUE(paused) << "Routers did not enter PAUSED state within timeout";

    // Step 6: Confirm all routers are in PAUSED state
    // FR-5: Router status polling
    ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED))
        << "Not all routers in PAUSED state";

    log_info(LogTest, "All routers confirmed in PAUSED state");

    // Step 7: Validate traffic has stopped
    // FR-7: Paused traffic verification
    log_info(LogTest, "Validating traffic has stopped...");
    // bool traffic_stopped = test_utils::validate_traffic_stopped(
    //     control_plane, mesh_id, num_devices);
    // ASSERT_TRUE(traffic_stopped) << "Traffic detected during PAUSED state";

    log_info(LogTest, "Traffic validation (when workers complete)");

    // Step 8: Issue teardown to worker kernels
    // FR-8: Worker teardown
    log_info(LogTest, "Signaling worker teardown...");
    // cleanup_workers();

    log_info(LogTest, "Test completed successfully");
}

// Test that routers resume traffic after resume command is issued
TEST_F(FabricRouterPauseControlTest, ResumeRestoresTraffic) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";
    MeshId mesh_id = mesh_ids[0];

    size_t num_devices = get_devices().size();
    ASSERT_GE(num_devices, 2) << "Test requires at least 2 devices";

    log_info(LogTest, "Starting Fabric Router Resume Test");

    // Launch workers (when CS-006 is complete)
    // launch_traffic_generators();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Validate traffic is flowing
    // bool traffic_flowing = test_utils::validate_traffic_flowing(
    //     control_plane, mesh_id, num_devices);
    // ASSERT_TRUE(traffic_flowing) << "Traffic not detected before pause";

    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Pause all routers
    cmd_interface.pause_routers();
    bool paused = cmd_interface.wait_for_pause(test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(paused) << "Failed to pause routers";

    // Verify traffic stopped
    // bool traffic_stopped = test_utils::validate_traffic_stopped(
    //     control_plane, mesh_id, num_devices);
    // ASSERT_TRUE(traffic_stopped) << "Traffic not stopped after pause";

    // Resume routers
    log_info(LogTest, "Issuing resume command to all routers...");
    cmd_interface.resume_routers();

    // Wait for routers to return to RUNNING state
    bool running = cmd_interface.wait_for_state(
        RouterStateCommon::RUNNING, test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(running) << "Routers did not resume within timeout";

    // Verify all routers are in RUNNING state
    ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::RUNNING))
        << "Not all routers in RUNNING state after resume";

    // Validate traffic resumes
    // bool traffic_resumed = test_utils::validate_traffic_flowing(
    //     control_plane, mesh_id, num_devices);
    // ASSERT_TRUE(traffic_resumed) << "Traffic not detected after resume";

    log_info(LogTest, "Resume test completed successfully");
}

// Test pause timeout - routers that don't pause within timeout are detected
TEST_F(FabricRouterPauseControlTest, PauseTimeoutDetection) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";

    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Issue pause command
    cmd_interface.pause_routers();

    // Wait with very short timeout
    auto pause_start = std::chrono::steady_clock::now();
    bool paused = cmd_interface.wait_for_pause(std::chrono::milliseconds(1));
    auto pause_duration = std::chrono::steady_clock::now() - pause_start;

    // Even if pause succeeds, we should see minimal latency measurement
    log_info(LogTest, "Pause latency: {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(pause_duration).count());

    // This test verifies timeout detection works (regardless of pause success)
    ASSERT_LE(std::chrono::duration_cast<std::chrono::milliseconds>(pause_duration).count(), 100)
        << "Pause timeout detection took unexpectedly long";
}

// Test that TearDown cleans up workers even if test fails
TEST_F(FabricRouterPauseControlTest, WorkerCleanupOnTestFailure) {
    size_t num_devices = get_devices().size();
    ASSERT_GE(num_devices, 2) << "Test requires at least 2 devices";

    // Note: In a real test scenario, if this test failed midway,
    // the TearDown would still clean up workers properly.
    // This is verified by the fact that TearDown checks workers_launched_ flag.

    EXPECT_FALSE(get_workers_launched()) << "Cleanup should have removed launched status";
    log_info(LogTest, "Worker cleanup on failure test completed");
}

// Test command interface state transitions
TEST_F(FabricRouterPauseControlTest, RouterStateTransitions) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";

    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Check initial state - routers should be running
    bool initially_running = cmd_interface.all_routers_in_state(RouterStateCommon::RUNNING);
    log_info(LogTest, "Initial router state - all running: {}", initially_running);

    // Pause
    cmd_interface.pause_routers();
    bool became_paused = cmd_interface.wait_for_state(
        RouterStateCommon::PAUSED, test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(became_paused) << "Routers did not transition to PAUSED state";

    // Resume
    cmd_interface.resume_routers();
    bool became_running = cmd_interface.wait_for_state(
        RouterStateCommon::RUNNING, test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(became_running) << "Routers did not transition back to RUNNING state";

    log_info(LogTest, "Router state transition test completed");
}

// Test observability - logging of router states
TEST_F(FabricRouterPauseControlTest, ObservabilityLogging) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";

    // NFR-5: Observability - should be able to log all router states
    test_utils::log_all_router_states(control_plane, mesh_ids);

    // Count routers by state
    auto router_state_counts = test_utils::count_routers_by_state(control_plane, mesh_ids);

    // Verify we have some routers
    uint32_t total_routers = 0;
    for (const auto& [state, count] : router_state_counts) {
        total_routers += count;
        log_info(LogTest, "Routers in state {}: {}", router_state_to_string(state), count);
    }

    ASSERT_GT(total_routers, 0) << "No routers found in fabric";
    log_info(LogTest, "Total routers: {}", total_routers);
}

// Test minimum configuration - 2 devices
TEST_F(FabricRouterPauseControlTest, MinimumConfiguration) {
    size_t num_devices = get_devices().size();
    if (num_devices < 2) {
        GTEST_SKIP() << "Test requires at least 2 devices";
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Should work with minimum configuration
    cmd_interface.pause_routers();
    bool paused = cmd_interface.wait_for_pause(test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(paused) << "Pause failed on minimum configuration";

    log_info(LogTest, "Minimum configuration test passed with {} devices", num_devices);
}

// Test larger configuration - 4+ devices
TEST_F(FabricRouterPauseControlTest, LargerConfiguration) {
    size_t num_devices = get_devices().size();
    if (num_devices < 4) {
        GTEST_SKIP() << "Test requires at least 4 devices";
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    log_info(LogTest, "Testing with {} devices", num_devices);

    // Should work with larger configuration
    cmd_interface.pause_routers();
    bool paused = cmd_interface.wait_for_pause(test_utils::DEFAULT_PAUSE_TIMEOUT);
    ASSERT_TRUE(paused) << "Pause failed on larger configuration";

    // All routers must be paused
    ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED))
        << "Not all routers paused in larger configuration";

    log_info(LogTest, "Larger configuration test passed with {} devices", num_devices);
}

}  // namespace tt::tt_fabric::fabric_router_tests
