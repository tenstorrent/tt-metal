// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <unordered_map>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "fabric_fixture.hpp"
#include "fabric_worker_kernel_helpers.hpp"
#include "fabric_command_interface.hpp"
#include "utils.hpp"

namespace tt::tt_fabric::traffic_generator_tests {

using namespace tt::tt_fabric::test_utils;
using tt::tt_fabric::fabric_router_tests::Fabric1DFixture;

// ============================================================================
// Local telemetry helper functions using read_fabric_telemetry API directly
// ============================================================================

namespace {

// Telemetry snapshot: maps channel key -> words_sent
using TelemetryMap = std::unordered_map<uint64_t, uint64_t>;

uint64_t make_channel_key(const FabricNodeId& node_id, uint8_t channel_id) {
    return (static_cast<uint64_t>(node_id.mesh_id.get()) << 32) |
           (static_cast<uint64_t>(node_id.chip_id) << 8) |
           static_cast<uint64_t>(channel_id);
}

TelemetryMap capture_telemetry(MeshId mesh_id, size_t num_devices) {
    TelemetryMap snapshot;

    for (size_t device_idx = 0; device_idx < num_devices; ++device_idx) {
        FabricNodeId node_id(mesh_id, static_cast<uint32_t>(device_idx));

        // Call the fabric telemetry reader API directly
        auto samples = tt::tt_fabric::read_fabric_telemetry(node_id);

        for (const auto& sample : samples) {
            if (sample.snapshot.dynamic_info.has_value()) {
                const auto& dyn = sample.snapshot.dynamic_info.value();
                uint64_t key = make_channel_key(sample.fabric_node_id, sample.channel_id);
                snapshot[key] = dyn.tx_bandwidth.words_sent + dyn.rx_bandwidth.words_sent;
            }
        }
    }
    return snapshot;
}

bool telemetry_increased(const TelemetryMap& before, const TelemetryMap& after) {
    for (const auto& [key, after_words] : after) {
        auto it = before.find(key);
        uint64_t before_words = (it != before.end()) ? it->second : 0;
        if (after_words > before_words) {
            return true;
        }
    }
    return false;
}

bool check_traffic_flowing(MeshId mesh_id, size_t num_devices, std::chrono::milliseconds interval) {
    auto baseline = capture_telemetry(mesh_id, num_devices);
    std::this_thread::sleep_for(interval);
    auto after = capture_telemetry(mesh_id, num_devices);
    return telemetry_increased(baseline, after);
}

}  // anonymous namespace


// ============================================================================
// INTEGRATION TESTS - Actually run the kernel and validate traffic generation
// ============================================================================

class FabricTrafficGeneratorKernelIntegrationTest : public Fabric1DFixture {
protected:

    static void SetUpTestSuite() {
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_telemetry(true);
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_all_telemetry();
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_bw_telemetry(true);
        Fabric1DFixture::SetUpTestSuite();
    }

    void SetUp() override {

        Fabric1DFixture::SetUp();
    }

    void TearDown() override {
        Fabric1DFixture::TearDown();
    }

    void launch_traffic_generator(tt::tt_metal::CoreCoord &worker_core) {
        auto devices = get_devices();
        if (devices.size() < 2) {
            GTEST_SKIP() << "Requires at least 2 devices";
        }

        // Launch kernel on first device
        mesh_device_ = devices[0];

        // Allocate memory
        memory_layout_ = allocate_worker_memory();

        // Create destination node (second device)
        FabricNodeId dest_node(MeshId{0}, 1);  // mesh_id=0, device_id=1

        // Create and launch program (note: correct argument order)
        program_ = create_traffic_generator_program(
            mesh_device_, worker_core, dest_node, memory_layout_);

        this->RunProgramNonblocking(mesh_device_, *program_);

        log_info(LogTest, "Traffic generator kernel launched on device {}, core ({},{})",
                 mesh_device_->id(), worker_core.x, worker_core.y);
    }

    WorkerMemoryLayout memory_layout_{};
    std::shared_ptr<tt_metal::Program> program_;
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device_;
};

TEST_F(FabricTrafficGeneratorKernelIntegrationTest, KernelGeneratesTraffic) {
    if (tt::tt_metal::MetalContext::instance().hal().get_arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Test only supported for BLACKHOLE architecture";
    }

    // Get control plane and mesh IDs
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty()) << "No meshes available";
    MeshId mesh_id = mesh_ids[0];

    tt::tt_metal::CoreCoord worker_core = {0,0};
    // Launch the traffic generator kernel
    launch_traffic_generator(worker_core);
    FabricCommandInterface cmd_interface(control_plane);

    ASSERT_TRUE(cmd_interface.wait_for_state(RouterState::RUNNING, std::chrono::milliseconds(5000)))
        << "Routers did start in running state";
    // Validate traffic is flowing
    ASSERT_TRUE(check_traffic_flowing(mesh_id, get_devices().size(), std::chrono::milliseconds(500)));

    // Signal pause via control plane
    cmd_interface.pause_routers();
    ASSERT_TRUE(cmd_interface.wait_for_pause(std::chrono::milliseconds(5000)))
        << "Routers did not pause within timeout";

    // Validate there is no traffic flowing after pause
    ASSERT_TRUE(!check_traffic_flowing(mesh_id, get_devices().size(), std::chrono::milliseconds(500)));

    // Signal resume via control plane
    cmd_interface.resume_routers();
    ASSERT_TRUE(cmd_interface.wait_for_state(RouterState::RUNNING, std::chrono::milliseconds(5000)))
        << "Routers did not resume within timeout";

    // Validate traffic is flowing again
    ASSERT_TRUE(check_traffic_flowing(mesh_id, get_devices().size(), std::chrono::milliseconds(500)));

    // Signal teardown and wait for completion
    signal_worker_teardown(mesh_device_, worker_core,
        memory_layout_.teardown_signal_address);

    log_info(LogTest, "Traffic generation validated - kernel is actively sending packets");
}


}  // namespace tt::tt_fabric::traffic_generator_tests
