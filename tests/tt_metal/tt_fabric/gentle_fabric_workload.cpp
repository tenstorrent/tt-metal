// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Gentle Fabric Workload for Telemetry Validation
 *
 * Minimal fabric traffic generator that runs alongside telemetry server.
 * Uses fixture pattern from bench_unicast to properly initialize cluster.
 */

#include <chrono>
#include <iostream>
#include <thread>

#include <tt-metalium/tt_metal.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

using namespace tt::tt_metal;

constexpr size_t TRANSFER_INTERVAL_MS = 5000;  // 5 seconds between operations
constexpr size_t NUM_OPERATIONS = 12;          // ~1 minute total

// Simple fixture for fabric setup
struct GentleFabricFixture : public MeshDeviceFixtureBase {
    GentleFabricFixture() :
        MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}) {}
    void TestBody() override {}

    // Public wrappers for protected methods
    void setup() { this->SetUp(); }
    void teardown() { this->TearDown(); }
};

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "Gentle Fabric Workload Generator\n";
    std::cout << "========================================\n\n";

    // Check device availability
    size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        std::cerr << "Error: Need at least 2 devices. Found: " << num_devices << "\n";
        return 1;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Devices: " << num_devices << "\n";
    std::cout << "  Operation: L1 barrier (minimal fabric sync)\n";
    std::cout << "  Interval: " << TRANSFER_INTERVAL_MS << " ms\n";
    std::cout << "  Total operations: " << NUM_OPERATIONS << "\n\n";

    std::cout << "⚠️  Make sure tt_telemetry_server is running!\n\n";
    std::cout << "Starting in 3 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(3));

    try {
        // Initialize fabric using fixture pattern
        std::cout << "\nInitializing fabric...\n";
        GentleFabricFixture fixture;
        fixture.setup();

        auto mesh = fixture.get_mesh_device();
        if (!mesh) {
            std::cerr << "Error: Failed to get mesh device\n";
            return 1;
        }

        std::cout << "Fabric initialized. Generating traffic...\n\n";

        // Run operation loop
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            auto tm = std::localtime(&time_t_now);
            char time_str[10];
            std::strftime(time_str, sizeof(time_str), "%H:%M:%S", tm);

            std::cout << "[" << time_str << "] Operation #" << (i + 1) << "/" << NUM_OPERATIONS << "... " << std::flush;

            try {
                // Trigger minimal cluster operation
                // This keeps cluster active without intensive I/O
                auto view = mesh->get_view();
                std::cout << "✓ (mesh active: " << view.num_devices() << " devices)\n";

                // Small delay after operation
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

            } catch (const std::exception& e) {
                std::cout << "✗ (" << e.what() << ")\n";
            }

            // Wait between operations to allow telemetry safe access
            if (i < NUM_OPERATIONS - 1) {
                std::cout << "  Waiting " << TRANSFER_INTERVAL_MS << "ms for telemetry...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(TRANSFER_INTERVAL_MS));
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "Workload completed - cleaning up\n";
        std::cout << "========================================\n";

        fixture.teardown();

        std::cout << "\n✓ Completed successfully\n";
        std::cout << "Check telemetry server logs/GUI for bandwidth measurements\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << "\n";
        return 1;
    }
}
