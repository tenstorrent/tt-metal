// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Simple test to check if GraphTracker is interfering with allocation tracking

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    std::cout << "=== GraphTracker Status Check ===" << std::endl;

    // Check GraphTracker status BEFORE device creation
    auto& tracker = GraphTracker::instance();
    std::cout << "GraphTracker enabled: " << (tracker.is_enabled() ? "YES" : "NO") << std::endl;
    std::cout << "GraphTracker has hook: " << (tracker.get_hook() != nullptr ? "YES" : "NO") << std::endl;
    std::cout << "GraphTracker processors: " << tracker.get_processors().size() << std::endl;

    std::cout << "\n=== Creating Device ===" << std::endl;
    IDevice* device = CreateDevice(0);
    std::cout << "Device created: " << device->id() << std::endl;

    // Check GraphTracker status AFTER device creation
    std::cout << "\nGraphTracker enabled: " << (tracker.is_enabled() ? "YES" : "NO") << std::endl;
    std::cout << "GraphTracker has hook: " << (tracker.get_hook() != nullptr ? "YES" : "NO") << std::endl;
    std::cout << "GraphTracker processors: " << tracker.get_processors().size() << std::endl;

    std::cout << "\n=== Allocating DRAM Buffer ===" << std::endl;
    auto buffer = Buffer::create(
        device,
        100 * 1024 * 1024,  // 100 MB
        4096,
        BufferType::DRAM);
    std::cout << "Buffer created at address: 0x" << std::hex << buffer->address() << std::dec << std::endl;
    std::cout << "Buffer size: " << buffer->size() << " bytes" << std::endl;

    std::cout << "\n💡 Check allocation_server_poc output to see if this was tracked!" << std::endl;
    std::cout << "   Sleeping for 5 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "\n=== Cleaning Up ===" << std::endl;
    CloseDevice(device);

    std::cout << "✓ Test complete" << std::endl;
    return 0;
}
