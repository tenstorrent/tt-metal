// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Simple C++ test to verify allocation tracking works

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

int main() {
    // Check if tracking is enabled
    const char* tracking_enabled = std::getenv("TT_ALLOC_TRACKING_ENABLED");
    std::cout << "TT_ALLOC_TRACKING_ENABLED = " << (tracking_enabled ? tracking_enabled : "not set") << std::endl;

    if (!tracking_enabled || std::string(tracking_enabled) != "1") {
        std::cerr << "ERROR: Set TT_ALLOC_TRACKING_ENABLED=1 before running!" << std::endl;
        return 1;
    }

    std::cout << "Opening device 0..." << std::endl;
    IDevice* device = CreateDevice(0);

    std::cout << "Device opened. Waiting 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Allocating DRAM buffer (10 MB)..." << std::endl;
    auto buffer = Buffer::create(
        device,
        1024 * 1024 * 1024,  // 1 GB
        4096,                // page size
        BufferType::DRAM);

    std::cout << "Buffer allocated at address: " << buffer->address() << std::endl;
    std::cout << "Waiting 5 seconds (check allocation_server_poc)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "Deallocating buffer..." << std::endl;
    buffer.reset();

    std::cout << "Waiting 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Closing device..." << std::endl;
    CloseDevice(device);

    std::cout << "Done!" << std::endl;
    return 0;
}
