// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// C++ test for 8-device mesh allocation tracking
// Tests distributed memory allocation across multiple devices using MeshDevice

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

using namespace tt::tt_metal;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshShape;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;
}

void print_step(int step, const std::string& description) {
    std::cout << "\n[Step " << step << "] " << description << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void print_info(const std::string& message) { std::cout << "ℹ  " << message << std::endl; }

void print_success(const std::string& message) { std::cout << "✓ " << message << std::endl; }

void print_warning(const std::string& message) { std::cout << "⚠  " << message << std::endl; }

std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = bytes;

    while (size >= 1024.0 && unit < 3) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

int main() {
    print_header("8-Device Mesh Allocation Tracking Test (C++)");

    // Check if tracking is enabled
    const char* tracking_enabled = std::getenv("TT_ALLOC_TRACKING_ENABLED");
    if (!tracking_enabled || std::string(tracking_enabled) != "1") {
        print_warning("TT_ALLOC_TRACKING_ENABLED is not set!");
        print_info("Set it with: export TT_ALLOC_TRACKING_ENABLED=1");
        print_info("Continuing anyway, but allocations won't be tracked...");
    } else {
        print_success("Allocation tracking is ENABLED");
    }

    print_info("PID: " + std::to_string(getpid()));

    // Step 1: Open 8-device mesh (2x4 topology)
    print_step(1, "Opening 8-Device Mesh (2x4 Topology)");
    print_info("Creating mesh with shape (2, 4) - 2 rows, 4 columns...");

    std::shared_ptr<MeshDevice> mesh_device;

    try {
        // Create mesh device configuration for 2x4 mesh
        MeshDeviceConfig mesh_config(MeshShape{2, 4});

        print_info("Opening mesh device...");
        mesh_device = MeshDevice::create(mesh_config);
        print_success("Mesh device opened: 2x4 grid, 8 devices");

        // Get individual devices from mesh
        auto devices = mesh_device->get_devices();
        print_info("Mesh contains " + std::to_string(devices.size()) + " devices:");
        for (size_t i = 0; i < devices.size(); i++) {
            print_info("  Device " + std::to_string(devices[i]->id()) + " ready");
        }

        print_info("\n📊 CHECK MONITOR: All 8 devices should be visible");
        print_info("   Run: ./allocation_monitor_client -a -r 500\n");

        std::this_thread::sleep_for(std::chrono::seconds(3));

    } catch (const std::exception& e) {
        std::cerr << "❌ Error opening mesh device: " << e.what() << std::endl;
        return 1;
    }

    // Get devices from mesh for buffer allocation
    auto devices = mesh_device->get_devices();

    // Step 2: Allocate buffers on each device (100MB per device)
    print_step(2, "Allocating Buffers on All Devices (100MB each)");
    print_info("Creating 100MB DRAM buffer on each device...");

    std::vector<std::shared_ptr<Buffer>> buffers_set1;
    const uint64_t BUFFER_SIZE = 100 * 1024 * 1024;  // 100MB
    const uint64_t PAGE_SIZE = 4096;

    for (size_t i = 0; i < devices.size(); i++) {
        int device_id = devices[i]->id();
        print_info("Allocating on device " + std::to_string(device_id) + " (" + format_bytes(BUFFER_SIZE) + ")...");

        auto buffer = Buffer::create(devices[i], BUFFER_SIZE, PAGE_SIZE, BufferType::DRAM);
        buffers_set1.push_back(buffer);
    }

    print_success("Allocated 100MB on each of 8 devices (800MB total)");
    print_info("\n📊 CHECK MONITOR: Each device should show ~100MB DRAM\n");
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Step 3: Allocate second set of buffers (200MB per device)
    print_step(3, "Allocating Second Set of Buffers (200MB each)");
    print_info("Creating 200MB DRAM buffer on each device...");

    std::vector<std::shared_ptr<Buffer>> buffers_set2;
    const uint64_t BUFFER_SIZE_2 = 200 * 1024 * 1024;  // 200MB

    for (size_t i = 0; i < devices.size(); i++) {
        int device_id = devices[i]->id();
        print_info("Allocating on device " + std::to_string(device_id) + " (" + format_bytes(BUFFER_SIZE_2) + ")...");

        auto buffer = Buffer::create(devices[i], BUFFER_SIZE_2, PAGE_SIZE, BufferType::DRAM);
        buffers_set2.push_back(buffer);
    }

    print_success("Allocated 200MB on each of 8 devices (1.6GB total)");
    print_info("\n📊 CHECK MONITOR: Each device should show ~300MB DRAM\n");
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Step 4: Allocate L1 buffers on each device
    print_step(4, "Allocating L1 Buffers (10MB each)");
    print_info("Creating 10MB L1 buffer on each device...");

    std::vector<std::shared_ptr<Buffer>> buffers_l1;
    const uint64_t L1_BUFFER_SIZE = 10 * 1024 * 1024;  // 10MB

    for (size_t i = 0; i < devices.size(); i++) {
        int device_id = devices[i]->id();
        try {
            print_info(
                "Allocating L1 on device " + std::to_string(device_id) + " (" + format_bytes(L1_BUFFER_SIZE) + ")...");

            auto buffer = Buffer::create(devices[i], L1_BUFFER_SIZE, PAGE_SIZE, BufferType::L1);
            buffers_l1.push_back(buffer);
        } catch (const std::exception& e) {
            print_warning("L1 allocation failed on device " + std::to_string(device_id) + ": " + e.what());
        }
    }

    if (!buffers_l1.empty()) {
        print_success("Allocated L1 on " + std::to_string(buffers_l1.size()) + " devices");
        print_info("\n📊 CHECK MONITOR: Devices should show L1 usage\n");
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Step 5: Deallocate - watch memory drop!
    print_step(5, "Deallocating Buffers - WATCH MEMORY DROP!");
    print_info("Deallocating buffers one set at a time...");

    print_info("\nDeallocating L1 buffers...");
    buffers_l1.clear();
    print_success("📊 L1 freed on all devices!");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    print_info("\nDeallocating second DRAM set (200MB per device)...");
    buffers_set2.clear();
    print_success("📊 200MB freed per device (1.6GB total)!");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    print_info("\nDeallocating first DRAM set (100MB per device)...");
    buffers_set1.clear();
    print_success("📊 100MB freed per device (800MB total)!");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    print_success("All buffers deallocated - memory back to baseline!");
    print_info("Note: System buffers (~14-15MB) remain until mesh device closes");

    // Step 6: Clear program cache (if any cached programs exist)
    print_step(6, "Clearing Program Cache");
    print_info("Clearing any cached programs...");
    print_info("\n📊 CHECK MONITOR: Watch for any cached program cleanup\n");

    mesh_device->disable_and_clear_program_cache();
    print_success("Program cache cleared!");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Step 7: Close mesh device
    print_step(7, "Closing Mesh Device");
    print_info("Closing mesh device (all 8 devices)...");
    print_info("\n📊 CHECK MONITOR: System buffers will be freed\n");

    mesh_device.reset();  // Close mesh device

    print_success("Mesh device closed");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Summary
    print_header("Test Complete!");

    if (tracking_enabled && std::string(tracking_enabled) == "1") {
        print_success("If allocation server is running, you should have seen:");
        std::cout << "  • Allocations on ALL 8 devices" << std::endl;
        std::cout << "  • Memory distributed across devices" << std::endl;
        std::cout << "  • Both DRAM and L1 allocations" << std::endl;
        std::cout << "  • Deallocations as buffers were freed" << std::endl;
        std::cout << "  • Program cache cleared" << std::endl;
        std::cout << "  • Memory returning to baseline" << std::endl;
    } else {
        print_warning("Tracking was disabled. To see allocations:");
        std::cout << "  1. Start server: ./allocation_server_poc" << std::endl;
        std::cout << "  2. Start monitor: ./allocation_monitor_client -a -r 500" << std::endl;
        std::cout << "  3. Rerun: TT_ALLOC_TRACKING_ENABLED=1 ./test_mesh_allocation_cpp" << std::endl;
    }

    std::cout << "\nCheck the allocation monitor for per-device stats!" << std::endl;
    std::cout << "You should see allocations distributed across devices 0-7\n" << std::endl;

    return 0;
}
