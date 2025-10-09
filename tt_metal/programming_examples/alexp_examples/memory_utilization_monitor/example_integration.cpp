// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Example: Integrating TracyMemoryMonitor into Your Application
 *
 * This example demonstrates how to use TracyMemoryMonitor to track
 * and query device memory allocations in your application.
 */

#include <tt_metal/impl/profiler/tracy_memory_monitor.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

using namespace tt::tt_metal;

// Helper function to format bytes
std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 3) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

// Example 1: Simple memory query
void example_simple_query() {
    std::cout << "\n=== Example 1: Simple Memory Query ===" << std::endl;

    auto& monitor = TracyMemoryMonitor::instance();

    // Query device 0
    auto stats = monitor.query_device(0);

    std::cout << "Device 0 Memory Statistics:" << std::endl;
    std::cout << "  DRAM:   " << format_bytes(stats.dram_allocated) << std::endl;
    std::cout << "  L1:     " << format_bytes(stats.l1_allocated) << std::endl;
    std::cout << "  Buffers: " << stats.num_buffers << std::endl;
    std::cout << "  Total Allocs: " << stats.total_allocs << std::endl;
    std::cout << "  Total Frees:  " << stats.total_frees << std::endl;
}

// Example 2: Monitor memory changes
void example_monitor_changes() {
    std::cout << "\n=== Example 2: Monitor Memory Changes ===" << std::endl;

    auto& monitor = TracyMemoryMonitor::instance();

    // Get baseline
    auto before = monitor.query_device(0);
    std::cout << "Before: " << format_bytes(before.get_total_allocated()) << std::endl;

    // Simulate allocation (in real code, this would be Buffer::create)
    std::cout << "Simulating buffer allocation..." << std::endl;
    monitor.track_allocation(0, 0x1000000, 1024 * 1024 * 100, TracyMemoryMonitor::BufferType::DRAM);

    // Check change
    auto after = monitor.query_device(0);
    std::cout << "After:  " << format_bytes(after.get_total_allocated()) << std::endl;

    uint64_t increase = after.get_total_allocated() - before.get_total_allocated();
    std::cout << "Increase: " << format_bytes(increase) << std::endl;

    // Cleanup
    monitor.track_deallocation(0, 0x1000000);

    auto final = monitor.query_device(0);
    std::cout << "After cleanup: " << format_bytes(final.get_total_allocated()) << std::endl;
}

// Example 3: Multi-device monitoring
void example_multi_device() {
    std::cout << "\n=== Example 3: Multi-Device Monitoring ===" << std::endl;

    auto& monitor = TracyMemoryMonitor::instance();

    // Simulate allocations on multiple devices
    monitor.track_allocation(0, 0x2000000, 1024 * 1024 * 50, TracyMemoryMonitor::BufferType::DRAM);
    monitor.track_allocation(1, 0x3000000, 1024 * 1024 * 75, TracyMemoryMonitor::BufferType::L1);
    monitor.track_allocation(2, 0x4000000, 1024 * 1024 * 100, TracyMemoryMonitor::BufferType::DRAM);

    // Query all devices
    auto all_stats = monitor.query_all_devices();

    std::cout << "Device Summary:" << std::endl;
    for (int i = 0; i < TracyMemoryMonitor::MAX_DEVICES; i++) {
        uint64_t total = all_stats[i].get_total_allocated();
        if (total > 0) {
            std::cout << "  Device " << i << ": " << format_bytes(total) << " (" << all_stats[i].num_buffers
                      << " buffers)" << std::endl;
        }
    }

    // Cleanup
    monitor.track_deallocation(0, 0x2000000);
    monitor.track_deallocation(1, 0x3000000);
    monitor.track_deallocation(2, 0x4000000);
}

// Example 4: Real-time monitoring loop
void example_realtime_monitoring() {
    std::cout << "\n=== Example 4: Real-time Monitoring (5 seconds) ===" << std::endl;

    auto& monitor = TracyMemoryMonitor::instance();

    // Simulate some activity
    std::thread activity_thread([&]() {
        for (int i = 0; i < 10; i++) {
            uint64_t addr = 0x5000000 + i * 0x100000;
            monitor.track_allocation(0, addr, 1024 * 1024 * 10, TracyMemoryMonitor::BufferType::L1);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            if (i > 0) {
                uint64_t prev_addr = 0x5000000 + (i - 1) * 0x100000;
                monitor.track_deallocation(0, prev_addr);
            }
        }

        // Cleanup remaining
        uint64_t last_addr = 0x5000000 + 9 * 0x100000;
        monitor.track_deallocation(0, last_addr);
    });

    // Monitor in main thread
    for (int i = 0; i < 17; i++) {  // 17 * 300ms ≈ 5s
        auto stats = monitor.query_device(0);
        std::cout << "\r" << std::string(80, ' ') << "\r";  // Clear line
        std::cout << "L1: " << std::setw(10) << format_bytes(stats.l1_allocated) << " | Buffers: " << std::setw(3)
                  << stats.num_buffers << " | Allocs: " << std::setw(3) << stats.total_allocs
                  << " | Frees: " << std::setw(3) << stats.total_frees << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    activity_thread.join();
    std::cout << std::endl;
}

// Example 5: Buffer type breakdown
void example_buffer_type_breakdown() {
    std::cout << "\n=== Example 5: Buffer Type Breakdown ===" << std::endl;

    auto& monitor = TracyMemoryMonitor::instance();

    // Allocate different buffer types
    monitor.track_allocation(0, 0x6000000, 1024 * 1024 * 100, TracyMemoryMonitor::BufferType::DRAM);
    monitor.track_allocation(0, 0x6100000, 1024 * 1024 * 50, TracyMemoryMonitor::BufferType::L1);
    monitor.track_allocation(0, 0x6200000, 1024 * 1024 * 10, TracyMemoryMonitor::BufferType::L1_SMALL);
    monitor.track_allocation(0, 0x6300000, 1024 * 1024 * 200, TracyMemoryMonitor::BufferType::SYSTEM_MEMORY);
    monitor.track_allocation(0, 0x6400000, 1024 * 1024 * 25, TracyMemoryMonitor::BufferType::TRACE);

    auto stats = monitor.query_device(0);

    std::cout << "Device 0 Buffer Type Breakdown:" << std::endl;
    std::cout << "  DRAM:       " << std::setw(12) << format_bytes(stats.dram_allocated) << std::endl;
    std::cout << "  L1:         " << std::setw(12) << format_bytes(stats.l1_allocated) << std::endl;
    std::cout << "  L1_SMALL:   " << std::setw(12) << format_bytes(stats.l1_small_allocated) << std::endl;
    std::cout << "  SYSTEM_MEM: " << std::setw(12) << format_bytes(stats.system_memory_allocated) << std::endl;
    std::cout << "  TRACE:      " << std::setw(12) << format_bytes(stats.trace_allocated) << std::endl;
    std::cout << "  ──────────────────────" << std::endl;
    std::cout << "  TOTAL:      " << std::setw(12) << format_bytes(stats.get_total_allocated()) << std::endl;

    // Cleanup
    monitor.track_deallocation(0, 0x6000000);
    monitor.track_deallocation(0, 0x6100000);
    monitor.track_deallocation(0, 0x6200000);
    monitor.track_deallocation(0, 0x6300000);
    monitor.track_deallocation(0, 0x6400000);
}

// Example 6: Integration with Tracy
void example_tracy_integration() {
    std::cout << "\n=== Example 6: Tracy Integration ===" << std::endl;

#ifdef TRACY_ENABLE
    std::cout << "Tracy profiling is ENABLED" << std::endl;
    std::cout << "All allocations are being sent to Tracy profiler." << std::endl;
    std::cout << "\nTo view in Tracy GUI:" << std::endl;
    std::cout << "1. Launch Tracy profiler: tracy" << std::endl;
    std::cout << "2. Connect to this application" << std::endl;
    std::cout << "3. Open Memory window" << std::endl;
    std::cout << "4. Select memory pool: TT_Dev0_DRAM, TT_Dev0_L1, etc." << std::endl;
    std::cout << "5. View allocations, timeline, and call stacks" << std::endl;
#else
    std::cout << "Tracy profiling is DISABLED" << std::endl;
    std::cout << "Compile with -DTRACY_ENABLE to enable Tracy integration." << std::endl;
    std::cout << "Only real-time queries are available." << std::endl;
#endif

    std::cout << "\nFeatures available:" << std::endl;
    std::cout << "  ✓ Real-time memory queries (always available)" << std::endl;
    std::cout << "  ✓ Per-device statistics (always available)" << std::endl;
    std::cout << "  ✓ Active buffer tracking (always available)" << std::endl;
#ifdef TRACY_ENABLE
    std::cout << "  ✓ Memory timeline (Tracy GUI)" << std::endl;
    std::cout << "  ✓ Allocation call stacks (Tracy GUI)" << std::endl;
    std::cout << "  ✓ Memory map visualization (Tracy GUI)" << std::endl;
#else
    std::cout << "  ✗ Memory timeline (requires Tracy)" << std::endl;
    std::cout << "  ✗ Allocation call stacks (requires Tracy)" << std::endl;
    std::cout << "  ✗ Memory map visualization (requires Tracy)" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║    TracyMemoryMonitor - Integration Examples              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    // Reset monitor for clean examples
    TracyMemoryMonitor::instance().reset();

    // Run examples
    example_simple_query();
    example_monitor_changes();
    example_multi_device();
    example_realtime_monitoring();
    example_buffer_type_breakdown();
    example_tracy_integration();

    // Final cleanup
    TracyMemoryMonitor::instance().reset();

    std::cout << "\n✅ All examples completed successfully!" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "  1. Integrate monitor queries into your code" << std::endl;
    std::cout << "  2. Use in tests to validate memory usage" << std::endl;
    std::cout << "  3. Enable Tracy for detailed profiling" << std::endl;
    std::cout << "  4. See TRACY_MEMORY_MONITOR.md for more info" << std::endl;

    return 0;
}
