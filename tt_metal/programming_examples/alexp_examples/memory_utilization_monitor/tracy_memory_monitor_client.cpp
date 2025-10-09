// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Real-time Memory Monitor Client using Tracy Infrastructure
 *
 * This tool provides real-time monitoring of device memory allocations
 * by directly querying the TracyMemoryMonitor embedded in the application.
 *
 * Unlike allocation_monitor_client.cpp which requires a separate server process,
 * this tool can be used as:
 * 1. A standalone library linked into your application
 * 2. Part of your test/benchmark code for real-time monitoring
 * 3. A visualization tool that polls memory stats
 *
 * Benefits over socket-based monitoring:
 * - No separate server process needed
 * - Lower latency (no IPC overhead)
 * - Works with Tracy profiler for detailed analysis
 * - Type-safe API (no binary protocol)
 */

#include "tt_metal/impl/profiler/tracy_memory_monitor.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include <csignal>
#include <atomic>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

using namespace tt::tt_metal;

#define TRACY_MEMORY_SOCKET "/tmp/tracy_memory_monitor.sock"

// Message protocol matching tracy_memory_server.cpp
struct __attribute__((packed)) MemoryQueryMessage {
    enum Type : uint8_t { QUERY = 1, RESPONSE = 2 };

    Type type;
    uint8_t pad1[3];
    int32_t device_id;
    uint8_t pad2[4];

    // Response fields
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t system_memory_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t num_buffers;
    uint64_t total_allocs;
    uint64_t total_frees;
};

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signal_handler(int signal) { g_running = false; }

// Socket-based monitor (cross-process)
class SocketMonitorClient {
private:
    int socket_fd_ = -1;
    bool connected_ = false;

public:
    SocketMonitorClient() { try_connect(); }

    ~SocketMonitorClient() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }

    bool try_connect() {
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            return false;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TRACY_MEMORY_SOCKET, sizeof(addr.sun_path) - 1);

        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(socket_fd_);
            socket_fd_ = -1;
            connected_ = false;
            return false;
        }

        connected_ = true;
        return true;
    }

    bool is_connected() const { return connected_; }

    TracyMemoryMonitor::DeviceMemoryStats query_device(int device_id) {
        if (!connected_) {
            return TracyMemoryMonitor::DeviceMemoryStats{};
        }

        MemoryQueryMessage query;
        memset(&query, 0, sizeof(query));
        query.type = MemoryQueryMessage::QUERY;
        query.device_id = device_id;

        if (send(socket_fd_, &query, sizeof(query), 0) < 0) {
            connected_ = false;
            return TracyMemoryMonitor::DeviceMemoryStats{};
        }

        MemoryQueryMessage response;
        if (recv(socket_fd_, &response, sizeof(response), 0) < 0) {
            connected_ = false;
            return TracyMemoryMonitor::DeviceMemoryStats{};
        }

        // Convert to DeviceMemoryStats
        TracyMemoryMonitor::DeviceMemoryStats stats;
        stats.dram_allocated = response.dram_allocated;
        stats.l1_allocated = response.l1_allocated;
        stats.system_memory_allocated = response.system_memory_allocated;
        stats.l1_small_allocated = response.l1_small_allocated;
        stats.trace_allocated = response.trace_allocated;
        stats.num_buffers = response.num_buffers;
        stats.total_allocs = response.total_allocs;
        stats.total_frees = response.total_frees;

        return stats;
    }
};

class MonitorDisplay {
private:
    // ANSI colors
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";

    std::string format_bytes(uint64_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);

        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            unit++;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    }

    double calculate_utilization(uint64_t allocated, uint64_t total) {
        if (total == 0) {
            return 0.0;
        }
        return (static_cast<double>(allocated) / static_cast<double>(total)) * 100.0;
    }

    const char* get_color(double percentage) {
        if (percentage >= 90.0) {
            return RED;
        }
        if (percentage >= 75.0) {
            return YELLOW;
        }
        return GREEN;
    }

    void print_bar(double percentage, int width = 30) {
        std::cout << "[";
        int filled = static_cast<int>((percentage / 100.0) * width);
        const char* color = get_color(percentage);

        for (int i = 0; i < width; ++i) {
            if (i < filled) {
                std::cout << color << "█" << RESET;
            } else {
                std::cout << "░";
            }
        }
        std::cout << "] " << color << std::fixed << std::setprecision(1) << percentage << "%" << RESET;
    }

public:
    void display_device_stats(int device_id, const TracyMemoryMonitor::DeviceMemoryStats& stats) {
        // Device capacities (typical values - adjust based on your hardware)
        uint64_t total_dram = 12ULL * 1024 * 1024 * 1024;   // 12GB
        uint64_t total_l1 = 75 * 1024 * 1024;               // 75MB
        uint64_t total_system = 4ULL * 1024 * 1024 * 1024;  // 4GB host memory

        std::cout << BOLD << CYAN << "Device " << device_id << RESET << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        // DRAM
        uint64_t dram_alloc = stats.dram_allocated;
        double dram_util = calculate_utilization(dram_alloc, total_dram);
        std::cout << "  " << BOLD << "DRAM" << RESET << ":      " << std::setw(12) << format_bytes(dram_alloc) << " / "
                  << std::setw(12) << format_bytes(total_dram) << "  ";
        print_bar(dram_util, 25);
        std::cout << std::endl;

        // L1
        uint64_t l1_alloc = stats.l1_allocated;
        double l1_util = calculate_utilization(l1_alloc, total_l1);
        std::cout << "  " << BOLD << "L1" << RESET << ":        " << std::setw(12) << format_bytes(l1_alloc) << " / "
                  << std::setw(12) << format_bytes(total_l1) << "  ";
        print_bar(l1_util, 25);
        std::cout << std::endl;

        // System Memory
        uint64_t sys_alloc = stats.system_memory_allocated;
        if (sys_alloc > 0) {
            double sys_util = calculate_utilization(sys_alloc, total_system);
            std::cout << "  " << BOLD << "SYS_MEM" << RESET << ":   " << std::setw(12) << format_bytes(sys_alloc)
                      << " / " << std::setw(12) << format_bytes(total_system) << "  ";
            print_bar(sys_util, 25);
            std::cout << std::endl;
        }

        // L1_SMALL
        uint64_t l1_small = stats.l1_small_allocated;
        if (l1_small > 0) {
            std::cout << "  " << BOLD << "L1_SMALL" << RESET << ": " << std::setw(12) << format_bytes(l1_small)
                      << std::endl;
        }

        // TRACE
        uint64_t trace = stats.trace_allocated;
        if (trace > 0) {
            std::cout << "  " << BOLD << "TRACE" << RESET << ":     " << std::setw(12) << format_bytes(trace)
                      << std::endl;
        }

        // Statistics
        std::cout << "  " << BLUE << "Active Buffers: " << RESET << stats.num_buffers << "   " << BLUE
                  << "Total Allocs: " << RESET << stats.total_allocs << "   " << BLUE << "Total Frees: " << RESET
                  << stats.total_frees << std::endl;

        std::cout << std::endl;
    }

    void display_header(int refresh_ms, const std::vector<int>& device_ids) {
        // Clear screen
        std::cout << "\033[2J\033[H";

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::cout << "═══════════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "  " << BOLD << "Tracy Memory Monitor" << RESET;
#ifdef TRACY_ENABLE
        std::cout << " " << GREEN << "[Tracy Profiling ENABLED]" << RESET;
#else
        std::cout << " " << YELLOW << "[Tracy Profiling DISABLED]" << RESET;
#endif
        std::cout << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════════════════" << std::endl;
        std::cout << "Time: " << std::put_time(&tm, "%H:%M:%S") << " | Refresh: " << refresh_ms << "ms"
                  << " | Devices: ";
        for (size_t i = 0; i < device_ids.size(); ++i) {
            std::cout << device_ids[i];
            if (i < device_ids.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl << std::endl;
    }

    void display_footer(const std::vector<int>& device_ids) {
        std::cout << "-----------------------------------------------------------------------" << std::endl;
        std::cout << MAGENTA << "💡 Features:" << RESET << std::endl;
        std::cout << "   • Real-time monitoring with lock-free queries" << std::endl;
        std::cout << "   • Integrated with Tracy profiler for detailed analysis" << std::endl;
        std::cout << "   • Tracks all buffer types (DRAM, L1, SYS_MEM, L1_SMALL, TRACE)" << std::endl;
#ifdef TRACY_ENABLE
        std::cout << "   • Connect Tracy GUI for memory timeline & allocation hotspots" << RESET << std::endl;
#endif
        std::cout << std::endl;
        std::cout << CYAN << "Press Ctrl+C to exit" << RESET << std::endl;
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Real-time memory monitor using Tracy infrastructure" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -r <ms>        Refresh interval in milliseconds (default: 1000)" << std::endl;
    std::cout << "  -d <id>        Device ID to monitor (can specify multiple times)" << std::endl;
    std::cout << "  -a, --all      Monitor all devices (0-7)" << std::endl;
    std::cout << "  -s, --single   Single query mode (print once and exit)" << std::endl;
    std::cout << "  -h, --help     Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " -d 0                  # Monitor device 0" << std::endl;
    std::cout << "  " << program_name << " -d 0 -d 1 -d 2        # Monitor devices 0, 1, 2" << std::endl;
    std::cout << "  " << program_name << " -a -r 500             # Monitor all devices, 500ms refresh" << std::endl;
    std::cout << "  " << program_name << " -s -d 0               # Single query for device 0" << std::endl;
    std::cout << std::endl;
    std::cout << "Integration:" << std::endl;
    std::cout << "  This tool can be used standalone or integrated into your application." << std::endl;
    std::cout << "  Link with: -ltt_metal -lprofiler" << std::endl;
#ifdef TRACY_ENABLE
    std::cout << "  Tracy profiling is ENABLED - connect Tracy GUI for detailed analysis" << std::endl;
#else
    std::cout << "  Tracy profiling is DISABLED - only real-time stats available" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    int refresh_ms = 1000;
    std::vector<int> device_ids;
    bool single_query = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) {
            refresh_ms = std::stoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            device_ids.push_back(std::stoi(argv[++i]));
        } else if (arg == "-a" || arg == "--all") {
            device_ids.clear();
            for (int d = 0; d < TracyMemoryMonitor::MAX_DEVICES; ++d) {
                device_ids.push_back(d);
            }
        } else if (arg == "-s" || arg == "--single") {
            single_query = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Default to device 0 if no devices specified
    if (device_ids.empty()) {
        device_ids.push_back(0);
    }

    // Setup signal handler for clean exit
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    MonitorDisplay display;

    // Try to connect to socket server first (cross-process)
    SocketMonitorClient socket_client;
    bool use_socket = socket_client.is_connected();

    if (use_socket) {
        std::cout << "\n✅ Connected to tracy_memory_server (cross-process mode)" << std::endl;
        std::cout << "   Will show allocations from ALL processes!" << std::endl;
    } else {
        std::cout << "\n📍 Using local singleton (same-process mode)" << std::endl;
        std::cout << "   Only shows allocations from THIS process." << std::endl;
        std::cout << "\n💡 For cross-process monitoring:" << std::endl;
        std::cout << "   Terminal 1: ./tracy_memory_server" << std::endl;
        std::cout << "   Terminal 2: ./tracy_memory_monitor_client -a" << std::endl;
        std::cout << "   Terminal 3: python test_mesh_allocation.py\n" << std::endl;
    }

    auto& local_monitor = TracyMemoryMonitor::instance();

    // Single query mode
    if (single_query) {
        std::cout << "Tracy Memory Monitor - Single Query" << std::endl;
        std::cout << "═══════════════════════════════════════" << std::endl;
        std::cout << std::endl;

        for (int device_id : device_ids) {
            auto stats = use_socket ? socket_client.query_device(device_id) : local_monitor.query_device(device_id);
            display.display_device_stats(device_id, stats);
        }

        return 0;
    }

    // Continuous monitoring mode
    std::cout << "\n📊 Tracy Memory Monitor Starting..." << std::endl;
    std::cout << "   Monitoring device(s): ";
    for (size_t i = 0; i < device_ids.size(); ++i) {
        std::cout << device_ids[i];
        if (i < device_ids.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
    std::cout << "   Refresh: " << refresh_ms << "ms" << std::endl;
    std::cout << "   Initializing..." << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    while (g_running) {
        display.display_header(refresh_ms, device_ids);

        // Query and display all devices
        for (int device_id : device_ids) {
            auto stats = use_socket ? socket_client.query_device(device_id) : local_monitor.query_device(device_id);
            display.display_device_stats(device_id, stats);
        }

        display.display_footer(device_ids);

        std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
    }

    std::cout << "\n👋 Tracy Memory Monitor stopped\n" << std::endl;
    return 0;
}
