// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include <vector>
#include <sstream>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Forward declarations for types not in public API
using chip_id_t = int;

class MemoryMonitor {
private:
    std::atomic<bool> running_{true};
    std::vector<IDevice*> devices_;
    int refresh_interval_ms_;

    // ANSI color codes for terminal output
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";

public:
    MemoryMonitor(int refresh_interval_ms = 1000) : devices_(), refresh_interval_ms_(refresh_interval_ms) {
        // Set up signal handler for graceful shutdown
        signal(SIGINT, [](int) {
            std::cout << "\n" << BOLD << YELLOW << "Shutting down memory monitor..." << RESET << std::endl;
            exit(0);
        });
    }

    ~MemoryMonitor() = default;

    bool initialize_devices() {
        try {
            auto num_devices = GetNumAvailableDevices();
            if (num_devices == 0) {
                std::cerr << RED << "Error: No TT devices available" << RESET << std::endl;
                return false;
            }

            std::cout << GREEN << "Found " << num_devices << " TT device(s)" << RESET << std::endl;

            // Create only the first device to avoid multi-device issues
            // Note: Multi-device initialization often requires CreateDevices() API
            try {
                auto device = CreateDevice(0);
                if (device && device->is_initialized()) {
                    devices_.push_back(device);
                    std::cout << GREEN << "Device 0 initialized successfully" << RESET << std::endl;
                } else {
                    std::cerr << RED << "Device 0 failed to initialize properly" << RESET << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << RED << "Error initializing device 0: " << e.what() << RESET << std::endl;
            }

            if (devices_.empty()) {
                std::cerr << RED << "Error: No devices could be initialized" << RESET << std::endl;
                return false;
            }

            std::cout << GREEN << "Successfully initialized " << devices_.size() << " device(s)" << RESET << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << RED << "Error initializing devices: " << e.what() << RESET << std::endl;
            return false;
        }
    }

    std::string format_bytes(size_t bytes) {
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

    double calculate_utilization_percentage(size_t allocated, size_t total) {
        if (total == 0) {
            return 0.0;
        }
        return (static_cast<double>(allocated) / static_cast<double>(total)) * 100.0;
    }

    std::string get_utilization_color(double percentage) {
        if (percentage >= 90.0) {
            return RED;
        }
        if (percentage >= 75.0) {
            return YELLOW;
        }
        return GREEN;
    }

    void print_header() {
        std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << CYAN << "|" << std::setw(40) << "TT Device Memory Utilization Monitor" << std::setw(40)
                  << "|" << RESET << std::endl;
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << WHITE << "Press Ctrl+C to exit" << RESET << std::endl;
        std::cout << std::endl;
    }

    void print_memory_buffer_stats(IDevice* device, BufferType buffer_type, const std::string& buffer_name) {
        try {
            // Check if device is initialized before accessing allocator
            if (!device || !device->is_initialized()) {
                std::cout << "  " << BOLD << buffer_name << " Memory:" << RESET << std::endl;
                std::cout << "    " << RED << "Device not initialized" << RESET << std::endl;
                std::cout << std::endl;
                return;
            }

            // Get actual runtime memory statistics from the allocator
            const auto& allocator = device->allocator();
            if (!allocator) {
                std::cout << "  " << BOLD << buffer_name << " Memory:" << RESET << std::endl;
                std::cout << "    " << RED << "Allocator not available" << RESET << std::endl;
                std::cout << std::endl;
                return;
            }

            auto stats = allocator->get_statistics(buffer_type);
            auto num_banks = allocator->get_num_banks(buffer_type);

            // Calculate totals across all banks
            size_t total_bytes = stats.total_allocatable_size_bytes * num_banks;
            size_t allocated_bytes = stats.total_allocated_bytes * num_banks;
            size_t free_bytes = stats.total_free_bytes * num_banks;
            size_t largest_free = stats.largest_free_block_bytes;

            double utilization = calculate_utilization_percentage(allocated_bytes, total_bytes);
            std::string color = get_utilization_color(utilization);

            std::cout << "  " << BOLD << buffer_name << " Memory:" << RESET << std::endl;
            std::cout << "    Banks: " << num_banks << std::endl;
            std::cout << "    Total:          " << std::setw(15) << format_bytes(total_bytes) << std::endl;
            std::cout << "    Allocated:      " << color << std::setw(15) << format_bytes(allocated_bytes) << " ("
                      << std::fixed << std::setprecision(1) << utilization << "%)" << RESET << std::endl;
            std::cout << "    Free:           " << std::setw(15) << format_bytes(free_bytes) << std::endl;
            std::cout << "    Largest Block:  " << std::setw(15) << format_bytes(largest_free) << std::endl;

            // Visual memory bar
            std::cout << "    Usage: [";
            int bar_width = 40;
            int filled = static_cast<int>((utilization / 100.0) * bar_width);
            for (int j = 0; j < bar_width; ++j) {
                if (j < filled) {
                    std::cout << color << "█" << RESET;
                } else {
                    std::cout << "░";
                }
            }
            std::cout << "] " << color << std::fixed << std::setprecision(1) << utilization << "%" << RESET
                      << std::endl;
            std::cout << std::endl;

        } catch (const std::exception& e) {
            std::cout << "    " << RED << "Error reading " << buffer_name << " memory: " << e.what() << RESET
                      << std::endl;
            std::cout << std::endl;
        }
    }

    void print_device_memory_status(IDevice* device, int device_index) {
        if (!device) {
            std::cout << BOLD << BLUE << "Device " << device_index << RESET << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            std::cout << "  " << RED << "Error: Device pointer is null" << RESET << std::endl;
            std::cout << std::endl;
            return;
        }

        std::cout << BOLD << BLUE << "Device " << device_index << " (ID: " << device->id() << ")" << RESET << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        try {
            // Get basic device information
            std::cout << "  " << BOLD << "Device Information:" << RESET << std::endl;
            std::cout << "    Architecture: " << static_cast<int>(device->arch()) << std::endl;
            if (device->is_initialized()) {
                std::cout << "    Initialized: " << GREEN << "Yes" << RESET << std::endl;
            } else {
                std::cout << "    Initialized: " << RED << "No" << RESET << std::endl;
            }
            std::cout << "    Hardware CQs: " << static_cast<int>(device->num_hw_cqs()) << std::endl;
            std::cout << "    DRAM Channels: " << device->num_dram_channels() << std::endl;

            // Grid information
            auto grid_size = device->grid_size();
            std::cout << "    Compute Grid: " << grid_size.x << "x" << grid_size.y << " ("
                      << (grid_size.x * grid_size.y) << " cores)" << std::endl;
            std::cout << std::endl;

            // Print ACTUAL runtime memory utilization for each buffer type
            std::cout << "  " << BOLD << CYAN << "Real-Time Memory Utilization:" << RESET << std::endl;
            std::cout << std::endl;

            // Query actual allocator statistics
            print_memory_buffer_stats(device, BufferType::DRAM, "DRAM");
            print_memory_buffer_stats(device, BufferType::L1, "L1");
            print_memory_buffer_stats(device, BufferType::L1_SMALL, "L1_SMALL");
            print_memory_buffer_stats(device, BufferType::TRACE, "TRACE");

        } catch (const std::exception& e) {
            std::cout << "    " << RED << "Error reading device information: " << e.what() << RESET << std::endl;
        }
    }

    void print_system_info() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::cout << BOLD << MAGENTA << "System Info:" << RESET << std::endl;
        std::cout << "  Time: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout << "  Refresh: " << refresh_interval_ms_ << "ms" << std::endl;
        std::cout << "  Devices: " << devices_.size() << std::endl;
        std::cout << std::endl;
    }

    void run() {
        if (!initialize_devices()) {
            return;
        }

        std::cout << BOLD << GREEN << "Starting memory monitor..." << RESET << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        while (running_) {
            print_header();
            print_system_info();

            for (size_t i = 0; i < devices_.size(); ++i) {
                print_device_memory_status(devices_[i], i);
                if (i < devices_.size() - 1) {
                    std::cout << std::string(80, '=') << std::endl;
                }
            }

            std::cout << std::endl;
            std::cout << BOLD << WHITE << "Last updated: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count()
                      << "ms" << RESET << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(refresh_interval_ms_));
        }
    }

    void stop() { running_ = false; }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -r, --refresh <ms>    Refresh interval in milliseconds (default: 1000)" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                    # Monitor with 1 second refresh" << std::endl;
    std::cout << "  " << program_name << " -r 500            # Monitor with 500ms refresh" << std::endl;
    std::cout << "  " << program_name << " --refresh 2000    # Monitor with 2 second refresh" << std::endl;
}

int main(int argc, char* argv[]) {
    int refresh_interval = 1000;  // Default 1 second

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-r" || arg == "--refresh") {
            if (i + 1 < argc) {
                try {
                    refresh_interval = std::stoi(argv[++i]);
                    if (refresh_interval < 100) {
                        std::cerr << "Warning: Refresh interval too low, setting to 100ms" << std::endl;
                        refresh_interval = 100;
                    }
                } catch (const std::exception&) {
                    std::cerr << "Error: Invalid refresh interval value" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: Refresh interval value required" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        MemoryMonitor monitor(refresh_interval);
        monitor.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
