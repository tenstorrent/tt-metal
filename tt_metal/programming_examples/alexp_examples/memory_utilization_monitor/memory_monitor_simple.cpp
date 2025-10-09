// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include <sstream>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

class SimpleMemoryMonitor {
private:
    std::atomic<bool> running_{true};
    std::vector<IDevice*> devices_;
    std::vector<std::shared_ptr<distributed::MeshDevice>> mesh_devices_;
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
    SimpleMemoryMonitor(int refresh_interval_ms = 1000) : refresh_interval_ms_(refresh_interval_ms) {
        // Set up signal handler for graceful shutdown
        signal(SIGINT, [](int) {
            std::cout << "\n" << BOLD << YELLOW << "Shutting down memory monitor..." << RESET << std::endl;
            exit(0);
        });
    }

    ~SimpleMemoryMonitor() = default;

    bool initialize_devices() {
        try {
            auto num_devices = GetNumAvailableDevices();
            if (num_devices == 0) {
                std::cerr << RED << "Error: No TT devices available" << RESET << std::endl;
                return false;
            }

            std::cout << GREEN << "Found " << num_devices << " TT device(s)" << RESET << std::endl;

            // Create mesh devices for each available device
            for (int i = 0; i < static_cast<int>(num_devices); ++i) {
                try {
                    auto mesh_device = distributed::MeshDevice::create_unit_mesh(i);
                    mesh_devices_.push_back(mesh_device);
                    devices_.push_back(mesh_device.get());
                } catch (const std::exception& e) {
                    std::cerr << YELLOW << "Warning: Could not initialize device " << i << ": " << e.what() << RESET
                              << std::endl;
                }
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

    void print_header() {
        std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << CYAN << "|" << std::setw(40) << "TT Device Memory Monitor" << std::setw(40) << "|" << RESET
                  << std::endl;
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << WHITE << "Press Ctrl+C to exit" << RESET << std::endl;
        std::cout << std::endl;
    }

    void print_device_status(IDevice* device, int device_index) {
        std::cout << BOLD << BLUE << "Device " << device_index << " (ID: " << device->id() << ")" << RESET << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        try {
            // Get basic device information
            std::cout << "  " << BOLD << "Device Information:" << RESET << std::endl;
            std::cout << "    Architecture: " << static_cast<int>(device->arch()) << std::endl;
            std::cout << "    Build ID: " << device->build_id() << std::endl;
            std::cout << "    Hardware CQs: " << static_cast<int>(device->num_hw_cqs()) << std::endl;
            std::cout << "    Initialized: " << (device->is_initialized() ? "Yes" : "No") << std::endl;

            // Memory information
            std::cout << "    DRAM Channels: " << device->num_dram_channels() << std::endl;
            std::cout << "    L1 Size per Core: " << format_bytes(device->l1_size_per_core()) << std::endl;
            std::cout << "    DRAM Size per Channel: " << format_bytes(device->dram_size_per_channel()) << std::endl;

            // Grid information
            auto grid_size = device->grid_size();
            auto logical_grid_size = device->logical_grid_size();
            auto dram_grid_size = device->dram_grid_size();

            std::cout << "    Grid Size: " << grid_size.x << "x" << grid_size.y << std::endl;
            std::cout << "    Logical Grid: " << logical_grid_size.x << "x" << logical_grid_size.y << std::endl;
            std::cout << "    DRAM Grid: " << dram_grid_size.x << "x" << dram_grid_size.y << std::endl;

            // Calculate total memory
            size_t total_l1_memory = device->l1_size_per_core() * grid_size.x * grid_size.y;
            size_t total_dram_memory = device->dram_size_per_channel() * device->num_dram_channels();

            std::cout << std::endl;
            std::cout << "  " << BOLD << "Memory Summary:" << RESET << std::endl;
            std::cout << "    Total L1 Memory: " << format_bytes(total_l1_memory) << std::endl;
            std::cout << "    Total DRAM Memory: " << format_bytes(total_dram_memory) << std::endl;
            std::cout << "    Total Device Memory: " << format_bytes(total_l1_memory + total_dram_memory) << std::endl;

            // Simple status indicator
            std::cout << std::endl;
            std::cout << "  " << BOLD << "Status:" << RESET << " ";
            if (device->is_initialized()) {
                std::cout << GREEN << "✓ Ready" << RESET << std::endl;
            } else {
                std::cout << RED << "✗ Not Ready" << RESET << std::endl;
            }
            std::cout << std::endl;

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
                print_device_status(devices_[i], i);
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
        SimpleMemoryMonitor monitor(refresh_interval);
        monitor.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
