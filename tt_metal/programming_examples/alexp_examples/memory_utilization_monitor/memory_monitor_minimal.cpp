// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>

// Minimal includes to avoid reflect issues
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

class MinimalMemoryMonitor {
private:
    std::atomic<bool> running_{true};
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
    MinimalMemoryMonitor(int refresh_interval_ms = 1000) : refresh_interval_ms_(refresh_interval_ms) {
        // Set up signal handler for graceful shutdown
        signal(SIGINT, [](int) {
            std::cout << "\n" << BOLD << YELLOW << "Shutting down memory monitor..." << RESET << std::endl;
            exit(0);
        });
    }

    ~MinimalMemoryMonitor() = default;

    void print_header() {
        std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << CYAN << "|" << std::setw(40) << "TT Device Memory Monitor" << std::setw(40) << "|" << RESET
                  << std::endl;
        std::cout << BOLD << CYAN << "=" << std::string(80, '=') << "=" << RESET << std::endl;
        std::cout << BOLD << WHITE << "Press Ctrl+C to exit" << RESET << std::endl;
        std::cout << std::endl;
    }

    void print_system_info() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::cout << BOLD << MAGENTA << "System Info:" << RESET << std::endl;
        std::cout << "  Time: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout << "  Refresh: " << refresh_interval_ms_ << "ms" << std::endl;

        // Get device count
        try {
            auto num_devices = GetNumAvailableDevices();
            std::cout << "  Available Devices: " << num_devices << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  Available Devices: " << RED << "Error - " << e.what() << RESET << std::endl;
        }
        std::cout << std::endl;
    }

    void print_device_info() {
        try {
            auto num_devices = GetNumAvailableDevices();
            if (num_devices == 0) {
                std::cout << RED << "No TT devices available" << RESET << std::endl;
                return;
            }

            std::cout << BOLD << BLUE << "Device Information:" << RESET << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            for (int i = 0; i < static_cast<int>(num_devices); ++i) {
                std::cout << "  Device " << i << ":" << std::endl;
                std::cout << "    Status: " << GREEN << "Available" << RESET << std::endl;
                std::cout << "    ID: " << i << std::endl;

                // Try to get PCIe device ID
                try {
                    auto pcie_id = GetPCIeDeviceID(i);
                    std::cout << "    PCIe ID: " << pcie_id << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "    PCIe ID: " << YELLOW << "Unknown" << RESET << std::endl;
                }

                std::cout << std::endl;
            }

        } catch (const std::exception& e) {
            std::cout << RED << "Error reading device information: " << e.what() << RESET << std::endl;
        }
    }

    void run() {
        std::cout << BOLD << GREEN << "Starting minimal memory monitor..." << RESET << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        while (running_) {
            print_header();
            print_system_info();
            print_device_info();

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
        MinimalMemoryMonitor monitor(refresh_interval);
        monitor.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
