// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Memory monitor with integrated allocation test
// This version allocates buffers from within the same process,
// so you can see real-time memory utilization changes

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include <vector>
#include <sstream>
#include <memory>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Same MemoryMonitor class but with test mode
class MemoryMonitorWithTest {
private:
    std::atomic<bool> running_{true};
    std::vector<IDevice*> devices_;
    std::vector<std::shared_ptr<Buffer>> test_buffers_;  // For test allocations
    int refresh_interval_ms_;
    bool test_mode_;
    int test_iteration_{0};

    // ANSI color codes
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
    MemoryMonitorWithTest(int refresh_interval_ms = 1000, bool test_mode = false) :
        devices_(), refresh_interval_ms_(refresh_interval_ms), test_mode_(test_mode) {
        signal(SIGINT, [](int) {
            std::cout << "\n" << BOLD << YELLOW << "Shutting down..." << RESET << std::endl;
            exit(0);
        });
    }

    bool initialize_devices() {
        try {
            auto num_devices = GetNumAvailableDevices();
            if (num_devices == 0) {
                std::cerr << RED << "Error: No TT devices available" << RESET << std::endl;
                return false;
            }

            std::cout << GREEN << "Found " << num_devices << " TT device(s)" << RESET << std::endl;

            try {
                auto device = CreateDevice(0);
                if (device && device->is_initialized()) {
                    devices_.push_back(device);
                    std::cout << GREEN << "Device 0 initialized successfully" << RESET << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << RED << "Error initializing device 0: " << e.what() << RESET << std::endl;
            }

            return !devices_.empty();
        } catch (const std::exception& e) {
            std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
            return false;
        }
    }

    void allocate_test_buffer(size_t size_bytes, BufferType buffer_type, const std::string& name) {
        if (devices_.empty()) {
            return;
        }

        try {
            // Use smaller page sizes for interleaved buffers
            size_t page_size = (buffer_type == BufferType::L1) ? 2048 : 2048;  // 2KB pages

            InterleavedBufferConfig config{
                .device = devices_[0], .size = size_bytes, .page_size = page_size, .buffer_type = buffer_type};

            auto buffer = CreateBuffer(config);
            test_buffers_.push_back(buffer);

            std::cout << GREEN << "✓ Allocated " << name << " (" << format_bytes(size_bytes) << ")" << RESET
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << RED << "Failed to allocate " << name << ": " << e.what() << RESET << std::endl;
        }
    }

    void deallocate_test_buffers(int count) {
        if (count > test_buffers_.size()) {
            count = test_buffers_.size();
        }

        for (int i = 0; i < count && !test_buffers_.empty(); ++i) {
            test_buffers_.pop_back();  // Destructor will deallocate
        }

        std::cout << YELLOW << "✓ Deallocated " << count << " buffer(s)" << RESET << std::endl;
    }

    void run_test_cycle() {
        test_iteration_++;

        std::cout << "\n" << BOLD << CYAN << "═══ Test Cycle " << test_iteration_ << " ═══" << RESET << std::endl;

        switch (test_iteration_) {
            case 3:  // After 3 refreshes, allocate L1
                std::cout << BOLD << "→ Allocating ~2MB L1 memory..." << RESET << std::endl;
                for (int i = 0; i < 4; ++i) {
                    allocate_test_buffer(512 * 1024, BufferType::L1, "L1 buffer");  // 512KB each
                }
                break;

            case 6:  // After 6 refreshes, allocate DRAM
                std::cout << BOLD << "→ Allocating ~100MB DRAM memory..." << RESET << std::endl;
                for (int i = 0; i < 4; ++i) {
                    allocate_test_buffer(25 * 1024 * 1024, BufferType::DRAM, "DRAM buffer");  // 25MB each
                }
                break;

            case 9:  // After 9 refreshes, allocate more L1
                std::cout << BOLD << "→ Allocating ~4MB more L1 memory..." << RESET << std::endl;
                for (int i = 0; i < 8; ++i) {
                    allocate_test_buffer(512 * 1024, BufferType::L1, "L1 buffer");  // 512KB each
                }
                break;

            case 12:  // Deallocate half L1
                std::cout << BOLD << "→ Deallocating some buffers..." << RESET << std::endl;
                deallocate_test_buffers(6);
                break;

            case 15:  // Deallocate all
                std::cout << BOLD << "→ Deallocating all buffers..." << RESET << std::endl;
                deallocate_test_buffers(test_buffers_.size());
                test_iteration_ = 0;  // Reset cycle
                break;
        }

        std::cout << std::endl;
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
        std::cout << "\033[2J\033[H";
        std::cout << BOLD << CYAN << "═══════════════════════════════════════════════════════════════════════════════"
                  << RESET << std::endl;
        std::cout << BOLD << CYAN << "║" << std::setw(45);
        if (test_mode_) {
            std::cout << "TT Memory Monitor (TEST MODE)";
        } else {
            std::cout << "TT Memory Monitor";
        }
        std::cout << std::setw(34) << "║" << RESET << std::endl;
        std::cout << BOLD << CYAN << "═══════════════════════════════════════════════════════════════════════════════"
                  << RESET << std::endl;
        std::cout << BOLD << WHITE << "Press Ctrl+C to exit" << RESET << std::endl;
        if (test_mode_) {
            std::cout << YELLOW << "Test mode: Automatically allocating/deallocating memory" << RESET << std::endl;
        }
        std::cout << std::endl;
    }

    void print_memory_buffer_stats(IDevice* device, BufferType buffer_type, const std::string& buffer_name) {
        try {
            if (!device || !device->is_initialized()) {
                return;
            }

            const auto& allocator = device->allocator();
            if (!allocator) {
                return;
            }

            auto stats = allocator->get_statistics(buffer_type);
            auto num_banks = allocator->get_num_banks(buffer_type);

            size_t total_bytes = stats.total_allocatable_size_bytes * num_banks;
            size_t allocated_bytes = stats.total_allocated_bytes * num_banks;
            // size_t free_bytes = stats.total_free_bytes * num_banks;

            double utilization = calculate_utilization_percentage(allocated_bytes, total_bytes);
            std::string color = get_utilization_color(utilization);

            std::cout << "  " << BOLD << buffer_name << ":" << RESET << " " << color << format_bytes(allocated_bytes)
                      << "/" << format_bytes(total_bytes) << " (" << std::fixed << std::setprecision(1) << utilization
                      << "%)" << RESET;

            // Mini bar
            std::cout << " [";
            int bar_width = 20;
            int filled = static_cast<int>((utilization / 100.0) * bar_width);
            for (int j = 0; j < bar_width; ++j) {
                if (j < filled) {
                    std::cout << color << "█" << RESET;
                } else {
                    std::cout << "░";
                }
            }
            std::cout << "]" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  " << buffer_name << ": " << RED << "Error" << RESET << std::endl;
        }
    }

    void print_compact_status() {
        if (devices_.empty()) {
            return;
        }
        auto device = devices_[0];

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::cout << MAGENTA << "Time: " << std::put_time(&tm, "%H:%M:%S") << " | Refresh: " << refresh_interval_ms_
                  << "ms";
        if (test_mode_) {
            std::cout << " | Test Cycle: " << test_iteration_;
        }
        std::cout << RESET << std::endl;
        std::cout << std::endl;

        std::cout << BOLD << "Device 0:" << RESET << std::endl;
        print_memory_buffer_stats(device, BufferType::DRAM, "DRAM  ");
        print_memory_buffer_stats(device, BufferType::L1, "L1    ");
        print_memory_buffer_stats(device, BufferType::L1_SMALL, "L1_SMALL");
        print_memory_buffer_stats(device, BufferType::TRACE, "TRACE ");
    }

    void run() {
        if (!initialize_devices()) {
            return;
        }

        std::cout << BOLD << GREEN << "Starting memory monitor..." << RESET << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        while (running_) {
            print_header();

            if (test_mode_) {
                run_test_cycle();
            }

            print_compact_status();

            std::this_thread::sleep_for(std::chrono::milliseconds(refresh_interval_ms_));
        }
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -r, --refresh <ms>    Refresh interval in milliseconds (default: 1000)" << std::endl;
    std::cout << "  -t, --test           Enable test mode (auto allocate/deallocate)" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                   # Normal monitoring" << std::endl;
    std::cout << "  " << program_name << " -t               # Test mode with auto allocation" << std::endl;
    std::cout << "  " << program_name << " -t -r 500        # Test mode with 500ms refresh" << std::endl;
}

int main(int argc, char* argv[]) {
    int refresh_interval = 1000;
    bool test_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-t" || arg == "--test") {
            test_mode = true;
        } else if (arg == "-r" || arg == "--refresh") {
            if (i + 1 < argc) {
                refresh_interval = std::stoi(argv[++i]);
                if (refresh_interval < 100) {
                    refresh_interval = 100;
                }
            }
        }
    }

    try {
        MemoryMonitorWithTest monitor(refresh_interval, test_mode);
        monitor.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
