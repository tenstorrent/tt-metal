// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Monitor client that queries the allocation server for real-time stats

#include <iostream>
#include <iomanip>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <sstream>
#include <vector>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t {
        ALLOC = 1,
        FREE = 2,
        QUERY = 3,
        RESPONSE = 4,
        DUMP_REMAINING = 5,
        DEVICE_INFO_QUERY = 6,
        DEVICE_INFO_RESPONSE = 7
    };

    Type type;            // 1 byte
    uint8_t pad1[3];      // 3 bytes padding
    int32_t device_id;    // 4 bytes
    uint64_t size;        // 8 bytes
    uint8_t buffer_type;  // 1 byte
    uint8_t pad2[3];      // 3 bytes padding
    int32_t process_id;   // 4 bytes
    uint64_t buffer_id;   // 8 bytes
    uint64_t timestamp;   // 8 bytes

    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;

    // Device info fields (for DEVICE_INFO_RESPONSE)
    uint64_t total_dram_size;
    uint64_t total_l1_size;
    uint32_t arch_type;  // 0=Invalid, 1=Grayskull, 2=Wormhole_B0, 3=Blackhole, 4=Quasar
    uint32_t num_dram_channels;
    uint32_t dram_size_per_channel;
    uint32_t l1_size_per_core;
    uint32_t is_available;
    uint32_t num_devices;
    // Total: 112 bytes
};

class MonitorClient {
private:
    int socket_fd_;

    // Device info cache
    struct DeviceInfo {
        bool is_available{false};
        uint32_t arch_type{0};
        uint64_t total_dram_size{0};
        uint64_t total_l1_size{0};
        uint32_t num_dram_channels{0};
        uint32_t dram_size_per_channel{0};
        uint32_t l1_size_per_core{0};
    };
    std::vector<DeviceInfo> device_info_cache_;
    uint32_t num_devices_{0};

    // ANSI colors
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";

public:
    MonitorClient() {
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to connect to allocation server");
        }

        // Query device info on connection
        query_device_info();
    }

    void query_device_info() {
        // First query for number of devices
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::DEVICE_INFO_QUERY;
        query.device_id = -1;  // Special: query for device count

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);

        num_devices_ = response.num_devices;

        // Query info for each device
        device_info_cache_.resize(num_devices_);
        for (uint32_t i = 0; i < num_devices_; ++i) {
            memset(&query, 0, sizeof(query));
            query.type = AllocMessage::DEVICE_INFO_QUERY;
            query.device_id = i;

            send(socket_fd_, &query, sizeof(query), 0);
            recv(socket_fd_, &response, sizeof(response), 0);

            if (response.is_available) {
                device_info_cache_[i].is_available = true;
                device_info_cache_[i].arch_type = response.arch_type;
                device_info_cache_[i].total_dram_size = response.total_dram_size;
                device_info_cache_[i].total_l1_size = response.total_l1_size;
                device_info_cache_[i].num_dram_channels = response.num_dram_channels;
                device_info_cache_[i].dram_size_per_channel = response.dram_size_per_channel;
                device_info_cache_[i].l1_size_per_core = response.l1_size_per_core;
            }
        }
    }

    uint32_t get_num_devices() const { return num_devices_; }

    const DeviceInfo* get_device_info(int device_id) const {
        if (device_id >= 0 && device_id < static_cast<int>(device_info_cache_.size())) {
            return &device_info_cache_[device_id];
        }
        return nullptr;
    }

    ~MonitorClient() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }

    AllocMessage query_device(int device_id) {
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::QUERY;
        query.device_id = device_id;

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);

        return response;
    }

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

    void display_stats(int device_id, const AllocMessage& stats) {
        // Get actual device capabilities
        const DeviceInfo* dev_info = get_device_info(device_id);
        uint64_t total_dram = 12ULL * 1024 * 1024 * 1024;  // Default fallback
        uint64_t total_l1 = 75 * 1024 * 1024;              // Default fallback
        const char* arch_name = "Unknown";

        if (dev_info && dev_info->is_available) {
            total_dram = dev_info->total_dram_size;
            total_l1 = dev_info->total_l1_size;

            // Get architecture name
            switch (dev_info->arch_type) {
                case 1: arch_name = "Grayskull"; break;
                case 2: arch_name = "Wormhole_B0"; break;
                case 3: arch_name = "Blackhole"; break;
                case 4: arch_name = "Quasar"; break;
                default: arch_name = "Unknown"; break;
            }
        }

        std::cout << BOLD << CYAN << "Device " << device_id;
        if (dev_info && dev_info->is_available) {
            std::cout << " (" << arch_name << ")";
        }
        std::cout << RESET << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // DRAM
        double dram_util = calculate_utilization(stats.dram_allocated, total_dram);
        std::cout << "  DRAM:   " << std::setw(12) << format_bytes(stats.dram_allocated) << " / " << std::setw(12)
                  << format_bytes(total_dram) << "  ";
        print_bar(dram_util, 25);
        std::cout << std::endl;

        // L1
        double l1_util = calculate_utilization(stats.l1_allocated, total_l1);
        std::cout << "  L1:     " << std::setw(12) << format_bytes(stats.l1_allocated) << " / " << std::setw(12)
                  << format_bytes(total_l1) << "  ";
        print_bar(l1_util, 25);
        std::cout << std::endl;

        // L1_SMALL
        if (stats.l1_small_allocated > 0) {
            std::cout << "  L1_SMALL: " << format_bytes(stats.l1_small_allocated) << std::endl;
        }

        // TRACE
        if (stats.trace_allocated > 0) {
            std::cout << "  TRACE:    " << format_bytes(stats.trace_allocated) << std::endl;
        }

        // Show device details if available
        if (dev_info && dev_info->is_available) {
            std::cout << "  " << BOLD << "Device Info:" << RESET << " " << dev_info->num_dram_channels
                      << " DRAM channels, " << format_bytes(dev_info->l1_size_per_core) << " L1/core" << std::endl;
        }

        std::cout << std::endl;
    }

    void display_multi_device_stats(const std::vector<int>& device_ids) {
        std::vector<AllocMessage> all_stats;

        // Query all devices
        for (int device_id : device_ids) {
            try {
                all_stats.push_back(query_device(device_id));
            } catch (...) {
                // Skip devices that fail to query
                AllocMessage empty;
                memset(&empty, 0, sizeof(empty));
                empty.device_id = device_id;
                all_stats.push_back(empty);
            }
        }

        // Display all devices
        for (size_t i = 0; i < device_ids.size(); ++i) {
            display_stats(device_ids[i], all_stats[i]);
        }
    }
};

int main(int argc, char* argv[]) {
    int refresh_ms = 1000;
    std::vector<int> device_ids;
    bool auto_detect = false;
    bool explicit_devices = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) {
            refresh_ms = std::stoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            device_ids.push_back(std::stoi(argv[++i]));
            explicit_devices = true;
        } else if (arg == "-a" || arg == "--all") {
            // Auto-detect all available devices
            auto_detect = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -r <ms>        Refresh interval (default: 1000)" << std::endl;
            std::cout << "  -d <id>        Device ID to monitor (can specify multiple times)" << std::endl;
            std::cout << "  -a, --all      Auto-detect and monitor all available devices" << std::endl;
            std::cout << "  -h, --help     Show this help" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << "                   # Auto-detect devices" << std::endl;
            std::cout << "  " << argv[0] << " -d 0               # Monitor device 0" << std::endl;
            std::cout << "  " << argv[0] << " -d 0 -d 1 -d 2     # Monitor devices 0, 1, 2" << std::endl;
            std::cout << "  " << argv[0] << " -a -r 500          # Auto-detect all devices, 500ms refresh" << std::endl;
            return 0;
        }
    }

    try {
        MonitorClient client;

        // Auto-detect devices if requested or no explicit devices given
        if (auto_detect || (!explicit_devices && device_ids.empty())) {
            uint32_t num_devices = client.get_num_devices();
            if (num_devices > 0) {
                device_ids.clear();
                for (uint32_t i = 0; i < num_devices; ++i) {
                    device_ids.push_back(i);
                }
                std::cout << "🔍 Auto-detected " << num_devices << " device(s)" << std::endl;
            } else {
                std::cout << "⚠️  No devices detected, defaulting to device 0" << std::endl;
                device_ids.push_back(0);
            }
        }

        std::cout << "\n📊 Allocation Server Monitor" << std::endl;
        std::cout << "   Monitoring device(s): ";
        for (size_t i = 0; i < device_ids.size(); ++i) {
            std::cout << device_ids[i];
            if (i < device_ids.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
        std::cout << "   Refresh: " << refresh_ms << "ms" << std::endl;
        std::cout << "   Press Ctrl+C to exit\n" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

        while (true) {
            // Clear screen
            std::cout << "\033[2J\033[H";

            // Get current time
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto tm = *std::localtime(&time_t);

            std::cout << "═══════════════════════════════════════════════════════════════════════" << std::endl;
            std::cout << "  Cross-Process Memory Monitor (via Allocation Server)" << std::endl;
            std::cout << "═══════════════════════════════════════════════════════════════════════" << std::endl;
            std::cout << "Time: " << std::put_time(&tm, "%H:%M:%S") << " | Refresh: " << refresh_ms << "ms"
                      << std::endl;
            std::cout << std::endl;

            // Query and display all devices
            if (device_ids.size() == 1) {
                // Single device - use original display
                auto stats = client.query_device(device_ids[0]);
                client.display_stats(device_ids[0], stats);
            } else {
                // Multiple devices - use new multi-device display
                client.display_multi_device_stats(device_ids);
            }

            std::cout << "💡 TIP: This monitor sees allocations from ALL processes!" << std::endl;
            if (device_ids.size() == 1 && client.get_num_devices() > 1) {
                std::cout << "   Add more devices with: -d 0 -d 1 -d 2  or use -a to auto-detect all" << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "\nMake sure allocation_server_poc is running!" << std::endl;
        std::cerr << "  Terminal 1: ./allocation_server_poc" << std::endl;
        std::cerr << "  Terminal 2: ./allocation_monitor_client" << std::endl;
        return 1;
    }

    return 0;
}
