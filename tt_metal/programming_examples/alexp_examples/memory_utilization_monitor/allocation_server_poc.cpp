// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Proof-of-Concept Allocation Server
// Demonstrates cross-process memory tracking using Unix domain sockets

#include <iostream>
#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>
#include <algorithm>

// TT-Metal includes for device detection
#include <tt-metalium/host_api.hpp>
#include <fstream>
#include <sstream>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"
#define MAX_DEVICES 8

// Message protocol - MUST match Python struct format exactly!
// Use __attribute__((packed)) to avoid padding issues
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
    uint8_t pad1[3];      // 3 bytes padding (explicit)
    int32_t device_id;    // 4 bytes
    uint64_t size;        // 8 bytes
    uint8_t buffer_type;  // 1 byte (0=DRAM, 1=L1, 2=L1_SMALL, 3=TRACE)
    uint8_t pad2[3];      // 3 bytes padding (explicit)
    int32_t process_id;   // 4 bytes
    uint64_t buffer_id;   // 8 bytes
    uint64_t timestamp;   // 8 bytes

    // Response fields (4x 8 bytes = 32 bytes)
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
    uint32_t is_available;  // 1 if device exists and is available, 0 otherwise
    uint32_t num_devices;   // Total number of visible devices (only in response to query for device -1)
    // Total: 1+3+4+8+1+3+4+8+8+32+8+8+4+4+4+4+4+4 = 112 bytes
};

class AllocationServer {
private:
    struct BufferInfo {
        uint64_t buffer_id;
        int device_id;
        uint64_t size;
        uint8_t buffer_type;
        pid_t owner_pid;
        std::chrono::steady_clock::time_point alloc_time;
        int ref_count;  // Track multiple allocations at the same address
    };

    struct DeviceStats {
        std::atomic<uint64_t> dram_allocated{0};
        std::atomic<uint64_t> l1_allocated{0};
        std::atomic<uint64_t> l1_small_allocated{0};
        std::atomic<uint64_t> trace_allocated{0};
        std::atomic<uint64_t> num_buffers{0};
    };

    struct DeviceInfo {
        bool is_available{false};
        uint32_t arch_type{0};  // 0=Invalid, 1=Grayskull, 2=Wormhole_B0, 3=Blackhole, 4=Quasar
        uint32_t num_dram_channels{0};
        uint32_t dram_size_per_channel{0};
        uint32_t l1_size_per_core{0};
        uint64_t total_dram_size{0};
        uint64_t total_l1_size{0};
        uint32_t grid_x{0};
        uint32_t grid_y{0};
    };

    // Composite key for buffer tracking: {device_id, buffer_id}
    // Each device has its own address space, so addresses can overlap!
    struct BufferKey {
        int device_id;
        uint64_t buffer_id;

        bool operator==(const BufferKey& other) const {
            return device_id == other.device_id && buffer_id == other.buffer_id;
        }
    };

    struct BufferKeyHash {
        std::size_t operator()(const BufferKey& k) const {
            return std::hash<int>()(k.device_id) ^ (std::hash<uint64_t>()(k.buffer_id) << 1);
        }
    };

    std::mutex registry_mutex_;
    std::unordered_map<BufferKey, BufferInfo, BufferKeyHash> allocations_;
    std::array<DeviceStats, MAX_DEVICES> device_stats_;
    std::array<DeviceInfo, MAX_DEVICES> device_info_;
    uint32_t num_available_devices_{0};

    int server_socket_;
    std::atomic<bool> running_{true};
    std::thread cleanup_thread_;

    const char* arch_to_string(tt::ARCH arch) {
        switch (arch) {
            case tt::ARCH::GRAYSKULL: return "Grayskull";
            case tt::ARCH::WORMHOLE_B0: return "Wormhole_B0";
            case tt::ARCH::BLACKHOLE: return "Blackhole";
            case tt::ARCH::QUASAR: return "Quasar";
            default: return "Unknown";
        }
    }

    void detect_devices() {
        // Use TT-Metal APIs directly for accurate device detection
        std::cout << "🔍 Device detection (using TT-Metal APIs):" << std::endl;

        try {
            // Get actual number of available devices (including remote devices)
            size_t num_available_devices = tt::tt_metal::GetNumAvailableDevices();

            if (num_available_devices == 0) {
                std::cout << "   No devices detected" << std::endl;
                std::cout << "   Server will track allocations anyway" << std::endl;
                return;
            }

            if (num_available_devices > MAX_DEVICES) {
                std::cout << "   Warning: Found " << num_available_devices << " devices, limiting to " << MAX_DEVICES
                          << std::endl;
                num_available_devices = MAX_DEVICES;
            }

            num_available_devices_ = num_available_devices;

            // Query each device for detailed information
            // Note: CreateDeviceMinimal devices should not be manually closed
            // They are managed by the device pool and will clean up automatically
            for (size_t i = 0; i < num_available_devices; ++i) {
                try {
                    // Create minimal device (lightweight, doesn't fully initialize)
                    // This only initializes enough to query basic device info
                    auto device = tt::tt_metal::CreateDeviceMinimal(i);

                    tt::ARCH arch = device->arch();
                    int num_dram_channels = device->num_dram_channels();
                    uint32_t dram_size_per_channel = device->dram_size_per_channel();
                    uint32_t l1_size_per_core = device->l1_size_per_core();
                    auto grid = device->grid_size();

                    uint64_t total_dram = (uint64_t)num_dram_channels * dram_size_per_channel;
                    uint64_t total_l1 = (uint64_t)l1_size_per_core * grid.x * grid.y;

                    // Store device info
                    device_info_[i].is_available = true;
                    device_info_[i].arch_type = static_cast<uint32_t>(arch);
                    device_info_[i].num_dram_channels = num_dram_channels;
                    device_info_[i].dram_size_per_channel = dram_size_per_channel;
                    device_info_[i].l1_size_per_core = l1_size_per_core;
                    device_info_[i].total_dram_size = total_dram;
                    device_info_[i].total_l1_size = total_l1;
                    device_info_[i].grid_x = grid.x;
                    device_info_[i].grid_y = grid.y;

                    std::cout << "   Device " << i << ": " << arch_to_string(arch) << " ("
                              << (total_dram / (1024 * 1024 * 1024)) << "GB DRAM, " << (total_l1 / (1024 * 1024))
                              << "MB L1)" << std::endl;

                    // Note: Do NOT call CloseDevice on minimal devices!
                    // They are managed internally and closing them causes segfaults.

                } catch (const std::exception& e) {
                    std::cerr << "   Warning: Could not query device " << i << ": " << e.what() << std::endl;
                    // Set some default values for this device
                    device_info_[i].is_available = false;
                }
            }

            std::cout << "   Total: " << num_available_devices_ << " device(s) detected" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "   Error during device detection: " << e.what() << std::endl;
            std::cout << "   Server will track allocations without device info" << std::endl;
            num_available_devices_ = 0;
        }
    }

public:
    AllocationServer() {
        server_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_socket_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

        unlink(TT_ALLOC_SERVER_SOCKET);

        if (bind(server_socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }

        if (listen(server_socket_, 128) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }

        // Detect available devices
        detect_devices();

        std::cout << "🚀 Allocation Server started" << std::endl;
        std::cout << "   Listening on: " << TT_ALLOC_SERVER_SOCKET << std::endl;
        std::cout << "   Press Ctrl+C to stop" << std::endl;
    }

    ~AllocationServer() { stop(); }

    void handle_allocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        // Validate buffer_type to prevent array out of bounds
        // BufferType: 0=DRAM, 1=L1, 2=SYSTEM_MEMORY(L1_SMALL), 3=L1_SMALL, 4=TRACE
        if (msg.buffer_type > 4) {
            std::cerr << "⚠ [PID " << msg.process_id << "] Invalid buffer_type " << (int)msg.buffer_type
                      << " for buffer " << msg.buffer_id << " on device " << msg.device_id << std::endl;
            return;
        }

        BufferKey key{msg.device_id, msg.buffer_id};

        // Check if buffer already exists (e.g., cached program buffers allocated multiple times)
        auto it = allocations_.find(key);
        if (it != allocations_.end()) {
            // Buffer already exists - increment ref count
            it->second.ref_count++;
            const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};
            std::cout << "✓ [PID " << msg.process_id << "] Allocated " << msg.size << " bytes of "
                      << type_name[msg.buffer_type] << " on device " << msg.device_id << " (buffer_id=0x" << std::hex
                      << msg.buffer_id << std::dec << ", ref_count=" << it->second.ref_count << ")" << std::endl;
            return;
        }

        // New buffer - create entry
        BufferInfo info{
            .buffer_id = msg.buffer_id,
            .device_id = msg.device_id,
            .size = msg.size,
            .buffer_type = msg.buffer_type,
            .owner_pid = msg.process_id,
            .alloc_time = std::chrono::steady_clock::now(),
            .ref_count = 1};

        allocations_[key] = info;

        auto& stats = device_stats_[msg.device_id];
        stats.num_buffers++;

        switch (msg.buffer_type) {
            case 0: stats.dram_allocated += msg.size; break;      // DRAM
            case 1: stats.l1_allocated += msg.size; break;        // L1
            case 2: stats.l1_small_allocated += msg.size; break;  // SYSTEM_MEMORY (treat as L1_SMALL)
            case 3: stats.l1_small_allocated += msg.size; break;  // L1_SMALL
            case 4: stats.trace_allocated += msg.size; break;     // TRACE
        }

        const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};
        const char* type_str = (msg.buffer_type <= 4) ? type_name[msg.buffer_type] : "UNKNOWN";
        std::cout << "✓ [PID " << msg.process_id << "] Allocated " << msg.size << " bytes of " << type_str
                  << " on device " << msg.device_id << " (buffer_id=0x" << std::hex << msg.buffer_id << std::dec << ")"
                  << std::endl;
    }

    void handle_deallocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        // Use the device_id from the message
        BufferKey key{msg.device_id, msg.buffer_id};
        auto it = allocations_.find(key);

        if (it == allocations_.end()) {
            // Buffer not tracked - likely allocated before tracking started
            std::cout << "⚠ [PID " << msg.process_id << "] Deallocation for unknown buffer 0x" << std::hex
                      << msg.buffer_id << std::dec << " on device " << msg.device_id
                      << " (allocated before tracking started)" << std::endl;
            return;
        }

        auto& info = it->second;

        // Decrement ref count
        info.ref_count--;

        if (info.ref_count > 0) {
            // Still has references - don't fully deallocate yet
            std::cout << "✗ [PID " << info.owner_pid << "] Freed buffer 0x" << std::hex << msg.buffer_id << std::dec
                      << " on device " << info.device_id << " (" << info.size << " bytes, ref_count=" << info.ref_count
                      << " remaining)" << std::endl;
            return;
        }

        // ref_count reached 0 - fully deallocate
        auto& stats = device_stats_[info.device_id];
        stats.num_buffers--;

        switch (info.buffer_type) {
            case 0: stats.dram_allocated -= info.size; break;      // DRAM
            case 1: stats.l1_allocated -= info.size; break;        // L1
            case 2: stats.l1_small_allocated -= info.size; break;  // SYSTEM_MEMORY (treat as L1_SMALL)
            case 3: stats.l1_small_allocated -= info.size; break;  // L1_SMALL
            case 4: stats.trace_allocated -= info.size; break;     // TRACE
        }

        std::cout << "✗ [PID " << info.owner_pid << "] Freed buffer 0x" << std::hex << msg.buffer_id << std::dec
                  << " on device " << info.device_id << " (" << info.size << " bytes, FINAL)" << std::endl;

        allocations_.erase(it);
    }

    void handle_query(int client_socket, const AllocMessage& msg) {
        AllocMessage response;
        memset(&response, 0, sizeof(response));
        response.type = AllocMessage::RESPONSE;
        response.device_id = msg.device_id;

        if (msg.device_id >= 0 && msg.device_id < MAX_DEVICES) {
            auto& stats = device_stats_[msg.device_id];
            response.dram_allocated = stats.dram_allocated.load();
            response.l1_allocated = stats.l1_allocated.load();
            response.l1_small_allocated = stats.l1_small_allocated.load();
            response.trace_allocated = stats.trace_allocated.load();
        }

        send(client_socket, &response, sizeof(response), 0);
    }

    void handle_device_info_query(int client_socket, const AllocMessage& msg) {
        AllocMessage response;
        memset(&response, 0, sizeof(response));
        response.type = AllocMessage::DEVICE_INFO_RESPONSE;
        response.device_id = msg.device_id;

        // Special case: device_id -1 means query for number of available devices
        if (msg.device_id == -1) {
            response.num_devices = num_available_devices_;
            response.is_available = (num_available_devices_ > 0) ? 1 : 0;
        } else if (msg.device_id >= 0 && msg.device_id < MAX_DEVICES) {
            // Return info for specific device
            auto& info = device_info_[msg.device_id];
            response.is_available = info.is_available ? 1 : 0;
            response.arch_type = info.arch_type;
            response.num_dram_channels = info.num_dram_channels;
            response.dram_size_per_channel = info.dram_size_per_channel;
            response.l1_size_per_core = info.l1_size_per_core;
            response.total_dram_size = info.total_dram_size;
            response.total_l1_size = info.total_l1_size;
        }

        send(client_socket, &response, sizeof(response), 0);
    }

    void cleanup_dead_processes() {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        // Check which PIDs are still alive
        std::set<pid_t> dead_pids;
        std::set<pid_t> all_pids;

        for (const auto& [key, info] : allocations_) {
            all_pids.insert(info.owner_pid);
        }

        for (pid_t pid : all_pids) {
            // Check if process exists: kill(pid, 0) returns 0 if alive, -1 if dead
            if (kill(pid, 0) != 0) {
                dead_pids.insert(pid);
            }
        }

        if (!dead_pids.empty()) {
            std::cout << "\n⚠️  Detected dead processes, cleaning up orphaned buffers..." << std::endl;

            for (pid_t dead_pid : dead_pids) {
                std::cout << "   PID " << dead_pid << " is dead, removing its buffers..." << std::endl;

                // Remove all buffers owned by this PID
                auto it = allocations_.begin();
                int removed_count = 0;
                uint64_t removed_bytes = 0;

                while (it != allocations_.end()) {
                    if (it->second.owner_pid == dead_pid) {
                        // Update stats
                        auto& stats = device_stats_[it->second.device_id];
                        stats.num_buffers--;

                        switch (it->second.buffer_type) {
                            case 0: stats.dram_allocated -= it->second.size; break;
                            case 1: stats.l1_allocated -= it->second.size; break;
                            case 2: stats.l1_small_allocated -= it->second.size; break;
                            case 3: stats.l1_small_allocated -= it->second.size; break;
                            case 4: stats.trace_allocated -= it->second.size; break;
                        }

                        removed_bytes += it->second.size;
                        removed_count++;
                        it = allocations_.erase(it);
                    } else {
                        ++it;
                    }
                }

                std::cout << "   ✓ Removed " << removed_count << " buffers (" << (removed_bytes / (1024.0 * 1024.0))
                          << " MB) from PID " << dead_pid << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void handle_dump_remaining() {
        // First, clean up any dead processes
        cleanup_dead_processes();

        std::lock_guard<std::mutex> lock(registry_mutex_);

        const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};

        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           REMAINING ALLOCATED BUFFERS                       ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "Total tracked allocations: " << allocations_.size() << "\n" << std::endl;
        std::cout.flush();

        if (allocations_.empty()) {
            std::cout << "✓ No buffers remaining - perfect cleanup!" << std::endl;
            std::cout.flush();
            return;
        }

        // Group by device
        std::map<int, std::vector<const BufferInfo*>> by_device;
        for (const auto& [key, info] : allocations_) {
            by_device[info.device_id].push_back(&info);
        }

        for (const auto& [device_id, buffers] : by_device) {
            std::cout << "Device " << device_id << ":" << std::endl;

            // Group by type
            std::map<int, std::vector<const BufferInfo*>> by_type;
            for (const auto* buf : buffers) {
                by_type[buf->buffer_type].push_back(buf);
            }

            for (const auto& [type, type_buffers] : by_type) {
                uint64_t total_size = 0;
                for (const auto* buf : type_buffers) {
                    total_size += buf->size;
                }

                std::cout << "  " << type_name[type] << ": " << type_buffers.size() << " buffers, "
                          << (total_size / (1024.0 * 1024.0)) << " MB total" << std::endl;

                // Show individual buffers if there are few
                if (type_buffers.size() <= 10) {
                    for (const auto* buf : type_buffers) {
                        std::cout << "    - Buffer 0x" << std::hex << buf->buffer_id << std::dec << ": "
                                  << (buf->size / 1024.0) << " KB"
                                  << " (PID " << buf->owner_pid << ", ref_count=" << buf->ref_count << ")" << std::endl;
                    }
                }
            }
            std::cout << std::endl;
        }

        std::cout << "Total remaining buffers: " << allocations_.size() << "\n" << std::endl;
        std::cout.flush();
    }

    void handle_client(int client_socket) {
        AllocMessage msg;

        while (running_) {
            ssize_t n = recv(client_socket, &msg, sizeof(msg), 0);
            if (n <= 0) {
                break;
            }

            switch (msg.type) {
                case AllocMessage::ALLOC: handle_allocation(msg); break;
                case AllocMessage::FREE: handle_deallocation(msg); break;
                case AllocMessage::QUERY: handle_query(client_socket, msg); break;
                case AllocMessage::DEVICE_INFO_QUERY: handle_device_info_query(client_socket, msg); break;
                case AllocMessage::DUMP_REMAINING:
                    std::cout << "📋 Received DUMP_REMAINING request..." << std::endl;
                    std::cout.flush();
                    handle_dump_remaining();
                    std::cout << "📋 DUMP_REMAINING complete." << std::endl;
                    std::cout.flush();
                    break;
                default: std::cerr << "Unknown message type: " << (int)msg.type << std::endl; break;
            }
        }

        close(client_socket);
    }

    void background_cleanup_loop() {
        std::cout << "🔄 Background cleanup thread started (checking every 10s)" << std::endl;

        while (running_) {
            // Sleep for 10 seconds
            int sleep_seconds = 5;
            for (int i = 0; i < sleep_seconds && running_; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            if (!running_) {
                break;
            }

            // Check for dead processes
            cleanup_dead_processes();
        }

        std::cout << "🔄 Background cleanup thread stopped" << std::endl;
    }

    void run() {
        // Start background cleanup thread
        cleanup_thread_ = std::thread(&AllocationServer::background_cleanup_loop, this);

        while (running_) {
            int client_socket = accept(server_socket_, nullptr, nullptr);
            if (client_socket < 0) {
                if (running_) {
                    std::cerr << "Accept failed" << std::endl;
                }
                continue;
            }

            std::thread(&AllocationServer::handle_client, this, client_socket).detach();
        }
    }

    void stop() {
        running_ = false;

        // Wait for cleanup thread to finish
        if (cleanup_thread_.joinable()) {
            cleanup_thread_.join();
        }

        if (server_socket_ >= 0) {
            close(server_socket_);
        }
        unlink(TT_ALLOC_SERVER_SOCKET);
        std::cout << "\n🛑 Allocation Server stopped" << std::endl;
    }

    void print_stats() {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        std::cout << "\n📊 Current Statistics:" << std::endl;
        std::cout.flush();

        for (int i = 0; i < MAX_DEVICES; i++) {
            auto& stats = device_stats_[i];
            uint64_t total =
                stats.dram_allocated + stats.l1_allocated + stats.l1_small_allocated + stats.trace_allocated;

            if (total > 0 || stats.num_buffers > 0) {
                std::cout << "  Device " << i << ":" << std::endl;
                std::cout << "    Buffers: " << stats.num_buffers << std::endl;
                std::cout << "    DRAM: " << stats.dram_allocated << " bytes (" << (stats.dram_allocated / 1024.0)
                          << " KB)" << std::endl;
                std::cout << "    L1: " << stats.l1_allocated << " bytes (" << (stats.l1_allocated / 1024.0) << " KB)"
                          << std::endl;
                std::cout << "    Total: " << total << " bytes" << std::endl;
            }
        }

        std::cout << "  Active allocations: " << allocations_.size() << std::endl;
        std::cout.flush();
    }
};

AllocationServer* g_server = nullptr;

void signal_handler(int sig) {
    if (g_server) {
        g_server->print_stats();
        g_server->stop();
    }
    exit(0);
}

int main() {
    // Disable stdout buffering to ensure immediate output
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        AllocationServer server;
        g_server = &server;

        // Print stats every 10 seconds in background
        std::thread stats_thread([&]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                server.print_stats();
            }
        });
        stats_thread.detach();

        server.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
