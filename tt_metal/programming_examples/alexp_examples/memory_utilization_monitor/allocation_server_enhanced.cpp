// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Enhanced Allocation Server with Stack Trace Capture
// Tracks buffer allocation/deallocation history to debug leaks and double-frees

#include <iostream>
#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <deque>
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
#include <sstream>
#include <iomanip>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"
#define MAX_DEVICES 8
#define MAX_HISTORY_PER_BUFFER 50  // Keep last 50 events per buffer

// Message protocol
struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t { ALLOC = 1, FREE = 2, QUERY = 3, RESPONSE = 4, DUMP_REMAINING = 5 };

    Type type;
    uint8_t pad1[3];
    int32_t device_id;
    uint64_t size;
    uint8_t buffer_type;
    uint8_t pad2[3];
    int32_t process_id;
    uint64_t buffer_id;
    uint64_t timestamp;

    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
};

class AllocationServer {
private:
    // Event types for history tracking
    enum EventType { ALLOC_EVENT, FREE_EVENT, REUSE_EVENT };

    struct BufferEvent {
        EventType type;
        uint64_t size;
        pid_t owner_pid;
        std::chrono::steady_clock::time_point timestamp;
        int ref_count_after;  // ref_count after this event

        std::string to_string() const {
            const char* type_str[] = {"ALLOC", "FREE", "REUSE"};
            std::stringstream ss;
            ss << type_str[type] << " size=" << size << " pid=" << owner_pid << " ref=" << ref_count_after;
            return ss.str();
        }
    };

    struct BufferInfo {
        uint64_t buffer_id;
        int device_id;
        uint64_t size;
        uint8_t buffer_type;
        pid_t owner_pid;
        std::chrono::steady_clock::time_point alloc_time;
        int ref_count;

        // History of events for this buffer
        std::deque<BufferEvent> history;
    };

    struct DeviceStats {
        std::atomic<uint64_t> dram_allocated{0};
        std::atomic<uint64_t> l1_allocated{0};
        std::atomic<uint64_t> l1_small_allocated{0};
        std::atomic<uint64_t> trace_allocated{0};
        std::atomic<uint64_t> num_buffers{0};
    };

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

    // Track history even for deallocated buffers
    struct BufferLifecycleHistory {
        int device_id;
        uint64_t buffer_id;
        uint8_t buffer_type;
        std::deque<BufferEvent> history;
        int total_allocs;
        int total_frees;

        bool is_leaked() const { return total_allocs > total_frees; }
        bool is_double_freed() const { return total_frees > total_allocs; }
    };

    std::mutex registry_mutex_;
    std::unordered_map<BufferKey, BufferInfo, BufferKeyHash> allocations_;
    std::unordered_map<BufferKey, BufferLifecycleHistory, BufferKeyHash> full_history_;
    std::array<DeviceStats, MAX_DEVICES> device_stats_;

    int server_socket_;
    std::atomic<bool> running_{true};
    std::thread cleanup_thread_;

    void add_event_to_history(BufferKey key, const BufferEvent& event) {
        auto& hist = full_history_[key];
        hist.device_id = key.device_id;
        hist.buffer_id = key.buffer_id;

        if (event.type == ALLOC_EVENT || event.type == REUSE_EVENT) {
            hist.total_allocs++;
        } else if (event.type == FREE_EVENT) {
            hist.total_frees++;
        }

        hist.history.push_back(event);
        if (hist.history.size() > MAX_HISTORY_PER_BUFFER) {
            hist.history.pop_front();
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

        std::cout << "🚀 Enhanced Allocation Server started (with history tracking)" << std::endl;
        std::cout << "   Listening on: " << TT_ALLOC_SERVER_SOCKET << std::endl;
    }

    ~AllocationServer() { stop(); }

    void handle_allocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        if (msg.buffer_type > 4) {
            std::cerr << "⚠ [PID " << msg.process_id << "] Invalid buffer_type " << (int)msg.buffer_type << std::endl;
            return;
        }

        BufferKey key{msg.device_id, msg.buffer_id};
        auto it = allocations_.find(key);

        BufferEvent event{
            .type = (it == allocations_.end()) ? ALLOC_EVENT : REUSE_EVENT,
            .size = msg.size,
            .owner_pid = msg.process_id,
            .timestamp = std::chrono::steady_clock::now(),
            .ref_count_after = 0};

        if (it != allocations_.end()) {
            // Buffer already exists - check if size changed
            if (it->second.size != msg.size) {
                std::cout << "⚠ [PID " << msg.process_id << "] Buffer reused with DIFFERENT size: "
                          << "old=" << it->second.size << " new=" << msg.size << " (buffer_id=" << msg.buffer_id
                          << ", device=" << msg.device_id << ")" << std::endl;
                // Update size
                it->second.size = msg.size;
            }
            it->second.ref_count++;
            event.ref_count_after = it->second.ref_count;
            add_event_to_history(key, event);

            const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};
            std::cout << "✓ [PID " << msg.process_id << "] Allocated " << msg.size << " bytes of "
                      << type_name[msg.buffer_type] << " on device " << msg.device_id << " (buffer_id=0x" << std::hex
                      << msg.buffer_id << std::dec << ", ref_count=" << it->second.ref_count << ")" << std::endl;
            return;
        }

        // New buffer
        BufferInfo info{
            .buffer_id = msg.buffer_id,
            .device_id = msg.device_id,
            .size = msg.size,
            .buffer_type = msg.buffer_type,
            .owner_pid = msg.process_id,
            .alloc_time = std::chrono::steady_clock::now(),
            .ref_count = 1};

        event.ref_count_after = 1;
        add_event_to_history(key, event);

        allocations_[key] = info;

        auto& stats = device_stats_[msg.device_id];
        stats.num_buffers++;

        switch (msg.buffer_type) {
            case 0: stats.dram_allocated += msg.size; break;
            case 1: stats.l1_allocated += msg.size; break;
            case 2: stats.l1_small_allocated += msg.size; break;
            case 3: stats.l1_small_allocated += msg.size; break;
            case 4: stats.trace_allocated += msg.size; break;
        }

        const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};
        std::cout << "✓ [PID " << msg.process_id << "] Allocated " << msg.size << " bytes of "
                  << type_name[msg.buffer_type] << " on device " << msg.device_id << " (buffer_id=0x" << std::hex
                  << msg.buffer_id << std::dec << ")" << std::endl;
    }

    void handle_deallocation(const AllocMessage& msg) {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        BufferKey key{msg.device_id, msg.buffer_id};
        auto it = allocations_.find(key);

        BufferEvent event{
            .type = FREE_EVENT,
            .size = (it != allocations_.end()) ? it->second.size : 0,
            .owner_pid = msg.process_id,
            .timestamp = std::chrono::steady_clock::now(),
            .ref_count_after = 0};

        if (it == allocations_.end()) {
            // Check history to see if this is a double-free
            auto hist_it = full_history_.find(key);
            if (hist_it != full_history_.end()) {
                auto& hist = hist_it->second;
                std::cout << "⚠ [PID " << msg.process_id << "] Deallocation for ALREADY FREED buffer 0x" << std::hex
                          << msg.buffer_id << std::dec << " on device " << msg.device_id
                          << " (allocs=" << hist.total_allocs << " frees=" << (hist.total_frees + 1) << ")"
                          << std::endl;
                std::cout << "   Last 5 events for this buffer:" << std::endl;
                int count = 0;
                for (auto rev_it = hist.history.rbegin(); rev_it != hist.history.rend() && count < 5;
                     ++rev_it, ++count) {
                    std::cout << "     " << rev_it->to_string() << std::endl;
                }
            } else {
                std::cout << "⚠ [PID " << msg.process_id << "] Deallocation for unknown buffer 0x" << std::hex
                          << msg.buffer_id << std::dec << " on device " << msg.device_id
                          << " (never tracked or allocated before server started)" << std::endl;
            }

            add_event_to_history(key, event);
            return;
        }

        auto& info = it->second;
        info.ref_count--;
        event.ref_count_after = info.ref_count;
        add_event_to_history(key, event);

        if (info.ref_count > 0) {
            std::cout << "✗ [PID " << info.owner_pid << "] Freed buffer 0x" << std::hex << msg.buffer_id << std::dec
                      << " on device " << info.device_id << " (" << info.size << " bytes, ref_count=" << info.ref_count
                      << " remaining)" << std::endl;
            return;
        }

        // ref_count reached 0
        auto& stats = device_stats_[info.device_id];
        stats.num_buffers--;

        switch (info.buffer_type) {
            case 0: stats.dram_allocated -= info.size; break;
            case 1: stats.l1_allocated -= info.size; break;
            case 2: stats.l1_small_allocated -= info.size; break;
            case 3: stats.l1_small_allocated -= info.size; break;
            case 4: stats.trace_allocated -= info.size; break;
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

    void cleanup_dead_processes() {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        std::set<pid_t> dead_pids;
        std::set<pid_t> all_pids;

        for (const auto& [key, info] : allocations_) {
            all_pids.insert(info.owner_pid);
        }

        for (pid_t pid : all_pids) {
            if (kill(pid, 0) != 0) {
                dead_pids.insert(pid);
            }
        }

        if (!dead_pids.empty()) {
            std::cout << "\n⚠️  Detected dead processes, cleaning up orphaned buffers..." << std::endl;

            for (pid_t dead_pid : dead_pids) {
                std::cout << "   PID " << dead_pid << " is dead, removing its buffers..." << std::endl;

                auto it = allocations_.begin();
                int removed_count = 0;
                uint64_t removed_bytes = 0;

                while (it != allocations_.end()) {
                    if (it->second.owner_pid == dead_pid) {
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
        cleanup_dead_processes();

        std::lock_guard<std::mutex> lock(registry_mutex_);

        const char* type_name[] = {"DRAM", "L1", "SYSTEM_MEMORY", "L1_SMALL", "TRACE"};

        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           REMAINING ALLOCATED BUFFERS                       ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "Total tracked allocations: " << allocations_.size() << "\n" << std::endl;

        if (allocations_.empty()) {
            std::cout << "✓ No buffers remaining - perfect cleanup!" << std::endl;

            // But check for double-frees in history
            int double_free_count = 0;
            for (const auto& [key, hist] : full_history_) {
                if (hist.is_double_freed()) {
                    double_free_count++;
                }
            }

            if (double_free_count > 0) {
                std::cout << "\n⚠️  However, found " << double_free_count
                          << " buffers that were DOUBLE-FREED during execution!" << std::endl;
                dump_double_frees();
            }

            return;
        }

        // Group by device
        std::map<int, std::vector<const BufferInfo*>> by_device;
        for (const auto& [key, info] : allocations_) {
            by_device[info.device_id].push_back(&info);
        }

        for (const auto& [device_id, buffers] : by_device) {
            std::cout << "Device " << device_id << ":" << std::endl;

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

                if (type_buffers.size() <= 10) {
                    for (const auto* buf : type_buffers) {
                        std::cout << "    - Buffer 0x" << std::hex << buf->buffer_id << std::dec << ": "
                                  << (buf->size / 1024.0) << " KB"
                                  << " (PID " << buf->owner_pid << ", ref_count=" << buf->ref_count << ")" << std::endl;

                        // Show history
                        BufferKey key{buf->device_id, buf->buffer_id};
                        auto hist_it = full_history_.find(key);
                        if (hist_it != full_history_.end()) {
                            std::cout << "      History (allocs=" << hist_it->second.total_allocs
                                      << " frees=" << hist_it->second.total_frees << "):" << std::endl;
                            int count = 0;
                            for (auto rev_it = hist_it->second.history.rbegin();
                                 rev_it != hist_it->second.history.rend() && count < 5;
                                 ++rev_it, ++count) {
                                std::cout << "        " << rev_it->to_string() << std::endl;
                            }
                        }
                    }
                }
            }
            std::cout << std::endl;
        }

        std::cout << "Total remaining buffers: " << allocations_.size() << "\n" << std::endl;
    }

    void dump_double_frees() {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║           DOUBLE-FREED BUFFERS                              ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

        for (const auto& [key, hist] : full_history_) {
            if (hist.is_double_freed()) {
                std::cout << "Buffer 0x" << std::hex << hist.buffer_id << std::dec << " on device " << hist.device_id
                          << ":" << std::endl;
                std::cout << "  Total allocs: " << hist.total_allocs << ", Total frees: " << hist.total_frees
                          << std::endl;
                std::cout << "  History:" << std::endl;
                for (const auto& event : hist.history) {
                    std::cout << "    " << event.to_string() << std::endl;
                }
                std::cout << std::endl;
            }
        }
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
                case AllocMessage::DUMP_REMAINING: handle_dump_remaining(); break;
                default: std::cerr << "Unknown message type: " << (int)msg.type << std::endl; break;
            }
        }

        close(client_socket);
    }

    void background_cleanup_loop() {
        while (running_) {
            for (int i = 0; i < 5 && running_; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            if (!running_) {
                break;
            }
            cleanup_dead_processes();
        }
    }

    void run() {
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

        if (cleanup_thread_.joinable()) {
            cleanup_thread_.join();
        }

        if (server_socket_ >= 0) {
            close(server_socket_);
        }
        unlink(TT_ALLOC_SERVER_SOCKET);
        std::cout << "\n🛑 Allocation Server stopped" << std::endl;

        // Final report
        dump_double_frees();
    }

    void print_stats() {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        std::cout << "\n📊 Current Statistics:" << std::endl;

        for (int i = 0; i < MAX_DEVICES; i++) {
            auto& stats = device_stats_[i];
            uint64_t total =
                stats.dram_allocated + stats.l1_allocated + stats.l1_small_allocated + stats.trace_allocated;

            if (total > 0 || stats.num_buffers > 0) {
                std::cout << "  Device " << i << ":" << std::endl;
                std::cout << "    Buffers: " << stats.num_buffers << std::endl;
                std::cout << "    DRAM: " << (stats.dram_allocated / 1024.0) << " KB" << std::endl;
                std::cout << "    L1: " << (stats.l1_allocated / 1024.0) << " KB" << std::endl;
                std::cout << "    Total: " << (total / 1024.0) << " KB" << std::endl;
            }
        }

        std::cout << "  Active allocations: " << allocations_.size() << std::endl;
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
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        AllocationServer server;
        g_server = &server;

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
