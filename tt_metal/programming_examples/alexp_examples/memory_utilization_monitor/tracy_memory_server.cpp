// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tracy Memory Monitor Server
 *
 * This server exposes TracyMemoryMonitor data via Unix socket for cross-process access.
 * It's a lightweight wrapper that allows the tracy_memory_monitor_client to see
 * allocations from other processes (like Python tests).
 *
 * Usage:
 *   Terminal 1: ./tracy_memory_server
 *   Terminal 2: ./tracy_memory_monitor_client -a
 *   Terminal 3: python test_mesh_allocation.py
 *
 * The server stays running and automatically exports stats as allocations happen.
 */

#include "tt_metal/impl/profiler/tracy_memory_monitor.hpp"
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>
#include <thread>
#include <atomic>

#define TRACY_MEMORY_SOCKET "/tmp/tracy_memory_monitor.sock"

using namespace tt::tt_metal;

// Message protocol for client/server communication
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
    // Total: 72 bytes
};

class TracyMemoryServer {
private:
    int server_socket_;
    std::atomic<bool> running_{true};

public:
    TracyMemoryServer() {
        server_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_socket_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TRACY_MEMORY_SOCKET, sizeof(addr.sun_path) - 1);

        // Remove existing socket file
        unlink(TRACY_MEMORY_SOCKET);

        if (bind(server_socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }

        if (listen(server_socket_, 128) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }

        std::cout << "🚀 Tracy Memory Monitor Server started" << std::endl;
        std::cout << "   Socket: " << TRACY_MEMORY_SOCKET << std::endl;
        std::cout << "   Exposing TracyMemoryMonitor for cross-process access" << std::endl;
        std::cout << "   Press Ctrl+C to stop" << std::endl;
    }

    ~TracyMemoryServer() { stop(); }

    void handle_query(int client_socket, const MemoryQueryMessage& query) {
        MemoryQueryMessage response;
        memset(&response, 0, sizeof(response));
        response.type = MemoryQueryMessage::RESPONSE;
        response.device_id = query.device_id;

        if (query.device_id >= 0 && query.device_id < TracyMemoryMonitor::MAX_DEVICES) {
            // Query the singleton in THIS process
            auto& monitor = TracyMemoryMonitor::instance();
            auto stats = monitor.query_device(query.device_id);

            response.dram_allocated = stats.dram_allocated;
            response.l1_allocated = stats.l1_allocated;
            response.system_memory_allocated = stats.system_memory_allocated;
            response.l1_small_allocated = stats.l1_small_allocated;
            response.trace_allocated = stats.trace_allocated;
            response.num_buffers = stats.num_buffers;
            response.total_allocs = stats.total_allocs;
            response.total_frees = stats.total_frees;
        }

        send(client_socket, &response, sizeof(response), 0);
    }

    void handle_client(int client_socket) {
        MemoryQueryMessage msg;

        while (running_) {
            ssize_t n = recv(client_socket, &msg, sizeof(msg), 0);
            if (n <= 0) {
                break;
            }

            if (msg.type == MemoryQueryMessage::QUERY) {
                handle_query(client_socket, msg);
            }
        }

        close(client_socket);
    }

    void run() {
        std::cout << "\n✅ Server ready - waiting for connections..." << std::endl;
        std::cout << "   Run: ./tracy_memory_monitor_client -a" << std::endl;
        std::cout << "   Then: python test_mesh_allocation.py\n" << std::endl;

        while (running_) {
            int client_socket = accept(server_socket_, nullptr, nullptr);
            if (client_socket < 0) {
                if (running_) {
                    std::cerr << "Accept failed" << std::endl;
                }
                continue;
            }

            std::cout << "📱 Client connected" << std::endl;

            // Handle client in a detached thread
            std::thread(&TracyMemoryServer::handle_client, this, client_socket).detach();
        }
    }

    void stop() {
        running_ = false;
        if (server_socket_ >= 0) {
            close(server_socket_);
        }
        unlink(TRACY_MEMORY_SOCKET);
        std::cout << "\n🛑 Tracy Memory Monitor Server stopped" << std::endl;
    }

    void print_stats() {
        auto& monitor = TracyMemoryMonitor::instance();

        std::cout << "\n📊 Current Statistics:" << std::endl;
        for (int i = 0; i < TracyMemoryMonitor::MAX_DEVICES; i++) {
            auto stats = monitor.query_device(i);

            if (stats.num_buffers > 0 || stats.get_total_allocated() > 0) {
                std::cout << "  Device " << i << ":" << std::endl;
                std::cout << "    Buffers: " << stats.num_buffers << std::endl;
                std::cout << "    DRAM: " << stats.dram_allocated << " bytes" << std::endl;
                std::cout << "    L1: " << stats.l1_allocated << " bytes" << std::endl;
                std::cout << "    Total: " << stats.get_total_allocated() << " bytes" << std::endl;
            }
        }
    }
};

TracyMemoryServer* g_server = nullptr;

void signal_handler(int sig) {
    std::cout << "\n⚠  Interrupt received..." << std::endl;
    if (g_server) {
        g_server->print_stats();
        g_server->stop();
    }
    exit(0);
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        TracyMemoryServer server;
        g_server = &server;

        // Print stats every 30 seconds in background
        std::thread stats_thread([&]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(30));
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
