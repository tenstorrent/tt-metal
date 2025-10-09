// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Demo client that simulates allocations and reports to server

#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t { ALLOC = 1, FREE = 2, QUERY = 3, RESPONSE = 4 };

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
};

class AllocationClient {
private:
    int socket_fd_;
    uint64_t next_buffer_id_{1};

public:
    AllocationClient() {
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

        std::cout << "✓ Connected to allocation server" << std::endl;
    }

    ~AllocationClient() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }

    uint64_t allocate(int device_id, uint64_t size, uint8_t buffer_type) {
        AllocMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = AllocMessage::ALLOC;
        msg.device_id = device_id;
        msg.size = size;
        msg.buffer_type = buffer_type;
        msg.process_id = getpid();
        msg.buffer_id = next_buffer_id_++;
        msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        send(socket_fd_, &msg, sizeof(msg), 0);

        return msg.buffer_id;
    }

    void deallocate(uint64_t buffer_id) {
        AllocMessage msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = AllocMessage::FREE;
        msg.buffer_id = buffer_id;

        send(socket_fd_, &msg, sizeof(msg), 0);
    }
};

int main() {
    try {
        AllocationClient client;
        std::vector<uint64_t> buffers;

        std::cout << "\n🧪 Allocation Client Demo [PID: " << getpid() << "]" << std::endl;
        std::cout << "   This simulates memory allocations reported to the server" << std::endl;
        std::cout << "   Watch the server and monitor output!" << std::endl;

        // Step 1: Allocate some L1
        std::cout << "\n[Step 1] Allocating 4MB of L1..." << std::endl;
        for (int i = 0; i < 4; ++i) {
            auto id = client.allocate(0, 1024 * 1024, 1);  // device 0, 1MB, L1
            buffers.push_back(id);
            std::cout << "  Allocated buffer " << id << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Step 2: Allocate some DRAM
        std::cout << "\n[Step 2] Allocating 100MB of DRAM..." << std::endl;
        for (int i = 0; i < 4; ++i) {
            auto id = client.allocate(0, 25 * 1024 * 1024, 0);  // device 0, 25MB, DRAM
            buffers.push_back(id);
            std::cout << "  Allocated buffer " << id << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Step 3: Allocate more L1
        std::cout << "\n[Step 3] Allocating 8MB more L1..." << std::endl;
        for (int i = 0; i < 8; ++i) {
            auto id = client.allocate(0, 1024 * 1024, 1);  // device 0, 1MB, L1
            buffers.push_back(id);
            std::cout << "  Allocated buffer " << id << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Step 4: Free half the buffers
        std::cout << "\n[Step 4] Freeing half the buffers..." << std::endl;
        size_t half = buffers.size() / 2;
        for (size_t i = 0; i < half; ++i) {
            client.deallocate(buffers[i]);
            std::cout << "  Freed buffer " << buffers[i] << std::endl;
        }
        buffers.erase(buffers.begin(), buffers.begin() + half);
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Step 5: Free all remaining
        std::cout << "\n[Step 5] Freeing all remaining buffers..." << std::endl;
        for (auto id : buffers) {
            client.deallocate(id);
            std::cout << "  Freed buffer " << id << std::endl;
        }
        buffers.clear();
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "\n✅ Demo complete!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Make sure allocation_server is running!" << std::endl;
        return 1;
    }

    return 0;
}
