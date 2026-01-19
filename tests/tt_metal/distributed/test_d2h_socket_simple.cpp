// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple D2H Socket Example
// This test demonstrates sending bytes from device to host using a D2H socket
// with a background receiver thread that continuously services the pinned memory.

#include <tt-metalium/distributed.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "gmock/gmock.h"
#include "tracy/Tracy.hpp"
#include <atomic>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace tt::tt_metal::distributed {

// Use GenericMeshDeviceFixture so this works on any system (including single device)
class SimpleD2HSocketTest : public GenericMeshDeviceFixture {};

// Thread-safe queue to pass received pages from receiver thread to main thread
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cv_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool try_pop(T& value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Background receiver thread function
void receiver_thread_func(
    D2HSocket* socket,
    uint32_t page_size,
    uint32_t total_pages,
    ThreadSafeQueue<std::vector<uint32_t>>* output_queue,
    std::atomic<bool>* done) {
    ZoneScoped;
    uint32_t pages_received = 0;
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    std::cout << "[Receiver Thread] Started, expecting " << total_pages << " pages\n";

    while (pages_received < total_pages) {
        // Wait for a page to be available
        socket->wait_for_pages(1);

        // Copy the page data
        std::vector<uint32_t> page_data(page_size_words);
        std::memcpy(page_data.data(), socket->get_read_ptr(), page_size);

        // Mark the page as consumed and notify device
        socket->pop_pages(1);
        socket->notify_sender();

        // Push to output queue for main thread
        output_queue->push(std::move(page_data));

        pages_received++;
        std::cout << "[Receiver Thread] Received page " << pages_received << "/" << total_pages << "\n";
    }

    // Wait for all data to be acknowledged
    socket->barrier();

    done->store(true);
    std::cout << "[Receiver Thread] Done, received all " << pages_received << " pages\n";
}

TEST_F(SimpleD2HSocketTest, SendBytesWithReceiverThread) {
    // Configuration: 64 bytes is the minimum PCIe-aligned page size on Blackhole
    constexpr std::size_t page_size = 64;
    constexpr std::size_t num_pages = 4;                      // Send multiple pages to demonstrate streaming
    constexpr std::size_t data_size = page_size * num_pages;  // 256 bytes total
    constexpr std::size_t fifo_size = 1024;

    std::cout << "\n=== D2H Socket Test with Receiver Thread ===\n";
    std::cout << "Sending " << data_size << " bytes (" << num_pages << " pages of " << page_size << " bytes)\n\n";

    // The sender core on device (0,0) at logical core (0,0)
    auto sender_core = MeshCoreCoord{MeshCoordinate(0, 0), CoreCoord(0, 0)};

    // Create D2H socket - this sets up pinned memory for device-to-host transfers
    auto output_socket = D2HSocket(mesh_device_, sender_core, BufferType::L1, fifo_size);
    output_socket.set_page_size(page_size);

    // Create a buffer on device to hold the source data
    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(CoreCoord(0, 0)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device_.get());

    // Create a simple kernel that sends data through the socket
    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/d2h_sender.cpp",
        CoreCoord(0, 0),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
            }});

    // Prepare source data: fill with recognizable pattern (0, 1, 2, 3, ...)
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    for (size_t i = 0; i < src_vec.size(); i++) {
        src_vec[i] = static_cast<uint32_t>(i);
    }

    // Write source data to device
    WriteShard(mesh_device_->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    // Set up the receiver thread
    ThreadSafeQueue<std::vector<uint32_t>> received_pages;
    std::atomic<bool> receiver_done{false};

    // Start the receiver thread BEFORE launching the kernel
    std::thread receiver_thread(
        receiver_thread_func, &output_socket, page_size, num_pages, &received_pages, &receiver_done);

    std::cout << "[Main Thread] Launching kernel...\n";

    // Launch the kernel (non-blocking)
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0));
    mesh_workload.add_program(devices, std::move(send_program));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);

    std::cout << "[Main Thread] Kernel launched, waiting for data...\n";

    // Main thread collects received pages from the queue
    std::vector<uint32_t> dst_vec;
    dst_vec.reserve(data_size / sizeof(uint32_t));

    for (uint32_t i = 0; i < num_pages; i++) {
        // Get page from receiver thread (blocking)
        auto page_data = received_pages.pop();

        std::cout << "[Main Thread] Got page " << (i + 1) << " from receiver thread: ";
        for (size_t j = 0; j < std::min(page_data.size(), size_t(4)); j++) {
            std::cout << page_data[j] << " ";
        }
        std::cout << "...\n";

        // Append to destination buffer
        dst_vec.insert(dst_vec.end(), page_data.begin(), page_data.end());
    }

    // Wait for receiver thread to finish
    receiver_thread.join();

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << "Received data (as uint32_t values):\n  ";
    for (size_t i = 0; i < dst_vec.size(); i++) {
        std::cout << dst_vec[i] << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << "\n  ";
        }
    }
    std::cout << "\n";

    // Verify the data matches
    EXPECT_EQ(src_vec, dst_vec);
    std::cout << "Data verification: " << (src_vec == dst_vec ? "PASSED" : "FAILED") << "\n";
    std::cout << "================================\n\n";
}

}  // namespace tt::tt_metal::distributed
