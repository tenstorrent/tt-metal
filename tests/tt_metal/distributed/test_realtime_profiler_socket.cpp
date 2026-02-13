// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler socket test
// This test demonstrates using the real-time profiler D2H socket that is initialized
// during device init to stream data from device to host.

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
class RealtimeProfilerSocketTest : public GenericMeshDeviceFixture {};

// Thread-safe queue to pass received pages from receiver thread to main thread
template <typename T>
class RealtimeProfilerQueue {
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

// Background receiver thread function for real-time profiler data
void realtime_profiler_receiver_thread_func(
    D2HSocket* socket,
    uint32_t page_size,
    uint32_t total_pages,
    RealtimeProfilerQueue<std::vector<uint32_t>>* output_queue,
    std::atomic<bool>* done) {
    ZoneScoped;
    uint32_t pages_received = 0;
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    std::cout << "[Realtime Profiler Receiver] Started, expecting " << total_pages << " pages of " << page_size
              << " bytes\n";

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

        // Print progress every 10 pages
        if (pages_received % 10 == 0 || pages_received == total_pages) {
            std::cout << "[Realtime Profiler Receiver] Received page " << pages_received << "/" << total_pages << "\n";
        }
    }

    // Wait for all data to be acknowledged
    socket->barrier();

    done->store(true);
    std::cout << "[Realtime Profiler Receiver] Done, received all " << pages_received << " pages\n";
}

TEST_F(RealtimeProfilerSocketTest, StreamRealtimeProfilerData) {
    // Configuration: 64 bytes is the minimum PCIe-aligned page size on Blackhole
    constexpr std::size_t page_size = 64;
    constexpr std::size_t num_pages = 100;                    // Send 100 pages
    constexpr std::size_t data_size = page_size * num_pages;  // 6400 bytes total

    std::cout << "\n=== Real-time profiler socket test ===\n";
    std::cout << "Streaming " << num_pages << " pages of " << page_size << " bytes (" << data_size
              << " bytes total)\n\n";

    // Get the real-time profiler socket that was initialized during device init
    D2HSocket* realtime_profiler_socket = mesh_device_->get_realtime_profiler_socket();
    ASSERT_NE(realtime_profiler_socket, nullptr) << "Real-time profiler socket was not initialized";

    // Set the page size for this transfer
    realtime_profiler_socket->set_page_size(page_size);

    std::cout << "[Main Thread] Using real-time profiler socket with config buffer at 0x" << std::hex
              << realtime_profiler_socket->get_config_buffer_address() << std::dec << "\n";

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

    // Create a kernel that sends real-time profiler data through the socket
    // Running on core (0,0) BRISC (RISCV_0)
    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/d2h_sender.cpp",
        CoreCoord(0, 0),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,  // BRISC
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(realtime_profiler_socket->get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
            }});

    // Prepare source data: fill with recognizable pattern
    // Each page contains a unique identifier and sequential data
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    for (size_t i = 0; i < src_vec.size(); i++) {
        // Pattern: page_number in upper 16 bits, word_index in lower 16 bits
        uint32_t page_num = static_cast<uint32_t>(i / (page_size / sizeof(uint32_t)));
        uint32_t word_in_page = static_cast<uint32_t>(i % (page_size / sizeof(uint32_t)));
        src_vec[i] = (page_num << 16) | word_in_page;
    }

    // Write source data to device
    WriteShard(mesh_device_->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

    // Set up the receiver thread
    RealtimeProfilerQueue<std::vector<uint32_t>> received_pages;
    std::atomic<bool> receiver_done{false};

    // Start the receiver thread BEFORE launching the kernel
    std::thread receiver_thread(
        realtime_profiler_receiver_thread_func,
        realtime_profiler_socket,
        page_size,
        num_pages,
        &received_pages,
        &receiver_done);

    std::cout << "[Main Thread] Launching kernel on core (0,0) BRISC...\n";

    // Launch the kernel (non-blocking)
    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0));
    mesh_workload.add_program(devices, std::move(send_program));
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);

    std::cout << "[Main Thread] Kernel launched, receiving real-time profiler data...\n";

    // Main thread collects received pages from the queue and prints them
    std::vector<uint32_t> dst_vec;
    dst_vec.reserve(data_size / sizeof(uint32_t));

    for (uint32_t i = 0; i < num_pages; i++) {
        // Get page from receiver thread (blocking)
        auto page_data = received_pages.pop();

        // Print first and last few pages, and every 20th page
        if (i < 3 || i >= num_pages - 3 || i % 20 == 0) {
            std::cout << "[Main Thread] Page " << std::setw(3) << i << ": ";
            // Print first 4 words of the page
            for (size_t j = 0; j < std::min(page_data.size(), size_t(4)); j++) {
                std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') << page_data[j] << " ";
            }
            std::cout << std::dec << std::setfill(' ') << "...\n";
        } else if (i == 3) {
            std::cout << "[Main Thread] ... (skipping intermediate pages) ...\n";
        }

        // Append to destination buffer
        dst_vec.insert(dst_vec.end(), page_data.begin(), page_data.end());
    }

    // Wait for receiver thread to finish
    receiver_thread.join();

    // Print summary
    std::cout << "\n=== Results ===\n";
    std::cout << "Total pages received: " << num_pages << "\n";
    std::cout << "Total bytes received: " << dst_vec.size() * sizeof(uint32_t) << "\n";

    // Verify the data matches
    bool data_matches = (src_vec == dst_vec);
    EXPECT_EQ(src_vec, dst_vec);
    std::cout << "Data verification: " << (data_matches ? "PASSED" : "FAILED") << "\n";

    if (!data_matches) {
        // Print first mismatch for debugging
        for (size_t i = 0; i < std::min(src_vec.size(), dst_vec.size()); i++) {
            if (src_vec[i] != dst_vec[i]) {
                std::cout << "First mismatch at index " << i << ": expected 0x" << std::hex << src_vec[i] << ", got 0x"
                          << dst_vec[i] << std::dec << "\n";
                break;
            }
        }
    }

    std::cout << "================================\n\n";
}

}  // namespace tt::tt_metal::distributed
