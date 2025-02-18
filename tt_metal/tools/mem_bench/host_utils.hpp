// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <chrono>
#include <vector>
#include <atomic>
#include <thread>
#include <iostream>
#include <tt-metalium/memcpy.hpp>
#include <tt-metalium/tt_align.hpp>

namespace tt::tt_metal::tools::mem_bench {

// Generate aligned source data.
tt::tt_metal::vector_memcpy_aligned<uint32_t> gen_src_data(uint32_t num_bytes);

// Get current host time, in seconds.
double get_current_time_seconds();

// Return device ids. If numa_node is specified then only device ids on that
// node will be returned. If numa_node == -1, then the node is not taken into
// consideration. Note: Less than number_of_devices may be returned.
std::vector<int> get_mmio_device_ids(int number_of_devices, int numa_node);

// Returns device ids. All devices are on different nodes. Note: Less than
// number_of_devices may be returned.
std::vector<int> get_mmio_device_ids_unique_nodes(int number_of_devices);

// Returns the number of MMIO connected chips.
int get_number_of_mmio_devices();

// Returns the hugepage pointer assigned to a device.
void* get_hugepage(int device_id, uint32_t base_offset);

// Returns the size of the hugepage assigned to a device.
uint32_t get_hugepage_size(int device_id);

// Copy data to hugepage. Returns the duration.
// repeating_src_vector: Keep copying the same elements to hugepage. This should force the source data in stay in the
// caches. fence: Memory barrier at the end of each copy. Returns the time in seconds
template <bool repeating_src_vector, bool fence = false>
double copy_to_hugepage(
    void* hugepage_base, uint32_t hugepage_size, std::span<uint32_t> src_data, size_t total_size, size_t page_size) {
    uint64_t hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
    uint64_t hugepage_end = hugepage_addr + hugepage_size;
    uint64_t src_addr = reinterpret_cast<uint64_t>(src_data.data());
    size_t num_pages;
    if (!page_size) {
        num_pages = 1;
        page_size = total_size;
    } else {
        num_pages = total_size / page_size;
    }

    auto start = get_current_time_seconds();
    for (int i = 0; i < num_pages; ++i) {
        tt::tt_metal::memcpy_to_device<fence>((void*)(hugepage_addr), (void*)(src_addr), page_size);

        // 64 bit host address alignment
        hugepage_addr = ((hugepage_addr + page_size - 1) | (tt::tt_metal::MEMCPY_ALIGNMENT - 1)) + 1;

        if constexpr (!repeating_src_vector) {
            src_addr += page_size;
        }

        // Wrap back to the beginning of hugepage
        if (hugepage_addr + page_size >= hugepage_end) {
            hugepage_addr = reinterpret_cast<uint64_t>(hugepage_base);
        }
    }
    auto end = get_current_time_seconds();

    return end - start;
}

// Copy data to hugepage with N threads to try saturating bandwidth.
// The total size of src_data to be copied to hugepage is split equally to the threads.
// Returns time taken in seconds for all copies to complete. Time is calculated by latest thread end - earliest thread
// start.
template <bool repeating_src_vector, bool fence = false>
double copy_to_hugepage_threaded(
    void* hugepage_base,
    uint32_t hugepage_size,
    std::span<uint32_t> src_data,
    size_t total_size,
    size_t page_size,
    int num_threads) {
    using namespace tt::tt_metal;
    static_assert((MEMCPY_ALIGNMENT & ((MEMCPY_ALIGNMENT)-1)) == 0);
    const auto bytes_per_thread = ((total_size / num_threads) + (MEMCPY_ALIGNMENT)-1) & -(MEMCPY_ALIGNMENT);
    const auto last_thread_bytes = total_size - (bytes_per_thread * (num_threads - 1));

    std::vector<double> thread_durations(num_threads);
    std::vector<double> thread_start_times(num_threads);
    std::vector<double> thread_end_times(num_threads);
    std::vector<std::thread> threads(num_threads);
    std::atomic<bool> start_flag{false};
    std::atomic<int> threads_ready{0};

    // Push back threads
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread([i,
                                  &threads_ready,
                                  &start_flag,
                                  bytes_per_thread,
                                  hugepage_base,
                                  hugepage_size,
                                  page_size,
                                  num_threads,
                                  last_thread_bytes,
                                  &src_data,
                                  &thread_durations,
                                  &thread_start_times,
                                  &thread_end_times]() {
            // Slice of the source data for this thread
            uint64_t thread_dst = (uint64_t)hugepage_base + (i * bytes_per_thread);
            uint64_t thread_bytes = (i == num_threads - 1) ? last_thread_bytes : bytes_per_thread;
            std::span<uint32_t> thread_src{src_data};
            thread_src = thread_src.subspan((i * bytes_per_thread) / sizeof(uint32_t), thread_bytes / sizeof(uint32_t));

            threads_ready++;

            // Wait to start
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            thread_start_times[i] = get_current_time_seconds();
            thread_durations[i] = copy_to_hugepage<repeating_src_vector, fence>(
                (void*)thread_dst, hugepage_size, thread_src, thread_bytes, page_size);
            thread_end_times[i] = get_current_time_seconds();
        });
    }

    while (threads_ready.load() < num_threads) {
        std::this_thread::yield();
    }

    start_flag.store(true);
    for (auto& thread : threads) {
        thread.join();
    }

    double earliest_start = *std::min_element(thread_start_times.begin(), thread_start_times.end());
    double latest_end = *std::max_element(thread_end_times.begin(), thread_end_times.end());

    return latest_end - earliest_start;
}

};  // namespace tt::tt_metal::tools::mem_bench
