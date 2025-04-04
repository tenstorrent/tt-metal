// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_align.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "dispatch/memcpy.hpp"
#include "vector_aligned.hpp"

namespace tt::tt_metal::tools::mem_bench {

// Generate random data aligned for memcpy_to_device.
tt::tt_metal::vector_aligned<uint32_t> generate_random_src_data(uint32_t num_bytes);

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
template <bool fence = false>
double copy_to_hugepage(
    void* hugepage_base,
    uint32_t hugepage_size,
    std::span<uint32_t> src_data,
    size_t total_size,
    size_t page_size,
    bool repeating_src_vector) {
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

        if (!repeating_src_vector) {
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

};  // namespace tt::tt_metal::tools::mem_bench
