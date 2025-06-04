// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

namespace tt::tt_metal::tools::mem_bench {

struct TestResult {
    double host_bytes_processed{0};
    double host_time_elapsed{0};
    double host_wait_for_kernel_time_elapsed{0};

    double total_cores_cycles{0};
    double total_cores_time{0};
    double total_cores_bytes_rd{0};
    double total_cores_bytes_wr{0};

    double kernel_0_cycles{0};
    double kernel_0_time{0};
    double kernel_0_bytes_rd{0};
    double kernel_0_bytes_wr{0};

    // Any additional values to be included in benchmark reports
    std::map<std::string, double> arb_counters;
};

struct L1MemoryMap {
    uint32_t cycles;
    uint32_t rd_bytes;
    uint32_t wr_bytes;
    uint32_t unreserved;
};

struct Context {
    std::map<chip_id_t, IDevice*> devices;
    L1MemoryMap device_address;
    uint32_t total_size{0};
    uint32_t page_size{0};
    int threads{0};
    int number_reader_kernels{0};
    int number_writer_kernels{0};
    bool enable_host_copy_with_kernels{0};
    int iterations{0};

    Context(
        const std::map<chip_id_t, IDevice*>& devices_,
        uint32_t total_size_,
        uint32_t page_size_,
        int threads_,
        int readers_,
        int writers_,
        bool enable_host_copy_with_kernels_,
        int iterations_) {
        // Devices can be empty if it's a host only test
        if (!devices_.empty()) {
            auto l1_alignment = hal::get_l1_alignment();
            auto l1_base = devices_.begin()->second->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
            device_address.cycles = l1_base;
            device_address.rd_bytes = align(device_address.cycles + sizeof(uint32_t), l1_alignment);
            device_address.wr_bytes = align(device_address.rd_bytes + sizeof(uint32_t), l1_alignment);
            device_address.unreserved = align(device_address.wr_bytes + sizeof(uint32_t), l1_alignment);
        }
        devices = devices_;
        total_size = total_size_;
        page_size = page_size_;
        threads = threads_;
        number_reader_kernels = readers_;
        number_writer_kernels = writers_;
        enable_host_copy_with_kernels = enable_host_copy_with_kernels_;
        iterations = iterations_;
    }
};

}  // namespace tt::tt_metal::tools::mem_bench
