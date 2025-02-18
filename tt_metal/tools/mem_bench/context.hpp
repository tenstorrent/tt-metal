// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/tt_align.hpp>
#include "host_utils.hpp"

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
    uint32_t total_size;
    uint32_t page_size;
    int threads;
    int number_reader_kernels;
    int number_writer_kernels;
    bool enable_host_copy_with_kernels;
    int iterations;

    Context(
        std::map<chip_id_t, IDevice*> devices_,
        uint32_t total_size_,
        uint32_t page_size_,
        int threads_,
        int readers_,
        int writers_,
        bool enable_host_copy_with_kernels_,
        int iterations_) {
        using namespace tt::tt_metal;
        auto l1_alignment = experimental::hal::get_l1_alignment();
        auto l1_base = experimental::hal::get_tensix_l1_unreserved_base();
        device_address.cycles = l1_base;
        device_address.rd_bytes = align(device_address.cycles + sizeof(uint32_t), l1_alignment);
        device_address.wr_bytes = align(device_address.rd_bytes + sizeof(uint32_t), l1_alignment);
        device_address.unreserved = align(device_address.wr_bytes + sizeof(uint32_t), l1_alignment);
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
