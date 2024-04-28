// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This test measures the bandwidth of host-to-device data transfer through hugepage.
// Modes available:
//  1. Just write to hugepage
//  2. Write to hugepage with kernel reading from hugepage in a loop
//  3. Write to hugepage and write to L1 via write_reg API with and without kernel reading
//  4. 3 plus read from L1 every <n> writes to write_reg
//
// Run ./test_pull_from_pcie --help to see usage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CACHE_LINE_SIZE 64
void nt_memcpy(uint8_t *__restrict dst, const uint8_t * __restrict src, size_t n)
{
    size_t num_lines = n / CACHE_LINE_SIZE;

    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < CACHE_LINE_SIZE / sizeof(__m128i); j++) {
            __m128i blk = _mm_stream_load_si128((__m128i *)src);
            /* non-temporal store */
            _mm_stream_si128((__m128i *)dst, blk);
            src += sizeof(__m128i);
            dst += sizeof(__m128i);
        }
        n -= CACHE_LINE_SIZE;
    }

    if (num_lines > 0)
        _mm_sfence();
}


int main(int argc, char** argv) {
    bool pass = true;
    std::vector<double> h2d_bandwidth;
    uint32_t num_tests = 10;
    uint32_t total_transfer_size = 512 * 1024 * 1024;
    uint32_t transfer_size = 512 * 1024 * 1024;
    bool enable_kernel_read = false;
    bool simulate_write_ptr_update = false;
    uint32_t write_ptr_readback_interval = 0;
    uint32_t copy_mode = 0;

    try {
        // Input arguments parsing
        std::vector<std::string> input_args(argv, argv + argc);

        if (test_args::has_command_option(input_args, "-h") ||
            test_args::has_command_option(input_args, "--help")) {
            log_info(LogTest, "Usage:");
            log_info(LogTest, "  --num-tests: number of iterations");
            log_info(LogTest, "  --total-transfer-size: total size to copy to hugepage in bytes (default {} B)", 512 * 1024 * 1024);
            log_info(LogTest, "  --transfer-size: size of one write to hugepage (default {} B)", 64 * 1024);
            log_info(LogTest, "  --enable-kernel-read: whether to run a kernel that reads from PCIe (default false)");
            log_info(LogTest, "  --simulate-wr-ptr-update: whether host writes to reg address at 32KB intervals (default false)");
            log_info(LogTest, "  --wr-ptr-rdbk-interval: after this many num writes to reg address, do readback (default 0 means no readbacks)");
            log_info(LogTest, "  --copy-mode: method used to write to pcie. 0: memcpy, 1: 4 byte writes, 2: nt_memcpy (uncached writes + 16B stores)");
            exit(0);
        }

        try {
            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

            std::tie(total_transfer_size, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--total-transfer-size", 512 * 1024 * 1024);

            std::tie(transfer_size, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--transfer-size", 64 * 1024);

            std::tie(enable_kernel_read, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--enable-kernel-read");

            std::tie(simulate_write_ptr_update, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--simulate-wr-ptr-update");

            std::tie(write_ptr_readback_interval, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--wr-ptr-rdbk-interval", 0);

            std::tie(copy_mode, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--copy-mode", 0);

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
        }

        TT_ASSERT(copy_mode == 0 or copy_mode == 1 or copy_mode == 2, "Invalid --copy-mode arg! Only three modes to copy data data from host into hugepages support! memcpy, 4 byte writes, and nt_copy");
        if (copy_mode == 2) {
            TT_ASSERT(transfer_size % 64 == 0, "Each copy to hugepage must be mod64==0 when using nt_memcpy");
        }

        // Device setup
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);
        CoreCoord logical_core(0, 0);
        CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        TT_ASSERT(device_id == mmio_device_id, "This test can only be run on MMIO device!");
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        void* host_hugepage_start = (void*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        uint32_t hugepage_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel);
        uint32_t host_write_ptr = 0;

        uint32_t reg_addr = dispatch_constants::PREFETCH_Q_BASE;
        uint32_t num_reg_entries = 128;

        std::vector<uint32_t> go_signal = {0};
        std::vector<uint32_t> done_signal = {1};
        tt_metal::detail::WriteToDeviceL1(device, logical_core, L1_UNRESERVED_BASE, go_signal);

        // Application setup
        tt_metal::Program program = tt_metal::Program();

        uint32_t kernel_read_size = 64 * 1024;

        auto pcie_reader = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/3_pcie_transfer/kernels/pull_from_pcie.cpp",
            logical_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {host_write_ptr, hugepage_size, kernel_read_size}
            });

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            total_transfer_size, 1000, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> result_vec;

        const std::string copy_mode_str = copy_mode == 0 ? "memcpy" : copy_mode == 1 ? "4 byte writes" : "nt_memcpy";

        log_info(
            LogTest,
            "Measuring host-to-device bandwidth for "
            "total_transfer_size={} B "
            "transfer_size={} B "
            "enable_kernel_read={} "
            "simulate_write_ptr_update={} "
            "write_ptr_readback_interval={} "
            "copy_mode={} ",
            total_transfer_size, transfer_size, enable_kernel_read, simulate_write_ptr_update, write_ptr_readback_interval, copy_mode_str);

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            // Execute application
            std::thread t1 ([&]() {
                if (enable_kernel_read) {
                    tt::tt_metal::detail::LaunchProgram(device, program);
                }
            });

            auto t_begin = std::chrono::steady_clock::now();

            uint32_t data_written_bytes = 0;
            while (data_written_bytes < total_transfer_size) {
                int32_t space_available = hugepage_size - host_write_ptr;
                if (space_available <= 0) {
                    host_write_ptr = 0;
                    space_available = hugepage_size;
                }
                uint32_t write_size_bytes = std::min((uint32_t)space_available, transfer_size);
                write_size_bytes = std::min(write_size_bytes, (total_transfer_size - data_written_bytes));
                uint8_t* host_mem_ptr = (uint8_t *)host_hugepage_start + host_write_ptr;

                if (copy_mode == 0) {
                    memcpy(host_mem_ptr, src_vec.data() + (data_written_bytes / sizeof(uint32_t)), write_size_bytes);
                } else if (copy_mode == 1) {

                    uint32_t *host_mem_ptr4B = (uint32_t *)host_mem_ptr;
                    uint32_t write_size_words = write_size_bytes / sizeof(uint32_t);
                    uint32_t src_data_offset = data_written_bytes / sizeof(uint32_t);

                    for (uint32_t i = 0; i < write_size_words; i++) {
                        *host_mem_ptr4B = src_vec[src_data_offset];
                        host_mem_ptr4B++;
                        src_data_offset++;
                    }

                } else if (copy_mode == 2) {
                    TT_ASSERT(host_write_ptr % 16 == 0 and data_written_bytes % 16 == 0);
                    nt_memcpy(host_mem_ptr, (uint8_t *)(src_vec.data() + (data_written_bytes / sizeof(uint32_t))), write_size_bytes);
                }

                uint32_t num_reg_writes = (reg_addr - dispatch_constants::PREFETCH_Q_BASE) / sizeof(uint32_t);
                uint32_t val_to_write = data_written_bytes;
                if (simulate_write_ptr_update) {
                    uint32_t num_write_ptr_updates = write_size_bytes / (32 * 1024);
                    for (int i = 0; i < num_write_ptr_updates; i++) {
                        tt::Cluster::instance().write_reg(&val_to_write, tt_cxy_pair(device->id(), physical_core), reg_addr);
                        reg_addr += sizeof(uint32_t);
                        num_reg_writes = (reg_addr - dispatch_constants::PREFETCH_Q_BASE) / sizeof(uint32_t);
                        if (num_reg_writes == num_reg_entries) {
                            reg_addr = dispatch_constants::PREFETCH_Q_BASE;
                        }
                    }
                }

                if (write_ptr_readback_interval > 0 and num_reg_writes == write_ptr_readback_interval) {
                    std::vector<std::uint32_t> read_hex_vec(1, 0);
                    tt::Cluster::instance().read_core(read_hex_vec.data(), sizeof(uint32_t), tt_cxy_pair(device->id(), physical_core), reg_addr);
                }

                host_write_ptr += write_size_bytes;
                data_written_bytes += write_size_bytes;
            }

            auto t_end = std::chrono::steady_clock::now();
            tt_metal::detail::WriteToDeviceL1(device, logical_core, L1_UNRESERVED_BASE, done_signal);

            t1.join();

            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            h2d_bandwidth.push_back((total_transfer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0));
            log_info(
                LogTest,
                "H2D BW: {:.3f}ms, {:.3f}GB/s",
                elapsed_us / 1000.0,
                h2d_bandwidth[i]);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_h2d_bandwidth = calculate_average(h2d_bandwidth);

    if (pass) {
        // goal is 70% of PCI-e Gen3 x16 for grayskull
        // TODO: check the theoritical peak of wormhole
        double target_bandwidth = 16.0 * 0.7;

        if (avg_h2d_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The host-to-device bandwidth does not meet the criteria. "
                "Current: {:.3f}GB/s, goal: {:.3f}GB/s",
                avg_h2d_bandwidth,
                target_bandwidth);
        }
    }

    log_info("test_pull_from_pcie");
    log_info("Bandwidth(GB/s): {:.3f}", avg_h2d_bandwidth);
    log_info("pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
