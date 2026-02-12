// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <emmintrin.h>
#include <cerrno>
#include <fmt/base.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include "impl/dispatch/command_queue_common.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch/memcpy.hpp"
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

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

void* align(void* ptr, std::size_t max_alignment) {
    const std::uintptr_t uptr = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t aligned = (uptr - 1u + max_alignment) & -max_alignment;
    // Make max alignment of ptr equal to the actual specified alignment
    // ex. if the current ptr here is 16, but we specified an alignment of 8,
    // then this is both 8 and 16 byte aligned, so we offset again by our
    // specified alignment to make the max alignment what was specified
    aligned = aligned & ((max_alignment << 1) - 1) ? aligned : aligned + max_alignment;

    return reinterpret_cast<void*>(aligned);
}

#define INNER_LOOP 8

template <bool stream_load, bool aligned_load>
void nt_memcpy_128b(uint8_t* __restrict dst, const uint8_t* __restrict src, size_t n) {
    size_t num_lines = n / (INNER_LOOP * sizeof(__m128i));
    constexpr size_t inner_blk_size = INNER_LOOP * sizeof(__m128i);
    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < INNER_LOOP; j++) {
            __m128i blk;
            if constexpr (stream_load) {
                blk = _mm_stream_load_si128((__m128i*)src);
            } else {
                if constexpr (aligned_load) {
                    blk = _mm_load_si128((__m128i*)src);
                } else {
                    blk = _mm_loadu_si128((__m128i*)src);
                }
            }
            /* non-temporal store */
            _mm_stream_si128((__m128i*)dst, blk);

            src += sizeof(__m128i);
            dst += sizeof(__m128i);
        }
        n -= inner_blk_size;
    }

    if (num_lines > 0) {
        tt_driver_atomics::sfence();
    }
}

template <bool stream_load, bool aligned_load>
void nt_memcpy_256b(uint8_t* __restrict dst, const uint8_t* __restrict src, size_t n) {
    size_t num_lines = n / (INNER_LOOP * sizeof(__m256i));
    constexpr size_t inner_blk_size = INNER_LOOP * sizeof(__m256i);
    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < INNER_LOOP; j++) {
            __m256i blk;
            if constexpr (stream_load) {
                static_assert(aligned_load);
                blk = _mm256_stream_load_si256((__m256i*)src);
            } else {
                if constexpr (aligned_load) {
                    blk = _mm256_load_si256((__m256i*)src);
                } else {
                    blk = _mm256_loadu_si256((__m256i*)src);
                }
            }
            /* non-temporal store */
            _mm256_stream_si256((__m256i*)dst, blk);

            src += sizeof(__m256i);
            dst += sizeof(__m256i);
        }
        n -= inner_blk_size;
    }

    if (num_lines > 0) {
        tt_driver_atomics::sfence();
    }
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
    constexpr uint32_t memcpy_alignment = sizeof(__m256i);
    std::size_t addr_align = memcpy_alignment;

    try {
        // Input arguments parsing
        std::vector<std::string> input_args(argv, argv + argc);

        if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
            log_info(LogTest, "Usage:");
            log_info(LogTest, "  --num-tests: number of iterations");
            log_info(
                LogTest,
                "  --total-transfer-size: total size to copy to hugepage in bytes (default {} B)",
                512 * 1024 * 1024);
            log_info(LogTest, "  --transfer-size: size of one write to hugepage (default {} B)", 64 * 1024);
            log_info(LogTest, "  --enable-kernel-read: whether to run a kernel that reads from PCIe (default false)");
            log_info(
                LogTest,
                "  --simulate-wr-ptr-update: whether host writes to reg address at 32KB intervals (default false)");
            log_info(
                LogTest,
                "  --wr-ptr-rdbk-interval: after this many num writes to reg address, do readback (default 0 means no "
                "readbacks)");
            log_info(
                LogTest,
                "  --copy-mode: method used to write to pcie. 0: memcpy, 1: 4 byte writes, 2: nt_memcpy (16B streaming "
                "loads + stores), 3: nt_memcpy (16B aligned loads + streaming stores), 4: nt_memcpy (16B unaligned "
                "loads + streaming stores), 5: nt_memcpy (32B streaming loads + stores), 6: nt_memcpy (32B aligned "
                "loads + streaming stores), 7: nt_memcpy (32B unaligned loads + streaming stores) 8: memcpy_to_device");
            log_info(
                LogTest,
                "  --addr-align: Alignment of start of data. Must be a power of 2 (default {} B)",
                memcpy_alignment);
            exit(0);
        }

        try {
            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

            std::tie(total_transfer_size, input_args) = test_args::get_command_option_uint32_and_remaining_args(
                input_args, "--total-transfer-size", 512 * 1024 * 1024);

            std::tie(transfer_size, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--transfer-size", 64 * 1024);

            std::tie(enable_kernel_read, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--enable-kernel-read");

            std::tie(simulate_write_ptr_update, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--simulate-wr-ptr-update");

            std::tie(write_ptr_readback_interval, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--wr-ptr-rdbk-interval", 0);

            std::tie(copy_mode, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--copy-mode", 0);

            std::tie(addr_align, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--addr-align", memcpy_alignment);

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "Command line arguments found exception", e.what());
        }
        TT_FATAL(
            (addr_align >= 4 && (addr_align & (addr_align - 1)) == 0), "Address alignment must be a power of 2 >= 4");
        TT_FATAL(
            copy_mode <= 8,
            "Invalid --copy-mode arg! Only eight modes to copy data data from host into hugepages support!");
        if (copy_mode >= 2 && copy_mode <= 7) {
            if (copy_mode == 2 || copy_mode == 3) {
                TT_FATAL(
                    addr_align % sizeof(__m128) == 0,
                    "Address alignment must be a multiple of 16 when using nt_memcpy");
            } else if (copy_mode == 5 || copy_mode == 6) {
                TT_FATAL(
                    addr_align % sizeof(__m256) == 0,
                    "Address alignment must be a multiple of 32 when using nt_memcpy");
            }
            if (copy_mode >= 2 && copy_mode <= 4) {
                TT_FATAL(
                    transfer_size % (INNER_LOOP * sizeof(__m128)) == 0,
                    "Each copy to hugepage must be mod32==0 when using nt_memcpy");
            } else if (copy_mode >= 5 && copy_mode <= 7) {
                TT_FATAL(
                    transfer_size % (INNER_LOOP * sizeof(__m256)) == 0,
                    "Each copy to hugepage must be mod64==0 when using nt_memcpy");
            }
        }

        // Device setup
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        CoreCoord logical_core(0, 0);
        CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);

        ChipId mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
        TT_FATAL(device_id == mmio_device_id, "This test can only be run on MMIO device!");
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_id);
        void* host_hugepage_start =
            (void*)tt::tt_metal::MetalContext::instance().get_cluster().host_dma_address(0, mmio_device_id, channel);
        uint32_t hugepage_size =
            tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_size(mmio_device_id, channel);
        uint32_t host_write_ptr = 0;

        uint32_t prefetch_q_base = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
            CommandQueueDeviceAddrType::UNRESERVED);

        uint32_t reg_addr = prefetch_q_base;
        uint32_t num_reg_entries = 128;

        std::vector<uint32_t> go_signal = {0};
        std::vector<uint32_t> done_signal = {1};
        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        tt_metal::detail::WriteToDeviceL1(device->get_devices()[0], logical_core, l1_unreserved_base, go_signal);

        // Application setup
        tt_metal::Program program = tt_metal::Program();

        uint32_t kernel_read_size = 64 * 1024;

        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/3_pcie_transfer/kernels/pull_from_pcie.cpp",
            logical_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::NOC_0,
                .compile_args = {host_write_ptr, hugepage_size, kernel_read_size, l1_unreserved_base}});

        // Add 2 * alignment so that we have enough space when aligning the ptr
        // First add is for aligning to next aligned addr
        // Second add is for making sure the specified alignment is the max alignment
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            total_transfer_size + (2 * addr_align), 1000, std::chrono::system_clock::now().time_since_epoch().count());

        uint32_t* start_ptr = (uint32_t*)align(src_vec.data(), addr_align);
        std::vector<uint32_t> result_vec;

        std::string copy_mode_str;
        if (copy_mode == 0) {
            copy_mode_str = "memcpy";
        } else if (copy_mode == 1) {
            copy_mode_str = "4 byte writes";
        } else {
            copy_mode_str = "nt_memcpy";
        }

        log_info(
            LogTest,
            "Measuring host-to-device bandwidth for "
            "total_transfer_size={} B "
            "transfer_size={} B "
            "enable_kernel_read={} "
            "simulate_write_ptr_update={} "
            "write_ptr_readback_interval={} "
            "copy_mode={} ",
            total_transfer_size,
            transfer_size,
            enable_kernel_read,
            simulate_write_ptr_update,
            write_ptr_readback_interval,
            copy_mode_str);

        log_info(LogTest, "Num tests {}", num_tests);

        // Create MeshWorkload for kernel execution
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        for (uint32_t i = 0; i < num_tests; ++i) {
            // Execute application
            std::thread t1([&]() {
                if (enable_kernel_read) {
                    tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
                    tt_metal::distributed::Finish(device->mesh_command_queue());
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
                uint8_t* host_mem_ptr = (uint8_t*)host_hugepage_start + host_write_ptr;
                uint32_t src_data_offset = data_written_bytes / sizeof(uint32_t);

                if (copy_mode == 0) {
                    memcpy(host_mem_ptr, start_ptr + src_data_offset, write_size_bytes);
                } else if (copy_mode == 1) {
                    uint32_t* host_mem_ptr4B = (uint32_t*)host_mem_ptr;
                    uint32_t write_size_words = write_size_bytes / sizeof(uint32_t);

                    for (uint32_t i = 0; i < write_size_words; i++) {
                        *host_mem_ptr4B = start_ptr[src_data_offset];
                        host_mem_ptr4B++;
                        src_data_offset++;
                    }

                } else if (copy_mode == 2) {
                    TT_FATAL(
                        host_write_ptr % 16 == 0 and data_written_bytes % 16 == 0,
                        "Alignment requirement not met for copy_mode 2");
                    nt_memcpy_128b<true, true>(host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 3) {
                    TT_FATAL(
                        host_write_ptr % 16 == 0 and data_written_bytes % 16 == 0,
                        "Alignment requirement not met for copy_mode 3");
                    nt_memcpy_128b<false, true>(
                        host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 4) {
                    TT_FATAL(host_write_ptr % 16 == 0, "Alignment requirement not met for copy_mode 4");
                    nt_memcpy_128b<false, false>(
                        host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 5) {
                    TT_FATAL(
                        host_write_ptr % 32 == 0 and data_written_bytes % 32 == 0,
                        "Alignment requirement not met for copy_mode 5");
                    nt_memcpy_256b<true, true>(host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 6) {
                    TT_FATAL(
                        host_write_ptr % 32 == 0 and data_written_bytes % 32 == 0,
                        "Alignment requirement not met for copy_mode 6");
                    nt_memcpy_256b<false, true>(
                        host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 7) {
                    TT_FATAL(host_write_ptr % 32 == 0, "Alignment requirement not met for copy_mode 7");
                    nt_memcpy_256b<false, false>(
                        host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                } else if (copy_mode == 8) {
                    TT_FATAL(host_write_ptr % 16 == 0, "Alignment requirement not met for copy_mode 8");
                    memcpy_to_device<true>(host_mem_ptr, (uint8_t*)(start_ptr + src_data_offset), write_size_bytes);
                }

                uint32_t num_reg_writes = (reg_addr - prefetch_q_base) / sizeof(uint32_t);
                uint32_t val_to_write = data_written_bytes;
                if (simulate_write_ptr_update) {
                    uint32_t num_write_ptr_updates = write_size_bytes / (32 * 1024);
                    for (int i = 0; i < num_write_ptr_updates; i++) {
                        tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
                            &val_to_write, tt_cxy_pair(device->get_devices()[0]->id(), physical_core), reg_addr);
                        reg_addr += sizeof(uint32_t);
                        num_reg_writes = (reg_addr - prefetch_q_base) / sizeof(uint32_t);
                        if (num_reg_writes == num_reg_entries) {
                            reg_addr = prefetch_q_base;
                        }
                    }
                }

                if (write_ptr_readback_interval > 0 and num_reg_writes == write_ptr_readback_interval) {
                    std::vector<std::uint32_t> read_hex_vec(1, 0);
                    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        read_hex_vec.data(),
                        sizeof(uint32_t),
                        tt_cxy_pair(device->get_devices()[0]->id(), physical_core),
                        reg_addr);
                }

                host_write_ptr += write_size_bytes;
                data_written_bytes += write_size_bytes;
            }

            auto t_end = std::chrono::steady_clock::now();
            tt_metal::detail::WriteToDeviceL1(device->get_devices()[0], logical_core, l1_unreserved_base, done_signal);

            t1.join();

            auto elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            h2d_bandwidth.push_back((total_transfer_size / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0));
            log_info(LogTest, "H2D BW: {:.3f}ms, {:.3f}GB/s", elapsed_us / 1000.0, h2d_bandwidth[i]);
        }

        pass &= device->close();
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

    log_info(tt::LogTest, "test_pull_from_pcie");
    log_info(tt::LogTest, "Bandwidth(GB/s): {:.3f}", avg_h2d_bandwidth);
    log_info(tt::LogTest, "pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
