// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <thread>

#include "common/bfloat16.hpp"
#include "test_tiles.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

template <typename T>
std::vector<T> slice_vec(std::vector<T> const &v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

void print_vec(std::vector<bfloat16> data, int rows, int cols, string name) {
    std::cout << name << ": " << std::endl;
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data.at(index).to_float() << " ";
            index++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t dprint;
        uint32_t print_tensor;
        uint32_t debug;
        uint32_t cb_n;
        uint32_t Nt;
        uint32_t num_cores_r;
        uint32_t num_cores_c;
        uint32_t single_read;
        uint32_t one_buffer_share;
        uint32_t validation;
        try {
            std::tie(debug, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--debug", 0);
            std::tie(dprint, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--dprint", 0);
            std::tie(print_tensor, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--print_tensor", 0);
            std::tie(Nt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--nt", 256);
            std::tie(cb_n, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--cb", 1);
            std::tie(num_cores_r, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--r", 9);
            std::tie(num_cores_c, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--c", 12);
            std::tie(single_read, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--same_buffer_read", 0);
            std::tie(one_buffer_share, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--one_buffer_share", 0);
            std::tie(validation, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--validation", 1);
        } catch (const std::exception &e) {
            TT_THROW("Command line arguments found exception", e.what());
        }

        log_info(LogTest, "one_buffer_share {}, same_buffer_read {}", one_buffer_share, single_read);

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // for convenience
        if (Nt % cb_n != 0) {
            log_error(LogTest, "activations({} tiles) should be divided cb buffer ({} tiles)", Nt, cb_n);
            TT_FATAL(false);
        }

        tt::DataFormat data_format = tt::DataFormat::Float16_b;
        uint32_t single_tile_size = 2 * 1024;

        uint32_t shape_Nt = Nt;
        if (one_buffer_share)
            shape_Nt = (num_cores_r * num_cores_c) * Nt;
        log_info(LogTest, "Activations = {}x{}", shape_Nt * 32, 32);

        SHAPE shape = {1, 1, shape_Nt * 32, 32};
        std::vector<tt::deprecated::Tensor<bfloat16>> tensors;
        for (int r = 0; r < num_cores_r; ++r) {
            for (int c = 0; c < num_cores_c; ++c) {
                auto tensor = tt::deprecated::initialize_tensor<bfloat16>(
                    shape,
                    tt::deprecated::Initialize::RANDOM,
                    100,
                    std::chrono::system_clock::now().time_since_epoch().count());

                tensors.push_back(tensor);
                if (single_read || one_buffer_share)
                    break;
            }
            if (single_read || one_buffer_share)
                break;
        }

        if (print_tensor) {
            for (int r = 0; r < num_cores_r; ++r) {
                for (int c = 0; c < num_cores_c; ++c) {
                    print_vec(
                        tensors[r * num_cores_c + c].get_values(),
                        1,
                        32,
                        std::string("input tensor " + std::to_string(r) + " " + std::to_string(c)));
                    if (single_read || one_buffer_share)
                        break;
                }
                if (single_read || one_buffer_share)
                    break;
            }
        }

        std::vector<std::vector<uint32_t>> packed_tensors;
        for (int r = 0; r < num_cores_r; ++r) {
            for (int c = 0; c < num_cores_c; ++c) {
                auto activations = pack_bfloat16_vec_into_uint32_vec(tensors[r * num_cores_c + c].get_values());
                packed_tensors.push_back(activations);
                if (single_read || one_buffer_share)
                    break;
            }
            if (single_read || one_buffer_share)
                break;
        }

        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
        const CoreRange all_cores(start_core, end_core);

        log_info(LogTest, "core range {},{} - {},{}", start_core.x, start_core.y, end_core.x, end_core.y);

        uint32_t dst_cb_index = 0;
        uint32_t dst_cb_addr = 120 * 1024;
        uint32_t cb_tiles = cb_n;  // cb can be smaller than l1 buffer.
        tt_metal::CircularBufferConfig cb_dst_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{dst_cb_index, data_format}})
                .set_page_size(dst_cb_index, single_tile_size);
        auto cb_dst = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

        uint32_t activations_addr = dst_cb_addr + (cb_tiles * single_tile_size);

        uint32_t total_tiles_size_bytes = Nt * single_tile_size;
        if (one_buffer_share) {
            total_tiles_size_bytes = shape_Nt * single_tile_size;
        }
        log_info(
            LogTest,
            "dst_cb_addr {} / {} index {} tiles, activations_addr {} / {} "
            "index {} tiles",
            dst_cb_addr,
            dst_cb_addr / 1024,
            cb_tiles,
            activations_addr,
            activations_addr / 1024,
            Nt);
        std::vector<tt_metal::Buffer> l1_buffers;

        int l1_buffers_size = 1;
        if (!(single_read || one_buffer_share)) {
            l1_buffers_size = num_cores_r * num_cores_c;
        }

        l1_buffers.reserve(l1_buffers_size);
        for (int r = 0; r < num_cores_r; ++r) {
            for (int c = 0; c < num_cores_c; ++c) {
                l1_buffers.emplace_back(device, total_tiles_size_bytes, single_tile_size, tt_metal::BufferType::L1);
                tt_metal::detail::WriteToBuffer(l1_buffers[r * num_cores_c + c], packed_tensors[r * num_cores_c + c]);

                if (single_read || one_buffer_share)
                    break;
            }
            if (single_read || one_buffer_share)
                break;
        }

        // validation
        for (int r = 0; r < num_cores_r; ++r) {
            for (int c = 0; c < num_cores_c; ++c) {
                std::vector<uint32_t> result_vec;
                tt_metal::detail::ReadFromBuffer(l1_buffers[r * num_cores_c + c], result_vec);
                auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

                if (print_tensor) {
                    print_vec(
                        result_bfp16,
                        1,
                        32,
                        std::string("from l1 buffer " + std::to_string(r) + " " + std::to_string(c)));
                    print_vec(
                        tensors[r * num_cores_c + c].get_values(),
                        1,
                        32,
                        std::string("tensor " + std::to_string(r) + " " + std::to_string(c)));
                }
                if (!(tensors[r * num_cores_c + c].get_values() == result_bfp16)) {
                    log_error(
                        LogTest,
                        "{}/{} - value read from l1 is wrong {} {}",
                        r,
                        c,
                        tensors[r * num_cores_c + c].get_values().size(),
                        result_bfp16.size());
                }

                if (single_read || one_buffer_share)
                    break;
            }
            if (single_read || one_buffer_share)
                break;
        }

        // kernel setup
        auto mm_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/old/noc/kernels/"
            "reader_global_l1.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto num_blocks = Nt / cb_n;
        for (int r = 0; r < num_cores_r; ++r) {
            for (int c = 0; c < num_cores_c; ++c) {
                CoreCoord core = {(size_t)c, (size_t)r};

                int l1_buffers_idx = (single_read || one_buffer_share) ? (0) : (r * num_cores_c + c);
                auto l1_buffer_addr = l1_buffers[l1_buffers_idx].address();

                uint32_t l1_buffer_offset = (one_buffer_share) ? ((r * num_cores_c + c) * Nt) : (0);

                if (debug) {
                    std::cout << c << "," << r << " " << l1_buffer_offset << " " << num_blocks << " " << cb_n << " "
                              << cb_n * single_tile_size << std::endl;
                }
                tt_metal::SetRuntimeArgs(
                    program, mm_reader_kernel, core, {l1_buffer_addr, l1_buffer_offset, num_blocks, cb_n});
            }
        }

        log_info(LogTest, "Running {} core test", num_cores_r * num_cores_c);
        auto begin = std::chrono::steady_clock::now();
        EnqueueProgram(device->command_queue(), &program, false);
        Finish(device->command_queue());
        auto end = std::chrono::steady_clock::now();
        auto elapsed_us = duration_cast<microseconds>(end - begin).count();
        auto bw = (total_tiles_size_bytes / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1000.0 / 1000.0);
        log_info(LogTest, "Total bytes transfered: {} Bytes", total_tiles_size_bytes);
        log_info(LogTest, "Read global to L1: {:.3f}ms, {:.3f}GB/s", elapsed_us / 1000.0, bw);
        tt_metal::DumpDeviceProfileResults(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        if (validation) {
            log_info(LogTest, "Validation");
            for (int r = 0; r < num_cores_r; ++r) {
                for (int c = 0; c < num_cores_c; ++c) {
                    std::vector<uint32_t> result_vec;
                    CoreCoord core = {(size_t)c, (size_t)r};
                    tt_metal::detail::ReadFromDeviceL1(
                        device, core, dst_cb_addr, cb_tiles * single_tile_size, result_vec);
                    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

                    int tensors_idx = (single_read || one_buffer_share) ? (0) : (r * num_cores_c + c);

                    int index = Nt;
                    if (one_buffer_share) {
                        index = (r * num_cores_c + c + 1) * Nt;
                    }

                    auto sliced_tensor =
                        slice_vec(tensors[tensors_idx].get_values(), (index - cb_tiles) * 1024, index * 1024 - 1);

                    if (print_tensor) {
                        print_vec(
                            result_bfp16,
                            32,
                            32,
                            std::string("result_bfp16 " + std::to_string(r) + " " + std::to_string(c)));

                        print_vec(
                            sliced_tensor,
                            32,
                            32,
                            std::string("sliced_tensor " + std::to_string(r) + " " + std::to_string(c)));
                    }

                    if (sliced_tensor != result_bfp16) {
                        log_error(LogTest, "{}/{} - comparision failed ", r, c);
                        pass = false;
                    } else {
                        if (debug) {
                            log_info(LogTest, "{}/{} - comparision passed", r, c);
                        }
                    }
                }
            }
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
