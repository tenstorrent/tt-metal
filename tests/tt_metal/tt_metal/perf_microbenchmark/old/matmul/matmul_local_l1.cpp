// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdlib.h>
#include <sys/types.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <ratio>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"

#define LAUNCH

using std::vector;
using namespace tt;

std::vector<bfloat16> get_row_slice(
    std::vector<bfloat16> data, int total_row_slices, int row_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int rows_per_slice = rows / total_row_slices;

    for (int i = rows_per_slice * row_slice_index * cols; i < rows_per_slice * (row_slice_index + 1) * cols; i++) {
        result.push_back(data.at(i));
    }
    return result;
}

std::vector<bfloat16> get_col_slice(
    std::vector<bfloat16> data, int total_col_slices, int col_slice_index, int rows, int cols) {
    std::vector<bfloat16> result;
    int cols_per_slice = cols / total_col_slices;

    for (int r = 0; r < rows; r++) {
        for (int c = cols_per_slice * col_slice_index; c < cols_per_slice * (col_slice_index + 1); c++) {
            result.push_back(data.at(r * cols + c));
        }
    }
    return result;
}

// Transpose 2D matrix of tiles so that its column major of tiles instead of row
// major. this is usually used for activation so that blocks data is contiguous
// in memory until we have a more generalized read kernel that can read tiles
// from different location in memory to make up a block in the activations CB
std::vector<std::uint32_t> transpose_tiles(
    std::vector<std::uint32_t> data, int row_tiles, int col_tiles, int in0_block_w) {
    std::vector<std::uint32_t> result;
    int tile_size = 512;
    for (int c = 0; c < col_tiles; c += in0_block_w) {
        for (int r = 0; r < row_tiles; r++) {
            for (int k = 0; k < in0_block_w; k++) {
                int offset = tile_size * col_tiles * r + c * tile_size + k * tile_size;
                for (int i = 0; i < tile_size; i++) {
                    result.push_back(data.at(offset + i));
                }
            }
        }
    }
    return result;
}

std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K, int N) {
    if (N == K) {
        return data;
    }
    std::vector<bfloat16> result;
    if (N > K) {
        for (int i = 0; i < M * 32; i++) {
            for (int j = 0; j < K * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
            for (int j = 0; j < (N - K) * 32; j++) {
                result.push_back((float)0);
            }
        }
    } else {
        for (int i = 0; i < M * 32; i++) {
            for (int j = 0; j < N * 32; j++) {
                int offset = i * K * 32;
                result.push_back(data.at(offset + j));
            }
        }
    }
    return result;
}

template <typename T>
std::vector<T> slice(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

int main(int argc, char** argv) {
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
        uint32_t cb;
        uint32_t Mt;
        uint32_t Nt;
        uint32_t Kt;
        uint32_t num_cores_r;
        uint32_t num_cores_c;
        uint32_t validation;
        try {
            std::tie(debug, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--debug", 0);
            std::tie(dprint, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--dprint", 0);
            std::tie(print_tensor, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--print_tensor", 0);
            std::tie(Mt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--mt", 72);
            std::tie(Nt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--nt", 96);
            std::tie(Kt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--kt", 24);
            std::tie(num_cores_r, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--r", 9);
            std::tie(num_cores_c, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--c", 12);
            std::tie(validation, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--validation", 1);
        } catch (const std::exception& e) {
            TT_THROW("Command line arguments found exception", e.what());
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Inputs Setup
        ////////////////////////////////////////////////////////////////////////////
        if (debug) {
            log_info(LogTest, "row {} x col {} = {} cores", num_cores_r, num_cores_c, num_cores_r * num_cores_c);
        }

        log_info(LogTest, "M = {}, N = {}, K = {}", Mt, Nt, Kt);
        log_info(LogTest, "Activation = {}x{}", Mt * 32, Kt * 32);
        log_info(LogTest, "Weights = {}x{}", Kt * 32, Nt * 32);

        if (Mt % num_cores_r != 0) {
            TT_THROW("Mt {} must be a multiple of num_cores_r {}", Mt, num_cores_r);
            TT_FATAL(false, "Error");
        }

        if (Nt % num_cores_c != 0) {
            TT_THROW("Nt {} must be a multiple of num_cores_c {}", Nt, num_cores_c);
            TT_FATAL(false, "Error");
        }

        tt::DataFormat data_format = tt::DataFormat::Float16_b;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t per_core_Mt = Mt / num_cores_r;
        uint32_t per_core_Nt = Nt / num_cores_c;
        uint32_t per_core_activations_tiles = per_core_Mt * Kt;
        uint32_t per_core_weights_tiles = per_core_Nt * Kt;
        uint32_t per_core_output_tiles = per_core_Mt * per_core_Nt;

        uint32_t activations_addr = 120 * 1024;
        uint32_t weights_addr = activations_addr + (per_core_activations_tiles * single_tile_size);
        uint32_t output_addr = weights_addr + (per_core_weights_tiles * single_tile_size);

        if (debug) {
            log_info(LogTest, "per core M = {}, N = {}, K = {}", per_core_Mt, per_core_Nt, Kt);
            log_info(
                LogTest,
                "per core activations tiles = {}, weights tiles = {}",
                per_core_activations_tiles,
                per_core_weights_tiles);
        }

        log_info(LogTest, "activations_addr ({} x 1024) {} tiles", activations_addr / 1024, per_core_activations_tiles);
        log_info(LogTest, "weights_addr ({} x 1024) {} tiles", weights_addr / 1024, per_core_weights_tiles);
        log_info(LogTest, "output_addr ({} x 1024) {} tiles", output_addr / 1024, per_core_output_tiles);

        if (output_addr + (per_core_output_tiles * single_tile_size) > 1024 * 1024) {
            log_error(LogTest, "inputs and output CBs don't fit in L1");
            TT_FATAL(false, "Error");
        }

        SHAPE shape = {1, 1, Mt * 32, Kt * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
            shape,
            tt::deprecated::Initialize::RANDOM,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        auto identity = create_identity_matrix(Kt * 32, Nt * 32, std::min(Kt, Nt) * 32);  // bflaot16 identity

        if (print_tensor) {
            print_vec_of_bfloat16(tensor.get_values(), 1, "Activation first row");
            print_vec_of_bfloat16(identity, 1, "Weights first row");
        }

        log_info(LogTest, "Slicing input tensors and copying them to L1");
        for (int r = 0; r < num_cores_r; r++) {
            std::vector<bfloat16> activation_slice =
                get_row_slice(tensor.get_values(), num_cores_r, r, Mt * 32, Kt * 32);
            for (int c = 0; c < num_cores_c; c++) {
                std::vector<bfloat16> weights_slice = get_col_slice(identity, num_cores_c, c, Kt * 32, Nt * 32);

                CoreCoord core = {(std::size_t)c, (std::size_t)r};
                auto activations_tilized = tilize_swizzled(activation_slice, per_core_Mt * 32, Kt * 32);
                auto activations_tile_layout =
                    convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
                auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
                pass &= tt_metal::detail::WriteToDeviceL1(device, core, activations_addr, activations);
                TT_FATAL(pass, "Error");

                auto identity_tilized = tilize_swizzled(weights_slice, Kt * 32, per_core_Nt * 32);
                auto weights_tile_layout =
                    convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity_tilized));
                auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
                auto weights_tile_transposed = transpose_tiles(weights, Kt, per_core_Nt, 1);
                pass &= tt_metal::detail::WriteToDeviceL1(device, core, weights_addr, weights_tile_transposed);
                TT_FATAL(pass, "Error");
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
        const CoreRange all_cores(start_core, end_core);

        log_info(LogTest, "start_core {},{}", start_core.x, start_core.y);
        log_info(LogTest, "end_core {},{}", end_core.x, end_core.y);

        // CB creation
        uint32_t cb_activations_index = 0;
        uint32_t cb_activations_tiles = per_core_activations_tiles;
        tt_metal::CircularBufferConfig cb_activations_config =
            tt_metal::CircularBufferConfig(
                cb_activations_tiles * single_tile_size, {{cb_activations_index, data_format}})
                .set_page_size(cb_activations_index, single_tile_size);
        auto cb_activations = tt_metal::CreateCircularBuffer(program, all_cores, cb_activations_config);

        uint32_t cb_weights_index = 1;
        uint32_t cb_weights_tiles = per_core_weights_tiles;
        tt_metal::CircularBufferConfig cb_weights_config =
            tt_metal::CircularBufferConfig(cb_weights_tiles * single_tile_size, {{cb_weights_index, data_format}})
                .set_page_size(cb_weights_index, single_tile_size);
        auto cb_weights = tt_metal::CreateCircularBuffer(program, all_cores, cb_weights_config);

        uint32_t cb_output_index = 16;
        uint32_t cb_output_tiles = per_core_output_tiles;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(cb_output_tiles * single_tile_size, {{cb_output_index, data_format}})
                .set_page_size(cb_output_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

        // compute kernel setup
        vector<uint32_t> compute_kernel_args = {uint(per_core_Mt), uint(Kt), uint(per_core_Nt)};

        auto mm_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/"
            "compute_local_l1.cpp",
            all_cores,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_kernel_args});

        std::chrono::duration<double, std::nano> duration;
        // took from run_operation.cpp
        auto start = std::chrono::high_resolution_clock::now();
        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        tt_metal::detail::DumpDeviceProfileResults(device);

        uint64_t num_of_matmul_ops =
            (2 * static_cast<uint64_t>(Kt) * 32 - 1) * (static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) * 1024);
        if (debug) {
            log_info(LogTest, "number of matmul ops: {}", num_of_matmul_ops);
        }

        double tflops = static_cast<double>(num_of_matmul_ops) / duration.count() / 1000;
        log_info(LogTest, "time duration: {} ns, TFLOPS {}", duration.count(), tflops);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto golden = select_columns(tensor.get_values(), Mt, Kt, Nt);
        if (validation) {
            log_info(LogTest, "Validation");
            for (int r = 0; r < num_cores_r; ++r) {
                auto golden_row = get_row_slice(golden, num_cores_r, r, Mt * 32, Nt * 32);
                for (int c = 0; c < num_cores_c; ++c) {
                    auto per_core_golden = get_col_slice(golden_row, num_cores_c, c, per_core_Mt * 32, Nt * 32);

                    CoreCoord core = {(size_t)c, (size_t)r};

                    std::vector<uint32_t> result_vec;
                    tt_metal::detail::ReadFromDeviceL1(
                        device, core, output_addr, cb_output_tiles * single_tile_size, result_vec);
                    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
                    auto result_flat_layout =
                        convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
                    auto result_untilized = untilize_swizzled(result_flat_layout, per_core_Mt * 32, per_core_Nt * 32);

                    if (print_tensor) {
                        print_vec_of_bfloat16(
                            result_untilized, 1, "result_untilized" + std::to_string(r) + " " + std::to_string(c));
                    }

                    if (!(per_core_golden == result_untilized)) {
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

    } catch (const std::exception& e) {
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

    TT_FATAL(pass, "Error");

    return 0;
}
