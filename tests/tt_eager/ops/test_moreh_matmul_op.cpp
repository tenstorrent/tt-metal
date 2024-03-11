// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/constants.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

Tensor diagonal(const Shape &shape, float value) {
    Tensor tensor = tt::numpy::zeros(shape);
    auto buffer = owned_buffer::get_as<bfloat16>(tensor);
    for (int i = 0; i < shape[0] * shape[1]; ++i) {
        for (int j = 0; j < std::min(shape[2], shape[3]); j++) {
            buffer[i * shape[2] * shape[3] + j * shape[3] + j] = bfloat16(value);
        }
    }
    return tensor;
}

static bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
    auto result = diff < std::max(abs_threshold, epsilon * norm);
    return result;
}

template <typename... Args>
static bool nearly_equal(bfloat16 a, bfloat16 b, Args... args) {
    return nearly_equal(a.to_float(), b.to_float(), args...);
}

inline bool compare(
    const Tensor &tensor_a,
    const Tensor &tensor_b,
    int B1,
    int B2,
    int M,
    int N,
    int K,
    int in0_B1,
    int in0_B2,
    bool print = false) {
    auto tensor_a_buffer = owned_buffer::get_as<bfloat16>(tensor_a);
    auto tensor_b_buffer = owned_buffer::get_as<bfloat16>(tensor_b);

    // debug print
    int print_cnt = 0;
    int print_cnt2 = 0;
    int count = 0;
    int print_limit = 10;

    int MN = M * N;
    int B2MN = B2 * MN;

    int MK = M * K;
    int in0_B2MK = in0_B2 * MK;

    for (int b1 = 0; b1 < B1; ++b1) {
        for (int b2 = 0; b2 < B2; ++b2) {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    int a_b1 = (b1 >= in0_B1) ? (0) : (b1);
                    int a_b2 = (b2 >= in0_B2) ? (0) : (b2);

                    int a_index = a_b1 * in0_B2MK + a_b2 * MK + m * K + n;
                    int b_index = b1 * B2MN + b2 * MN + m * N + n;

                    if (n >= K) {
                        if (tensor_b_buffer[b_index] != 0) {
                            count++;
                            if (print && print_cnt++ < print_limit) {
                                log_error(
                                    LogTest,
                                    "(b1, b2, m, n) = ({}, {}, {}, {}), output {} should be zero.",
                                    b1,
                                    b2,
                                    m,
                                    n,
                                    tensor_b_buffer[b_index]);
                            }
                        }
                        continue;
                    }
                    if (not nearly_equal(tensor_a_buffer[a_index], tensor_b_buffer[b_index])) {
                        count++;
                        if (print && print_cnt2++ < print_limit) {
                            log_error(
                                LogTest,
                                "(b1, b2, m, n) = ({}, {}, {}, {}), activation {} != output {}",
                                b1,
                                b2,
                                m,
                                n,
                                tensor_a_buffer[a_index],
                                tensor_b_buffer[b_index]);
                        }
                    }
                }
            }
        }
    }

    if (count) {
        if (print) {
            log_error(LogTest, "{} diffs", count);
        }
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t a_Mt, a_Kt, b_Kt, b_Nt, a_B1, a_B2, b_B1, b_B2, transpose_b;
        std::tie(a_B1, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--a-b1", 1);
        std::tie(a_B2, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--a-b2", 1);
        std::tie(a_Mt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--a-mt", 3);
        std::tie(a_Kt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--a-kt", 2);
        std::tie(b_B1, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--b-b1", 1);
        std::tie(b_B2, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--b-b2", 1);
        std::tie(b_Kt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--b-kt", 2);
        std::tie(b_Nt, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--b-nt", 4);
        std::tie(transpose_b, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--trans-b", 0);

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        Shape shapea = {a_B1, a_B2, a_Mt * TILE_HEIGHT, a_Kt * TILE_WIDTH};
        Shape shapeb = {b_B1, b_B2, b_Kt * TILE_HEIGHT, b_Nt * TILE_WIDTH};
        Shape shapec = {b_B1, b_B2, a_Mt * TILE_HEIGHT, b_Nt * TILE_WIDTH};

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = tt::numpy::random::random(shapea).to(Layout::TILE).to(device);
        Tensor b = diagonal(shapeb, 1.0f).to(Layout::TILE).to(device);
        Tensor output_tensor_cpu = create_device_tensor(shapec, a.get_dtype(), Layout::TILE, device, a.memory_config());
        Tensor out_cpu = tt::operations::primary::moreh_matmul(a, b, std::nullopt, false, static_cast<bool>(transpose_b)).cpu();

        const auto expected_storage = std::get<DeviceStorage>(output_tensor_cpu.get_storage());
        Tensor with_output_tensor = tt::operations::primary::moreh_matmul(a, b, output_tensor_cpu, false, static_cast<bool>(transpose_b));
        const auto actual_storage = std::get<DeviceStorage>(with_output_tensor.get_storage());
        Tensor out_cpu_with_output_tensor = with_output_tensor.cpu();
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        const auto &out_shape = out_cpu.get_legacy_shape();
        log_info(
            LogTest,
            "out_cpu shape {} - {}, {}, {}, {}",
            out_shape.rank(),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3]);

        const auto &out_cpu_with_output_tensor_shape = out_cpu_with_output_tensor.get_legacy_shape();
        log_info(
            LogTest,
            "out_cpu_with_output_tensor shape {} - {}, {}, {}, {}",
            out_cpu_with_output_tensor_shape.rank(),
            out_cpu_with_output_tensor_shape[0],
            out_cpu_with_output_tensor_shape[1],
            out_cpu_with_output_tensor_shape[2],
            out_cpu_with_output_tensor_shape[3]);

        pass &= compare(
            a.cpu().to(Layout::ROW_MAJOR),
            out_cpu.to(Layout::ROW_MAJOR),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            shapea[3],
            shapea[0],
            shapea[1],
            true);

        pass &= out_shape == out_cpu_with_output_tensor_shape;
        pass &= expected_storage.buffer.get() == actual_storage.buffer.get();
        pass &= compare(
            a.cpu().to(Layout::ROW_MAJOR),
            out_cpu_with_output_tensor.to(Layout::ROW_MAJOR),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            shapea[3],
            shapea[0],
            shapea[1],
            true);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        TT_THROW("System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
