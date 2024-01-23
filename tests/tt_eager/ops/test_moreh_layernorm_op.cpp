// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "tt_eager/tt_numpy/functions.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    bool pass = true;

    const std::vector<std::string> run_types{"autoformat", "primary"};

    const std::vector<uint32_t> normalized_dims_vec{1, 2, 3, 4};

    const std::vector<std::vector<uint32_t>> input_shapes{
        {1, 1, TILE_HEIGHT, TILE_WIDTH},
        {2, 2, 2 * TILE_HEIGHT - 15, 2 * TILE_WIDTH - 17},
    };

    const std::vector<BufferType> output_buffer_types{BufferType::DRAM, BufferType::L1};

    for (const auto &input_shape : input_shapes) {
        for (const auto normalized_dims : normalized_dims_vec) {
            for (const auto &run_type : run_types) {
                for (const auto output_buffer_type : output_buffer_types) {
                    try {
                        ////////////////////////////////////////////////////////////////////////////
                        //                      Device Setup
                        ////////////////////////////////////////////////////////////////////////////
                        int device_id = 0;
                        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

                        ////////////////////////////////////////////////////////////////////////////
                        //                      Parameters Setup
                        ////////////////////////////////////////////////////////////////////////////
                        const bool is_autoformat = run_type == "autoformat";
                        const auto nan = std::numeric_limits<float>::quiet_NaN();

                        // input
                        auto input_data = tt::numpy::random::random(input_shape);
                        auto input = is_autoformat ? input_data.pad_to_tile(nan).to(device)
                                                   : input_data.pad_to_tile(nan).to(Layout::TILE).to(device);

                        log_info(
                            LogTest,
                            "N: {}, C: {}, H: {}, W: {}.",
                            input_shape[0],
                            input_shape[1],
                            input_shape[2],
                            input_shape[3]);

                        log_info(LogTest, "normalized_dims: {}", normalized_dims);

                        // gamma_beta_shape
                        Shape gamma_beta_shape = {1, 1, 1, 1};
                        const uint32_t input_dim = input_shape.size();
                        for (uint32_t i = 0; i < normalized_dims; ++i) {
                            const int64_t dim = input_dim - i - 1;
                            gamma_beta_shape[dim] = input_shape[dim];
                        }

                        // gamma
                        auto gamma_data = tt::numpy::zeros(gamma_beta_shape);
                        auto gamma = is_autoformat ? gamma_data.pad_to_tile(nan).to(device)
                                                   : gamma_data.pad_to_tile(nan).to(Layout::TILE).to(device);

                        // beta
                        auto beta_data = tt::numpy::ones(gamma_beta_shape);
                        auto beta = is_autoformat ? beta_data.pad_to_tile(nan).to(device)
                                                  : beta_data.pad_to_tile(nan).to(Layout::TILE).to(device);

                        // Validation
                        auto expected = tt::numpy::ones(input_shape);

                        // y * 0 + 1
                        log_info(LogTest, "{} test start.", run_type);
                        auto actual_npu = is_autoformat ? tt_metal::moreh_layernorm(
                                                              input,
                                                              normalized_dims,
                                                              1e-5f,
                                                              gamma,
                                                              beta,
                                                              std::nullopt,
                                                              std::nullopt,
                                                              {.buffer_type = output_buffer_type})
                                                        : operations::primary::moreh_layernorm(
                                                              input,
                                                              normalized_dims,
                                                              1e-5f,
                                                              gamma,
                                                              beta,
                                                              std::nullopt,
                                                              std::nullopt,
                                                              {.buffer_type = output_buffer_type});

                        pass &= actual_npu.buffer()->buffer_type() == output_buffer_type;

                        auto actual = actual_npu.cpu().to(Layout::ROW_MAJOR).unpad_from_tile(input_shape);

                        pass &= CloseDevice(device);
                        pass &= tt::numpy::allclose<bfloat16>(actual, expected);
                        if (pass) {
                            log_info(LogTest, "{} test passed.", run_type);
                        } else {
                            TT_THROW("{} test failed.", run_type);
                        }
                    } catch (const std::exception &e) {
                        pass = false;
                        log_error(LogTest, "{}", e.what());
                        log_error(LogTest, "System error message: {}", std::strerror(errno));
                    }
                }  // output_buffer_type
            }      // run_type
        }          // normalized_dims
    }              // input_shape

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
