// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;

using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;

void test_operation_infrastructure() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    tt::log_info(tt::LogTest, "Running {}", __func__);
    using namespace tt::tt_metal;

    tt::tt_metal::Array4D input_shape = {1, 1, 18, 13};
    tt::tt_metal::Array4D padded_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    auto input_tensor = ttnn::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);
    auto output_tensor = ttnn::pad(input_tensor, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    TT_FATAL(output_tensor.get_padded_shape() == padded_shape, "Error");
    TT_FATAL(output_tensor.get_logical_shape() == input_shape, "Error");
}

int main(int argc, char** argv) {
    test_operation_infrastructure();
    return 0;
}
