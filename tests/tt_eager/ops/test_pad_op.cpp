// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/operation.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;

using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;

void test_operation_infrastructure() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    tt::log_info(tt::LogTest, "Running {}", __func__);
    using namespace tt::tt_metal;

    auto input_shape = Shape{1, 1, 18, 13};
    auto padded_shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};

    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);
    auto output_tensor = operation::run(PadOnHost{padded_shape, {0, 0, 0, 0}, 0}, {input_tensor}).at(0);

    auto output_shape = output_tensor.get_legacy_shape();
    TT_FATAL(output_shape == padded_shape);
    TT_FATAL(output_shape.without_padding() == input_shape);
}

int main(int argc, char** argv) {
    test_operation_infrastructure();
    return 0;
}
