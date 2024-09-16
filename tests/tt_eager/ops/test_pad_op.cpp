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
#include "tt_numpy/functions.hpp"

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

    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);
    auto output_tensor = ttnn::pad(input_tensor, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    auto output_shape = output_tensor.get_legacy_shape();
    TT_FATAL(output_shape == tt::tt_metal::LegacyShape(padded_shape));
    TT_FATAL(output_shape.without_padding() == tt::tt_metal::LegacyShape(input_shape));
}

int main(int argc, char** argv) {
    test_operation_infrastructure();
    return 0;
}
