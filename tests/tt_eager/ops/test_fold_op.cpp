// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <ttnn/operations/functions.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/fold/fold.hpp"
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace constants;

void run_fold(tt::tt_metal::IDevice* device, const ttnn::Shape& shape) {
    ttnn::Tensor input_tensor = ttnn::random::random(shape).to_layout(ttnn::Layout::ROW_MAJOR).to_device(device);
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    ttnn::Tensor device_output_tensor = ttnn::fold(ttnn::DefaultQueueId, input_tensor, stride_h, stride_w);
    ttnn::Tensor output_tensor = device_output_tensor.cpu();
}

int main(int argc, char** argv) {
    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    run_fold(device, ttnn::Shape({1, 2, 2, 2}));
    bool pass = CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
