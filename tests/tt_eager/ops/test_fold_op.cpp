// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <tt_numpy/functions.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

void run_fold(Device *device, Shape shape) {
    Tensor input_tensor = tt::numpy::random::random(shape).to(Layout::ROW_MAJOR).to(device);
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint8_t queue_id = 0;
    Tensor device_output_tensor = ttnn::fold(queue_id, input_tensor, stride_h, stride_w);
    Tensor output_tensor = device_output_tensor.cpu();
}

int main(int argc, char **argv) {
    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);

    run_fold(device, {1, 2, 2, 2});
    bool pass = CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
