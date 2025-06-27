// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

void run_softmax(distributed::MeshDevice* device, const ttnn::Shape& shape) {
    Tensor input_tensor = ttnn::random::random(shape).to_layout(Layout::TILE).to_device(device);
    Tensor device_output_tensor = ttnn::softmax_in_place(input_tensor);
    Tensor output_tensor = device_output_tensor.cpu();
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device_owner = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto device = device_owner.get();
        // https://github.com/tenstorrent/tt-metal/issues/23824
        device->disable_and_clear_program_cache();

        run_softmax(device, Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}));
        run_softmax(device, Shape({1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}));
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
