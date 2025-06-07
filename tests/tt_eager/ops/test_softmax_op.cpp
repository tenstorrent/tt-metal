// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

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
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    run_softmax(device.get(), Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}));
    run_softmax(device.get(), Shape({1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}));

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
