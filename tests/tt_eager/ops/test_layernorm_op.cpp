// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/functions.hpp>
#include <cstring>
#include <exception>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
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
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});
        Tensor a = ttnn::random::random(shape).to_layout(Layout::TILE).to_device(device.get());
        Tensor c = ttnn::layer_norm(a, 1e-4f);
        Tensor d = c.cpu();
        Tensor host_a = a.cpu();  // Move tensor a to host to validate
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
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
