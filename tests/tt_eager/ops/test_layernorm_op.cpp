// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include <ttnn/operations/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <optional>

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
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
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    log_info(LogTest, "Test Passed");

    return 0;
}
