// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include <ttnn/operations/numpy/functions.hpp>

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
int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        tt::tt_metal::LegacyShape shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        Tensor a = ttnn::numpy::random::random(shape).to(Layout::TILE).to(device);
        Tensor c = ttnn::layer_norm(a, 1e-4f);
        Tensor d = c.cpu();
        Tensor host_a = a.cpu(); // Move tensor a to host to validate
        pass &= CloseDevice(device);
    } catch (const std::exception &e) {
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
