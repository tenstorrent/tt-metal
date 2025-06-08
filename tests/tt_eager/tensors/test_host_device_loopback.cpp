// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstring>
#include <exception>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_single_tile_single_dram_bank_loopback(distributed::MeshDevice* device) {
    bool pass = true;
    ttnn::Shape single_tile_shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    Tensor host_a = ttnn::random::random(single_tile_shape).to_layout(Layout::TILE);
    Tensor device_a = host_a.to_device(device);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = host_buffer::get_as<bfloat16>(host_a);
    auto loopbacked_a_data = host_buffer::get_as<bfloat16>(loopbacked_a);
    pass &= std::equal(host_a_data.begin(), host_a_data.end(), loopbacked_a_data.begin());

    return pass;
}

bool test_multi_tile_multi_dram_bank_loopback(distributed::MeshDevice* device) {
    bool pass = true;
    ttnn::Shape multi_tile_shape({1, 1, 4 * TILE_HEIGHT, 3 * TILE_WIDTH});

    Tensor host_a = ttnn::random::random(multi_tile_shape).to_layout(Layout::TILE);
    Tensor device_a = host_a.to_device(device);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = host_buffer::get_as<bfloat16>(host_a);
    auto loopbacked_a_data = host_buffer::get_as<bfloat16>(loopbacked_a);
    pass &= std::equal(host_a_data.begin(), host_a_data.end(), loopbacked_a_data.begin());
    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

        pass &= test_single_tile_single_dram_bank_loopback(device.get());

        pass &= test_multi_tile_multi_dram_bank_loopback(device.get());
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
