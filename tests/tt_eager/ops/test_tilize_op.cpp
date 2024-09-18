// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <ttnn/operations/functions.hpp>

#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt_metal;
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



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt::tt_metal::LegacyShape shape = {1, 64, 32, 64};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a =  ttnn::numpy::random::random(shape).to(device);
        Tensor b = ttnn::tilize(a);

        Tensor c = b.cpu();
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        log_debug(LogTest, "Moving src data to host to validate");
        Tensor host_a = a.cpu(); // Move tensor a to host to validate
        Tensor golden = host_a.to(Layout::TILE);
        auto golden_vec = owned_buffer::get_as<bfloat16>(golden);
        auto result_vec = owned_buffer::get_as<bfloat16>(c);
        pass &= (result_vec == golden_vec);
        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
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
