// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tt_dnn/op_library/program_cache.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

void run_moreh_adam(Device* device, Shape shape) {
    Tensor param_tensor = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor grad_tensor = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor exp_avg_in = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor exp_avg_sq_in = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    // Tensor max_exp_avg_sq_in = nullptr; // tt::numpy::random::random(shape).to(Layout::TILE).to(device);

    tt::operations::primary::moreh_adam(param_tensor, grad_tensor, exp_avg_in, exp_avg_sq_in,
                                    0.001f, 0.9f, 0.999f, 0.000000001f, 0.3, 1, false); // , std::nullopt); // nullptr);
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);

    run_moreh_adam(device, {1, 1, TILE_HEIGHT * 1, TILE_WIDTH * 1});
    pass &= CloseDevice(device);


    if (pass) {
        log_info(LogTest, "Test Passed1");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
