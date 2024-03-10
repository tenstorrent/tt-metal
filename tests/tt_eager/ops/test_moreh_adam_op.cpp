// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

void run_moreh_adam(Device* device, Shape shape,
    float lr, float beta1, float beta2, float eps, float weight_decay, uint32_t step, bool amsgrad) {
    Tensor param_tensor = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor grad_tensor = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor exp_avg_in = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor exp_avg_sq_in = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    if (amsgrad) {
        Tensor max_exp_avg_sq_in = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
        tt::operations::primary::moreh_adam(param_tensor, grad_tensor, exp_avg_in, exp_avg_sq_in,
                                    lr, beta1, beta2, eps, weight_decay, step, amsgrad, max_exp_avg_sq_in);
    }
    else {
        tt::operations::primary::moreh_adam(param_tensor, grad_tensor, exp_avg_in, exp_avg_sq_in,
                                    lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    }
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

    float lr = 0.01;
    float beta1 = 0.8;
    float beta2 = 0.888;
    float eps = 1e-06;
    float weight_decay = 0.1;
    int32_t step = 1;
    bool amsgrad = false;

    run_moreh_adam(device, {1, 1, TILE_HEIGHT * 1, TILE_WIDTH * 1}
                        , lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    run_moreh_adam(device, {1, 1, TILE_HEIGHT * 1, TILE_WIDTH * 1}
                        , lr, beta1, beta2, eps, weight_decay, step, !amsgrad);
    run_moreh_adam(device, {1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}
                        , lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    run_moreh_adam(device, {1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}
                        , lr, beta1, beta2, eps, weight_decay, step, !amsgrad);
    run_moreh_adam(device, {12, 6, TILE_HEIGHT * 2, TILE_WIDTH * 2}
                        , lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    run_moreh_adam(device, {12, 6, TILE_HEIGHT * 2, TILE_WIDTH * 2}
                        , lr, beta1, beta2, eps, weight_decay, step, !amsgrad);
    pass &= CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
