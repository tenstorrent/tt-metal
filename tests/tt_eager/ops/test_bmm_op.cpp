// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "common/constants.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>

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
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t Mt = 3;
        uint32_t Kt = 2;
        uint32_t Nt = 4;
        uint32_t B = 5;
        Shape shapea = {B, 1, Mt*TILE_HEIGHT, Kt*TILE_WIDTH};
        Shape shapeb = {B, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};
        Shape shapeb1 = {1, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = tt::numpy::random::random(shapea).to(Layout::TILE).to(device);
        Tensor b = tt::numpy::zeros(shapeb, DataType::BFLOAT16).to(Layout::TILE).to(device);
        Tensor b1 = tt::numpy::zeros(shapeb1, DataType::BFLOAT16).to(Layout::TILE).to(device);

        Tensor mm = ttnn::operations::matmul::matmul(a, b, /*bias=*/std::nullopt,
                ttnn::operations::matmul::Matmul{/*program_config=*/std::nullopt, /*bcast_batch=*/std::nullopt,operation::DEFAULT_OUTPUT_MEMORY_CONFIG, /*output_dtype=*/std::nullopt, /*compute_kernel_config=*/std::nullopt, /*untilize_out=*/false, /*user_core_coord=*/std::nullopt, /*user_fused_activation=*/std::nullopt, /*user_run_batched=*/true}).cpu();
        Tensor mm1 = ttnn::operations::matmul::matmul(a, b1).cpu();

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.cpu(); // Move tensor a to host to validate

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
