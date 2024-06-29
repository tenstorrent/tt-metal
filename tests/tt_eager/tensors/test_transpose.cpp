// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "tensor/host_buffer/functions.hpp"
#include "tensor/host_buffer/types.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_tensor_transpose_non_tile_aligned(Device *device) {
    bool pass = true;

    const size_t tensor_width = 16;
    const size_t tensor_height = 256;

    auto buffer = tt::tt_metal::owned_buffer::create(
        create_random_vector_of_bfloat16_native(
            tensor_width * tensor_height * 2
            , 2, 42, -1));
    auto x = tt::tt_metal::Tensor(
        OwnedStorage{std::move(buffer)},
        {1, 1,tensor_width, tensor_height},
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::Layout::ROW_MAJOR);
    x = tt::tt_metal::tilize_with_zero_padding(x.to(device));


    auto y = tt::tt_metal::transpose(x, 2, 3);
    pass &= (y.shape()[0] == 1 && y.shape()[1] == 1 && y.shape()[2] == tensor_height && y.shape()[3] == tensor_width);

    return pass;
}


int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        pass &= test_tensor_transpose_non_tile_aligned(device);


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

    TT_FATAL(pass);

    return 0;
}
