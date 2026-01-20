// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/functions.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_2d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 2;

    return pass;
}

bool test_3d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 3;

    return pass;
}

bool test_4d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({2, 3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 4;

    return pass;
}

bool test_5d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({2, 2, 3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 5;

    return pass;
}

bool test_6d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({2, 2, 2, 3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 6;

    return pass;
}

bool test_7d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({2, 2, 2, 2, 3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 7;

    return pass;
}

bool test_8d_tensor(distributed::MeshDevice* device) {
    bool pass = true;

    ttnn::Shape shape({2, 2, 2, 2, 2, 3, 30, 30});
    Tensor tensor = ttnn::random::random(shape);
    tensor = tensor.pad_to_tile(0.0f);
    tensor = tensor.to_layout(Layout::TILE);
    tensor = tensor.to_device(device);
    pass &= tensor.logical_shape().rank() == 8;

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device_owner = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto device = device_owner.get();

        pass &= test_2d_tensor(device);
        pass &= test_3d_tensor(device);
        pass &= test_4d_tensor(device);
        pass &= test_5d_tensor(device);
        pass &= test_6d_tensor(device);
        pass &= test_7d_tensor(device);
        pass &= test_8d_tensor(device);
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
