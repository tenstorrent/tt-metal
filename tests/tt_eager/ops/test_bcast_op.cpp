// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fmt/base.h>
#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/functions.hpp>
#include <array>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt_metal;
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
        auto device_owner = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto device = device_owner.get();
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto shapes = std::vector<ttnn::Shape>{
            ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}),
            ttnn::Shape({1, 1, TILE_HEIGHT * 2, TILE_WIDTH * 2}),
            ttnn::Shape({1, 1, TILE_HEIGHT * 3, TILE_WIDTH * 4})};

        auto run_operations = [&shapes, device] {
            for (const auto& shape : shapes) {
                for (auto bcast_dim : magic_enum::enum_values<ttnn::BcastOpDim>()) {
                    auto input_shape_a = shape;
                    if (bcast_dim == ttnn::BcastOpDim::H) {
                        input_shape_a[-1] = 32;
                    } else if (bcast_dim == ttnn::BcastOpDim::W) {
                        input_shape_a[-2] = 32;
                    } else if (bcast_dim == ttnn::BcastOpDim::HW) {
                        // do nothing
                    } else {
                        throw std::runtime_error("Unsupported Dim!");
                    }

                    Tensor a = ttnn::random::random(input_shape_a).to_layout(Layout::TILE).to_device(device);
                    Tensor b = ttnn::zeros(
                        ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), DataType::BFLOAT16, Layout::TILE, *device);

                    for (auto bcast_math : magic_enum::enum_values<ttnn::BcastOpMath>()) {
                        Tensor c = ttnn::bcast(ttnn::DefaultQueueId, a, b, bcast_math, bcast_dim);
                        Tensor d = c.cpu();

                        ////////////////////////////////////////////////////////////////////////////
                        //                      Validation & Teardown
                        ////////////////////////////////////////////////////////////////////////////
                        Tensor host_a = a.cpu();  // Move tensor a to host to validate
                        // pass &= (host_a.data() == d.data()); // src1 is all 0's
                    }
                }
            }

            {
                Tensor a = ttnn::random::random(Shape({1, 1, 32, 4544})).to_layout(Layout::TILE).to_device(device);
                Tensor b = ttnn::zeros(ttnn::Shape({1, 1, 32, 4544}), DataType::BFLOAT16, Layout::TILE, *device);
                Tensor c = ttnn::bcast(ttnn::DefaultQueueId, a, b, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::H);
                Tensor d = c.cpu();
            }

            {
                Tensor a = ttnn::random::random(Shape({1, 1, 32, 4544})).to_layout(Layout::TILE).to_device(device);
                Tensor b = ttnn::zeros(ttnn::Shape({1, 1, 32, 4544}), DataType::BFLOAT16, Layout::TILE, *device);
                Tensor c = ttnn::bcast(ttnn::DefaultQueueId, a, b, ttnn::BcastOpMath::ADD, ttnn::BcastOpDim::H);
                Tensor d = c.cpu();
            }

            {
                Tensor a = ttnn::random::random(Shape({1, 71, 32, 32})).to_layout(Layout::TILE).to_device(device);
                Tensor b = ttnn::zeros(ttnn::Shape({1, 1, 32, 32}), DataType::BFLOAT16, Layout::TILE, *device);
                Tensor c = ttnn::bcast(ttnn::DefaultQueueId, a, b, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::HW);
                Tensor d = c.cpu();
            }

            {
                Tensor a = ttnn::random::random(Shape({1, 71, 32, 64})).to_layout(Layout::TILE).to_device(device);
                Tensor b = ttnn::zeros(ttnn::Shape({1, 1, 32, 32}), DataType::BFLOAT16, Layout::TILE, *device);
                Tensor c = ttnn::bcast(ttnn::DefaultQueueId, a, b, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::HW);
                Tensor d = c.cpu();
            }
        };
        run_operations();

        run_operations();
        run_operations();
        run_operations();
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
