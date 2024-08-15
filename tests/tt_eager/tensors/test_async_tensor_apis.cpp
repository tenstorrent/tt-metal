// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/types.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

TEST_F(CommonFixture, TestTensorOwnershipSanity) {
}

TEST_F(CommonFixture, TestAsyncEltwiseBinary) {
}


TEST_F(CommonFixture, TestAsyncRefCountManager) {
}

// TEST_F(CommonFixture, TestAsyncEltwiseBinaryAutoFormat) {
//     // Test usecase where both inputs and outputs are on host and autoformat is used
//     Device* device = this->devices_[0];
//     device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
//     AutoFormat::SetDefaultDevice(device);

//     for (int i = 0; i < 5; i++) {
//         // Initialize tensors and keep them on host. Since none of the tensors are divisible by tile dims, the inputs
//         // and outputs are on host.
//         Tensor input_tensor_a =
//             tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
//         Tensor input_tensor_b =
//             tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
//         Tensor input_tensor_c =
//             tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
//         Tensor output_tensor_device = ttnn::multiply(ttnn::add(input_tensor_a, input_tensor_b), input_tensor_c);
//         Tensor output_tensor_device_2 = neg(ttnn::subtract(output_tensor_device, input_tensor_c));

//         EXPECT_EQ(output_tensor_device.get_shape(), ttnn::Shape(Shape({1, 1, 1023, 1023})));
//         EXPECT_EQ(output_tensor_device.get_dtype(), DataType::BFLOAT16);

//         Tensor output_tensor_host = output_tensor_device_2.cpu();
//         // Verify output data
//         auto& buf =
//             std::get<owned_buffer::Buffer<bfloat16>>(std::get<OwnedStorage>(output_tensor_host.get_storage()).buffer);
//         for (int j = 0; j < 1023 * 1023; j++) {
//             EXPECT_EQ(bfloat16(buf[j]), bfloat16(static_cast<float>(i - 2 * i * i)));
//         }
//     }
//     device->set_worker_mode(WorkExecutorMode::SYNCHRONOUS);
// }

TEST_F(CommonFixture, TestTensorAsyncDataMovement) {
}
