// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <vector>
#include "assert.hpp"
#include "small_vector.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "dispatch_core_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace conv::conv2d {
namespace test {

struct Conv2DParam {};

class Conv2DFixture : public ::testing::Test, public testing::WithParamInterface<Conv2DParam> {};

float pcc(std::vector<float>& x, std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    int n = x.size();
    float mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    float numerator = 0, sum_sq_x = 0, sum_sq_y = 0;
    for (int i = 0; i < n; ++i) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    float denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator == 0) {
        return 0;
    }

    return numerator / denominator;
}

// returns flattened output for easier PCC calculation
std::vector<float> conv2d(
    const std::vector<float>& input,   // (1,input_channels,input_height,input_width)
    const std::vector<float>& kernel,  // (output_channels,input_channels,kernel_height,kernel_width)
    const uint32_t input_channels,
    const uint32_t output_channels,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding) {
    auto [kernel_height, kernel_width] = kernel_size;
    auto [padding_height, padding_width] = padding;
    auto [stride_height, stride_width] = stride;
    auto Xk = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    auto Xw = (input_width - kernel_width + 2 * padding_width) / stride_width + 1;

    std::vector<float> output = std::vector<float>(output_channels * Xk * Xw);
    for (int co = 0, i = 0; co < output_channels; co++) {
        for (int h = 0; h < input_height; h += stride_height) {
            std::vector<float> row;
            for (int w = 0; w < input_width; w += stride_width) {
                float sum = 0;
                for (int ci = 0; ci < input_channels; ci++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            if (h + kh - padding_height >= 0 && h + kh - padding_height < input_height &&
                                w + kw - padding_width >= 0 && w + kw - padding_width < input_width) {
                                sum += input
                                           [ci * input_height * input_width + (h + kh - padding_height) * input_width +
                                            w + kw - padding_width] *
                                       kernel
                                           [co * input_channels * kernel_height * kernel_width +
                                            ci * kernel_height * kernel_width + kh * kernel_width + kw];
                            }
                        }
                    }
                }
                output[i] = sum;
                i++;
            }
        }
    }
    return output;
}

TEST_P(Conv2DFixture, Conv2DCalculateCorrectly) {
    const auto device_id = 0;
    IDevice* device = CreateDevice(device_id, 1, 16384, 0, DispatchCoreConfig(DispatchCoreType::ETH));

    const uint32_t input_channels = 32;   // in_channels
    const uint32_t output_channels = 32;  // out_channels

    const uint32_t input_height = 256;                   // input_height
    const uint32_t input_width = 512;                    // input_width
    const uint32_t batch_size = 1;                       // batch_size
    const std::array<uint32_t, 2> kernel_size = {3, 3};  // kernel_size
    const std::array<uint32_t, 2> stride = {1, 1};       // stride
    const std::array<uint32_t, 2> padding = {1, 1};      // padding

    MemoryConfig dram_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

    std::array<uint32_t, 4> dimensions = {batch_size, input_channels, input_height, input_width};
    std::array<uint32_t, 4> dimensions_weight = {output_channels, input_channels, kernel_size[0], kernel_size[1]};

    random::seed(42);
    Tensor input_tensor =
        ttnn::random::random(Shape(dimensions), tt::tt_metal::DataType::BFLOAT16).to_device(device, dram_mem_config);
    Tensor weight_tensor = ttnn::random::random(Shape(dimensions_weight), tt::tt_metal::DataType::BFLOAT16);
    std::vector<float> input_vector = input_tensor.to_vector<float>();
    std::vector<float> weight_vector = weight_tensor.to_vector<float>();

    // (N,C,H,W) -> (N,H,W,C)
    input_tensor = ttnn::permute(input_tensor, SmallVector<int64_t>{0, 2, 3, 1});

    Result r = conv2d::conv2d(
        input_tensor,
        weight_tensor,
        device,
        input_channels,
        output_channels,
        batch_size,  // ba
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        {1, 1},  // dilation
        1,       // groups
        std::nullopt,
        conv2d::Conv2dConfig{
            // because the default is TILE layout, row major layout is easier to compare with refference implementation
            .output_layout = ROW_MAJOR_LAYOUT,
        },
        std::nullopt);
    auto [output_tensor, output_height, output_width, r1, r2] = r;

    // (1,1,HW,C) -> (1,H,W,C)
    output_tensor = ttnn::reshape(output_tensor, Shape({1, output_height, output_width, output_channels}));
    // (1,H,W,C) -> (1,C,H,W)
    output_tensor = ttnn::permute(output_tensor, SmallVector<int64_t>{0, 3, 1, 2});

    std::vector<float> res = output_tensor.to_vector<float>();

    auto ref_res = conv2d(
        input_vector,
        weight_vector,
        input_channels,
        output_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding);

    auto pcc_calculated = pcc(res, ref_res);

    auto pass = CloseDevice(device);
    std::cout << "PCC: " << pcc_calculated << std::endl;
    TT_FATAL(pass, "Error");
    TT_FATAL(pcc_calculated > 0.99, "Failed pcc");
}

INSTANTIATE_TEST_SUITE_P(Conv2DTests, Conv2DFixture, ::testing::Values(Conv2DParam{}));

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
