// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>
#include <random>
#include "assert.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "dispatch_core_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace conv::conv2d {
namespace test {

struct Conv2DParam {};

class Conv2DFixture : public TTNNFixture, public testing::WithParamInterface<Conv2DParam> {};

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
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel,
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
                                sum +=
                                    input[ci][h + kh - padding_height][w + kw - padding_width] * kernel[co][ci][kh][kw];
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

std::vector<float> flatten_result(
    const std::vector<float>& result, uint32_t output_channels, uint32_t height, uint32_t width) {
    std::vector<float> output = std::vector<float>(output_channels * height * width);
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width; w++) {
            for (uint32_t co = 0; co < output_channels; co++) {
                output[co * height * width + h * width + w] = result[co + (h * width + w) * output_channels];
            }
        }
    }
    return output;
}

std::vector<float> random_vector(const std::array<uint32_t, 4> shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 5.0);
    std::vector<float> vec(Shape(shape).volume());
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

std::vector<std::vector<std::vector<float>>> reorder_input(
    const std::vector<float>& input, uint32_t input_channels, uint32_t image_height, uint32_t image_width) {
    auto output = std::vector<std::vector<std::vector<float>>>(input_channels);
    for (uint32_t ci = 0, i = 0; ci < input_channels; ci++) {
        output[ci] = std::vector<std::vector<float>>(image_height);
        for (uint32_t h = 0; h < image_height; h++) {
            output[ci][h] = std::vector<float>(image_width);
        }
    }
    uint32_t c, w, h;
    for (uint32_t i = 0; i < input.size(); i++) {
        c = i % (input_channels);
        w = (i / input_channels) % image_width;
        h = i / input_channels / image_width;

        output[c][h][w] = input[i];
    }
    return output;
}

std::vector<std::vector<std::vector<std::vector<float>>>> reorder_weigths(
    const std::vector<float>& weights,
    uint32_t output_channels,
    uint32_t input_channels,
    uint32_t kernel_height,
    uint32_t kernel_width) {
    std::vector<std::vector<std::vector<std::vector<float>>>> output =
        std::vector<std::vector<std::vector<std::vector<float>>>>(output_channels);
    for (uint32_t co = 0; co < output_channels; co++) {
        output[co] = std::vector<std::vector<std::vector<float>>>(input_channels);
        for (uint32_t ci = 0; ci < input_channels; ci++) {
            output[co][ci] = std::vector<std::vector<float>>(kernel_height);
            for (uint32_t kh = 0; kh < kernel_height; kh++) {
                output[co][ci][kh] = std::vector<float>(kernel_width);
            }
        }
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
        uint32_t co = i % output_channels;
        uint32_t ci = (i / output_channels) % input_channels;
        uint32_t kh = (i / output_channels) / input_channels / kernel_width;
        uint32_t kw = ((i / output_channels) / input_channels) % kernel_width;
        output[co][ci][kh][kw] = weights[i];
    }
    return output;
}

TEST_P(Conv2DFixture, Conv2DCalculateCorrectly) {
    auto param = GetParam();
    const auto device_id = 0;
    auto* device = CreateDevice(device_id, 1, 16384, 0, DispatchCoreConfig(DispatchCoreType::ETH));

    const uint32_t input_channels = 32;   // in_channels
    const uint32_t output_channels = 32;  // out_channels

    const uint32_t input_height = 256;                   // input_height
    const uint32_t input_width = 256;                    // input_width
    const std::array<uint32_t, 2> kernel_size = {3, 3};  // kernel_size
    const std::array<uint32_t, 2> stride = {1, 1};       // stride
    const std::array<uint32_t, 2> padding = {1, 1};      // padding

    MemoryConfig dram_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

    std::array<uint32_t, 4> dimensions = {1, input_channels, input_height, input_width};
    std::array<uint32_t, 4> dimensions_weight = {output_channels, input_channels, kernel_size[0], kernel_size[1]};

    std::array<uint32_t, 4> tt_dimensions = {
        1, input_height /* input_h */, input_width /* input_w*/, input_channels /* input_c */};
    std::array<uint32_t, 4> tt_dimensions_weight = {
        1, 1, input_channels * kernel_size[0] * kernel_size[1] /* Kh * Kw * input_c */, output_channels /* output_c */};

    auto input_vector = random_vector(dimensions);
    auto weight_vector = random_vector(dimensions_weight);

    auto input_tensor_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        DataType::BFLOAT16,
        PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        dram_mem_config,
        /* logical */ Shape(dimensions),
        /* padded */ Shape(dimensions));
    auto input_tensor =
        Tensor::from_vector(input_vector, TensorSpec(Shape(tt_dimensions), input_tensor_layout), device);

    auto weight_tensor_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        DataType::BFLOAT16,
        PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        dram_mem_config,
        /* logical */ Shape(dimensions_weight),
        /* padded */ Shape(dimensions_weight));
    auto weight_tensor =
        Tensor::from_vector(weight_vector, TensorSpec(Shape(tt_dimensions_weight), weight_tensor_layout))
            .to_layout(TILE_LAYOUT)
            .to_device(device, dram_mem_config);

    Result r = conv2d::conv2d(
        input_tensor,
        weight_tensor,
        device,
        input_channels,
        output_channels,
        1,             // batch_size
        input_height,  // input_height
        input_width,   // input_width
        kernel_size,   // kernel_size
        stride,        // stride
        padding,       // padding
        {1, 1},        // dilation
        1,             // groups
        std::nullopt,
        conv2d::Conv2dConfig({
            .dtype = DataType::BFLOAT16,
            .weights_dtype = DataType::BFLOAT16,
            .input_channels_alignment = 16,
            .output_layout = ROW_MAJOR_LAYOUT,
        }),
        std::nullopt);
    auto [output_tensor, output_height, output_width, r1, r2] = r;
    // std::cout<<"output_tensor logical shape
    // ("<<output_tensor.get_logical_shape()[0]<<","<<output_tensor.get_logical_shape()[1]<<","<<output_tensor.get_logical_shape()[2]<<","<<output_tensor.get_logical_shape()[3]<<")"<<std::endl;
    // std::cout<<"output_height: "<<output_height<<std::endl;
    // std::cout<<"output_width: "<<output_width<<std::endl;

    auto res = flatten_result(output_tensor.to_vector<float>(), output_channels, output_height, output_width);

    auto ref_res = conv2d(
        reorder_input(input_vector, input_channels, input_height, input_width),
        reorder_weigths(weight_vector, output_channels, input_channels, kernel_size[0], kernel_size[1]),
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
    TT_FATAL(pcc_calculated > 0.95, "Failed pcc");
}

INSTANTIATE_TEST_SUITE_P(Conv2DTests, Conv2DFixture, ::testing::Values(Conv2DParam{}));

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
