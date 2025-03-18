// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/core/core.hpp"

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

/*
    Reference implementation of Conv2D

    Takes in input tensor with original shape (N,Ci,H,W) that is flattened in row major order

    and flattened kernel tensor with original shape (Co,Ci,KH,KW) that is also flattened in row major order.

    Returns flattened tensor with original shape (N,Co,Xh,Xw) in row major order, where Xh and Xw are calculated based
    on input tensor,kernel tensor, stride and padding.


    The output vector is flattened in row major order.

    input_channels - Ci
    output_channels - Co
    input_height - H
    input_width - W
    batch_size - N
    output_height - Xh
    output_width - Xw
    kernel_size - (KH,KW)
    stride - (SH,SW)
    padding - (PH,PW)
*/
std::vector<float> reference_implementation_conv2d(
    const std::vector<float>& input,   // (N,Ci,H,W)
    const std::vector<float>& kernel,  // (Co,Ci,H',W')
    const uint32_t input_channels,
    const uint32_t output_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding) {
    uint32_t kernel_height = kernel_size[0];
    uint32_t kernel_width = kernel_size[1];
    uint32_t padding_height = padding[0];
    uint32_t padding_width = padding[1];
    uint32_t stride_height = stride[0];
    uint32_t stride_width = stride[1];

    // Calculate output height and width
    uint32_t Xh = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    uint32_t Xw = (input_width - kernel_width + 2 * padding_width) / stride_width + 1;

    std::vector<float> output = std::vector<float>(batch_size * output_channels * Xh * Xw);
    int i = 0;
    for (int n = 0; n < batch_size; n++) {
        for (int co = 0; co < output_channels; co++) {
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
                                               [n * input_channels * input_height * input_width +
                                                ci * input_height * input_width +
                                                (h + kh - padding_height) * input_width + w + kw - padding_width] *
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
    }
    return output;
}

TEST_P(Conv2DFixture, Conv2DCalculateCorrectly) {
    const chip_id_t device_id = 0;
    IDevice* device = CreateDevice(device_id, 1, 16384);

    const uint32_t input_channels = 3;                   // in_channels
    const uint32_t output_channels = 17;                 // out_channels
    const uint32_t batch_size = 5;                       // batch_size
    const uint32_t input_height = 111;                   // input_height
    const uint32_t input_width = 25;                     // input_width
    const std::array<uint32_t, 2> kernel_size = {3, 3};  // kernel_size
    const std::array<uint32_t, 2> stride = {1, 1};       // stride
    const std::array<uint32_t, 2> padding = {1, 1};      // padding

    MemoryConfig dram_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

    // (N,Ci,H,W)
    std::array<uint32_t, 4> dimensions = {batch_size, input_channels, input_height, input_width};
    // (Co,Ci,KH,KW)
    std::array<uint32_t, 4> dimensions_weight = {output_channels, input_channels, kernel_size[0], kernel_size[1]};

    random::seed(42);
    // Create input tensor on device
    Tensor input_tensor =
        ttnn::random::random(Shape(dimensions), tt::tt_metal::DataType::BFLOAT16).to_device(device, dram_mem_config);

    // Create weight tensor on device (weight tensor on device would require to be tield if
    // Conv2DConfig.always_preprocess_weights isn't used)
    Tensor weight_tensor = ttnn::random::random(Shape(dimensions_weight), tt::tt_metal::DataType::BFLOAT16);

    // Copy input tensor and weight tensor to host for reference implementation
    std::vector<float> input_vector = input_tensor.to_vector<float>();
    std::vector<float> weight_vector = weight_tensor.to_vector<float>();

    // (N,Ci,H,W) -> (N,H,W,Ci)
    input_tensor = ttnn::permute(input_tensor, SmallVector<int64_t>{0, 2, 3, 1});

    // Run Conv2D
    auto [output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device] = conv2d::conv2d(
        input_tensor,
        weight_tensor,
        device,
        input_channels,
        output_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        {1, 1},  // dilation
        1        // groups
    );

    // move output tensor to dram
    output_tensor = ttnn::to_memory_config(output_tensor, dram_mem_config);

    // untilize output tensor because the default output tensor layout is TILE layout
    output_tensor = ttnn::untilize(output_tensor);

    // unpad output vector
    output_tensor = ttnn::slice(
        output_tensor,
        std::array<uint32_t, 4>({0, 0, 0, 0}),
        std::array<uint32_t, 4>({1, 1, batch_size * output_height * output_width, output_channels}),
        std::array<uint32_t, 4>({1, 1, 1, 1}),
        dram_mem_config);

    // H'  - output_height
    // W'  - output_width
    // (1,1,NH'W',Co) -> (N,H',W',Co)
    output_tensor = ttnn::reshape(output_tensor, Shape({batch_size, output_height, output_width, output_channels}));

    // (N,H',W',Co) -> (N,Co,H',W')
    output_tensor = ttnn::permute(output_tensor, SmallVector<int64_t>{0, 3, 1, 2});

    // Copy output tensor to host for comparison
    std::vector<float> res = output_tensor.to_vector<float>();

    // Run reference implementation of Conv2D
    std::vector<float> ref_res = reference_implementation_conv2d(
        input_vector,
        weight_vector,
        input_channels,
        output_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding);

    float pcc_calculated = pcc(res, ref_res);

    bool pass = CloseDevice(device);
    std::cout << "PCC: " << pcc_calculated << std::endl;
    TT_FATAL(pass, "Error");
    TT_FATAL(pcc_calculated > 0.99, "Failed pcc");
}

INSTANTIATE_TEST_SUITE_P(Conv2DTests, Conv2DFixture, ::testing::Values(Conv2DParam{}));

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
