// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>
#include <tt-metalium/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/small_vector.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/core/core.hpp"
#include "common_test_utils.hpp"

namespace ttnn {
namespace operations {
namespace conv::conv2d {
namespace test {

struct Conv2DParam {
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 2> padding;
};

class Conv2DFixture : public ::testing::Test, public testing::WithParamInterface<Conv2DParam> {};

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
    uint32_t i = 0;
    for (uint32_t n = 0; n < batch_size; n++) {
        for (uint32_t co = 0; co < output_channels; co++) {
            for (uint32_t h = 0; h < input_height; h += stride_height) {
                std::vector<float> row;
                for (uint32_t w = 0; w < input_width; w += stride_width) {
                    float sum = 0;
                    for (uint32_t ci = 0; ci < input_channels; ci++) {
                        for (uint32_t kh = 0; kh < kernel_height; kh++) {
                            for (uint32_t kw = 0; kw < kernel_width; kw++) {
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
    const Conv2DParam param = GetParam();

    // Sets the size for L1 small on the device - 16KB
    // The halo op which is contained in the Conv2D op uses L1 small memory
    // Without this, the convolution operation will fail due to L1_SMALL Out of Memory error
    const size_t l1_small_size = 16384;

    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0, l1_small_size);

    try {
        MemoryConfig dram_mem_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

        // (N,Ci,H,W)
        Shape dimensions{param.batch_size, param.input_channels, param.input_height, param.input_width};
        // (Co,Ci,KH,KW)
        Shape dimensions_weight{
            param.output_channels, param.input_channels, param.kernel_size[0], param.kernel_size[1]};

        random::seed(42);
        // Create input tensor on device
        Tensor input_tensor =
            ttnn::random::random(dimensions, tt::tt_metal::DataType::BFLOAT16).to_device(device.get(), dram_mem_config);

        // Create weight tensor on device (weight tensor on device would require to be tiled if
        // Conv2DConfig.always_preprocess_weights isn't used)
        Tensor weight_tensor = ttnn::random::random(dimensions_weight, tt::tt_metal::DataType::BFLOAT16);

        // Copy input tensor and weight tensor to host for reference implementation
        std::vector<float> input_vector = input_tensor.to_vector<float>();
        std::vector<float> weight_vector = weight_tensor.to_vector<float>();

        // (N,Ci,H,W) -> (N,H,W,Ci)
        input_tensor = ttnn::permute(input_tensor, SmallVector<int64_t>{0, 2, 3, 1});

        // Run Conv2D
        auto [output_tensor, output_dimensions] = std::get<static_cast<int>(ResultType::OUTPUT_DIM)>(ttnn::conv2d(
            DefaultQueueId,
            input_tensor,
            weight_tensor,
            device.get(),
            param.input_channels,
            param.output_channels,
            param.batch_size,
            param.input_height,
            param.input_width,
            param.kernel_size,
            param.stride,
            param.padding,
            std::array<uint32_t, 2>{1, 1},  // dilation
            1,                              // groups
            std::nullopt,                   // bias tensor
            std::nullopt,                   // conv config
            std::nullopt,                   // compute config
            std::nullopt,                   // memory config
            std::nullopt,                   // slice config
            true                            // return_output_dim
            ));

        // move output tensor to dram
        output_tensor = ttnn::to_memory_config(output_tensor, dram_mem_config);

        // H'  - output_height
        // W'  - output_width
        // (1,1,NH'W',Co) -> (N,H',W',Co)
        output_tensor = ttnn::reshape(
            output_tensor,
            Shape(
                {param.batch_size,
                 std::get<0>(output_dimensions),
                 std::get<1>(output_dimensions),
                 param.output_channels}));

        // (N,H',W',Co) -> (N,Co,H',W')
        output_tensor = ttnn::permute(output_tensor, SmallVector<int64_t>{0, 3, 1, 2});

        // Copy output tensor to host for comparison
        std::vector<float> res = output_tensor.to_vector<float>();

        // Run reference implementation of Conv2D
        std::vector<float> ref_res = reference_implementation_conv2d(
            input_vector,
            weight_vector,
            param.input_channels,
            param.output_channels,
            param.batch_size,
            param.input_height,
            param.input_width,
            param.kernel_size,
            param.stride,
            param.padding);

        EXPECT_GT(test_utils::pcc(res, ref_res), 0.99);
    } catch (const std::exception& e) {
        FAIL() << "Caught exception in Conv2D test: " << e.what();
        throw e;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Conv2DTests,
    Conv2DFixture,
    ::testing::Values(
        Conv2DParam{
            .input_channels = 3,
            .output_channels = 17,
            .batch_size = 5,
            .input_height = 111,
            .input_width = 25,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        },
        Conv2DParam{
            .input_channels = 32,
            .output_channels = 32,
            .batch_size = 2,
            .input_height = 256,
            .input_width = 256,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        },
        Conv2DParam{
            .input_channels = 3,
            .output_channels = 15,
            .batch_size = 7,
            .input_height = 3,
            .input_width = 3,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        }));

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
