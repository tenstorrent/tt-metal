// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include "tests/tt_metal/tt_metal/common/dispatch_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/bcast_to/bcast_to.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace broadcast_to {
namespace test {

struct BroadcastParam {
    uint32_t n;  // input tensor batch
    uint32_t c;  // input tensor channel
    uint32_t h;  // input tensor height
    uint32_t w;  // input tensor width
    std::vector<int32_t> broadcast_shape;
};

class Broadcast_toFixture : public TTNNFixture, public testing::WithParamInterface<BroadcastParam> {};

TEST_P(Broadcast_toFixture, Broadcast_to) {
    auto param = GetParam();
    const auto device_id = 0;
    auto& device = ttnn::open_device(device_id);
    std::array<uint32_t, 4> dimensions = {param.n, param.c, param.h, param.w};
    ttnn::Shape input_shape(dimensions);
    std::vector<uint32_t> output_size = {
        param.broadcast_shape[0],
        param.broadcast_shape[1],
        param.broadcast_shape[2],
        param.broadcast_shape[3],
    };
    ttnn::Shape output_shape(output_size);

    const std::optional<MemoryConfig> memory_config = std::nullopt;
    std::optional<ttnn::Tensor> output = std::nullopt;
    {
        const auto input_tensor = ttnn::ones(input_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        auto output_tensor =
            ttnn::experimental::broadcast_to(input_tensor, param.broadcast_shape, output, memory_config);
        const auto expected_tensor =
            ttnn::operations::creation::full(output_shape, 1, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        TT_FATAL(
            ttnn::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)), "Error");
    }
    ttnn::close_device(device);
}

// no change - single tile
INSTANTIATE_TEST_SUITE_P(
    no_change_single_tile, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 32, 32, {1, 1, 32, 32}}));

// no change - multiple tiles - multiple cores
INSTANTIATE_TEST_SUITE_P(
    no_change_multi_tile, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 64, {1, 1, 64, 64}}));

// single dimension width
INSTANTIATE_TEST_SUITE_P(dim_w, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 1, {1, 1, 64, 64}}));

// single dimension height
INSTANTIATE_TEST_SUITE_P(dim_h, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 1, 64, {1, 1, 64, 64}}));

// both dimension change - scalar
INSTANTIATE_TEST_SUITE_P(dim_w_h, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 1, 1, {1, 1, 64, 64}}));

// higher dimension change - c
INSTANTIATE_TEST_SUITE_P(dim_c, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 64, {1, 3, 64, 64}}));

// higher dimension change - n
INSTANTIATE_TEST_SUITE_P(dim_n, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 64, {3, 1, 64, 64}}));

// size 32x32x64 will have each core read 32x32x1
// no change large tensor
INSTANTIATE_TEST_SUITE_P(
    no_change_large,
    Broadcast_toFixture,
    ::testing::Values(BroadcastParam{1, 1, 32 * 16, 32 * 16, {1, 1, 32 * 16, 32 * 16}}));

// large tensor - w
INSTANTIATE_TEST_SUITE_P(
    large_w, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 1, 1, {1, 1, 1, 32 * 32 * 64}}));

// large tensor - h
INSTANTIATE_TEST_SUITE_P(
    large_h, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 1, 1, {1, 1, 32 * 32 * 64, 1}}));

// large tensor - scalar
INSTANTIATE_TEST_SUITE_P(
    large_w_h, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 1, 1, {1, 1, 32 * 16, 32 * 16}}));

// large tensor in c
INSTANTIATE_TEST_SUITE_P(
    large_c, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 64, {1, 32, 64, 64}}));

// large tensor in n
INSTANTIATE_TEST_SUITE_P(
    large_n, Broadcast_toFixture, ::testing::Values(BroadcastParam{1, 1, 64, 64, {32, 1, 64, 64}}));

}  // namespace test
}  // namespace broadcast_to
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
