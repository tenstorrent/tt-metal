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
    std::vector<uint32_t> broadcast_shape;
};

class Broadcast_toFixture : public TTNNFixtureWithDevice, public testing::WithParamInterface<BroadcastParam> {};

TEST_P(Broadcast_toFixture, Broadcast_to) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 4> dimensions = {param.n, param.c, param.h, param.w};
    ttnn::Shape input_shape(dimensions);
    ttnn::Shape output_shape(param.broadcast_shape);

    const auto memory_config = std::nullopt;
    const auto output = std::nullopt;

    const auto input_tensor = ttnn::ones(input_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto output_tensor = ttnn::experimental::broadcast_to(input_tensor, output_shape, memory_config, output);
    const auto expected_tensor =
        ttnn::operations::creation::full(output_shape, 1, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    TT_FATAL(ttnn::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)), "Error");

    ttnn::close_device(device);
}

// Spatial dimension broadcasts (H and W)
INSTANTIATE_TEST_SUITE_P(
    SpatialDimensions,
    Broadcast_toFixture,
    ::testing::Values(
        BroadcastParam{1, 1, 64, 1, {1, 1, 64, 64}},  // width only
        BroadcastParam{1, 1, 1, 64, {1, 1, 64, 64}},  // height only
        BroadcastParam{1, 1, 1, 1, {1, 1, 64, 64}}    // both height and width (scalar to 2D)
        ));

// Channel and batch dimension broadcasts
INSTANTIATE_TEST_SUITE_P(
    ChannelAndBatch,
    Broadcast_toFixture,
    ::testing::Values(
        BroadcastParam{1, 1, 64, 64, {1, 3, 64, 64}},  // channel expansion
        BroadcastParam{1, 1, 64, 64, {3, 1, 64, 64}}   // batch expansion
        ));

// Large tensor broadcasts
INSTANTIATE_TEST_SUITE_P(
    LargeTensor,
    Broadcast_toFixture,
    ::testing::Values(
        BroadcastParam{1, 1, 1, 1, {1, 1, 1, 32 * 32 * 64}},   // large width
        BroadcastParam{1, 1, 1, 1, {1, 1, 32 * 32 * 64, 1}},   // large height
        BroadcastParam{1, 1, 1, 1, {1, 1, 32 * 16, 32 * 16}},  // large 2D
        BroadcastParam{1, 1, 64, 64, {1, 32, 64, 64}},         // large channel
        BroadcastParam{1, 1, 64, 64, {32, 1, 64, 64}}          // large batch
        ));
}  // namespace test
}  // namespace broadcast_to
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
