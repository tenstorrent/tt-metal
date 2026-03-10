// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/bcast_to/bcast_to.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::experimental::broadcast_to::test {
struct BroadcastParam {
    uint32_t n = 0;  // input tensor batch
    uint32_t c = 0;  // input tensor channel
    uint32_t h = 0;  // input tensor height
    uint32_t w = 0;  // input tensor width
    std::vector<uint32_t> broadcast_shape;
};

class Broadcast_toFixture : public TTNNFixtureWithSuiteDevice<Broadcast_toFixture>,
                            public testing::WithParamInterface<BroadcastParam> {};

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
    const auto expected_tensor = ttnn::full(output_shape, 1, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    TT_FATAL(ttnn::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)), "Error");
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

// For unknown reason, this test hang for N300, but N150 is OK. Also the same shape tests are fine if doing it in pytest
// Combined dimension broadcasts (N, C, H, W simultaneously)
INSTANTIATE_TEST_SUITE_P(
    CombinedDimensions,
    Broadcast_toFixture,
    ::testing::Values(
        BroadcastParam{1, 1, 1, 1, {8, 17, 32, 64}},    // scalar to 4D tensor
        BroadcastParam{1, 3, 1, 1, {8, 3, 32, 64}},     // broadcast N, H, W (preserve C)
        BroadcastParam{2, 1, 4, 1, {2, 16, 4, 64}},     // broadcast C and W
        BroadcastParam{1, 1, 32, 32, {7, 17, 32, 32}},  // broadcast N and C (preserve H, W)
        BroadcastParam{1, 3, 1, 4, {8, 3, 32, 4}}       // broadcast N and H (preserve C, W)
        ));

// Non tile aligned dimension broadcasts
INSTANTIATE_TEST_SUITE_P(
    NonAlignedDimensions,
    Broadcast_toFixture,
    ::testing::Values(
        BroadcastParam{1, 1, 17, 1, {1, 1, 17, 51}},   // odd height, non-aligned width
        BroadcastParam{1, 1, 1, 23, {1, 1, 47, 23}},   // non-aligned height, odd width
        BroadcastParam{1, 1, 7, 13, {1, 1, 7, 13}},    // small prime dimensions (no broadcast, just validation)
        BroadcastParam{1, 1, 1, 1, {1, 1, 19, 31}},    // scalar to non-aligned 2D
        BroadcastParam{1, 1, 30, 30, {1, 5, 30, 30}},  // almost-aligned dimensions
        BroadcastParam{1, 3, 15, 29, {4, 3, 15, 29}},  // preserve odd dimensions, broadcast batch
        BroadcastParam{2, 1, 33, 65, {2, 7, 33, 65}}   // odd-plus-32 dimensions
        ));

}  // namespace ttnn::operations::experimental::broadcast_to::test
