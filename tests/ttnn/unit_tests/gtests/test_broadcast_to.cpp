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
    uint32_t h;  // input tensor height
    uint32_t w;  // input tensor width
    std::vector<int32_t> broadcast_shape;
};

class Broadcast_toFixture : public TTNNFixture, public testing::WithParamInterface<BroadcastParam> {};

TEST_P(Broadcast_toFixture, Broadcast_to) {
    auto param = GetParam();
    const auto device_id = 0;
    auto& device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape input_shape(dimensions);
    std::vector<uint32_t> output_size = {param.broadcast_shape[0], param.broadcast_shape[1]};
    ttnn::Shape output_shape(output_size);

    std::vector<int32_t> broadcast_to = param.broadcast_shape;
    const std::optional<MemoryConfig> memory_config = std::nullopt;
    {
        const auto input_tensor = ttnn::ones(input_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
#if 1
        std::optional<ttnn::Tensor> output_tensor = std::nullopt;
#else
        const auto output_tensor = ttnn::zeros(output_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
#endif
        ttnn::experimental::broadcast_to(input_tensor, broadcast_to, output_tensor, memory_config);
        const auto expected_tensor =
            ttnn::operations::creation::full(output_shape, 1, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        TT_FATAL(output_tensor.has_value(), "Output error");
        TT_FATAL(
            ttnn::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor.value())),
            "Error");
    }
    ttnn::close_device(device);
}

INSTANTIATE_TEST_SUITE_P(Broadcast_toTests, Broadcast_toFixture, ::testing::Values(BroadcastParam{32, 32, {32, 32}}));

}  // namespace test
}  // namespace broadcast_to
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn
