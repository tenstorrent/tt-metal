// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct Add1DTensorAndScalarParam {
    float scalar;
    uint32_t h;
    uint32_t w;
};

class Add1DTensorAndScalarFixture : public TTNNFixture,
                                    public testing::WithParamInterface<Add1DTensorAndScalarParam> {};

TEST_P(Add1DTensorAndScalarFixture, AddsScalarCorrectly) {
    auto param = GetParam();
    const auto device_id = 0;
    auto& device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);

    {
        const auto input_tensor = ttnn::zeros(shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = input_tensor + param.scalar;
        const auto expected_tensor =
            ttnn::operations::creation::full(shape, param.scalar, ttnn::bfloat16, ttnn::TILE_LAYOUT, device);
        TT_FATAL(tt::numpy::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)));
    }
    ttnn::close_device(device);
}

INSTANTIATE_TEST_SUITE_P(
    Add1DTensorAndScalarTests, Add1DTensorAndScalarFixture, ::testing::Values(Add1DTensorAndScalarParam{3.0f, 32, 64}));

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
