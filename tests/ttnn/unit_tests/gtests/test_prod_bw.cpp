// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "smoke_test_utils.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"

namespace ttnn::operations::reduction::test {

class ReductionSmoke : public TTNNFixtureWithSuiteDevice<ReductionSmoke> {};

namespace detail {
using ttnn::test_utils::make_device_tensor;
}  // namespace detail

TEST_F(ReductionSmoke, ProdBw_ZeroElement_SimpleCase) {
    auto& device = *device_;

    // Input: [2, 0, 4]
    const std::vector<float> in_data = {2.0f, 0.0f, 4.0f};
    const auto input = detail::make_device_tensor(device, ttnn::Shape{3}, in_data, DataType::FLOAT32, Layout::TILE);

    // Upstream grad (keepdim=True for dim=0): shape {1}
    const std::vector<float> grad_data = {1.0f};
    const auto grad = detail::make_device_tensor(device, ttnn::Shape{1}, grad_data, DataType::FLOAT32, Layout::TILE);

    // Call prod_bw reducing dim=0 (keepdim semantics expected by prod_bw when dim provided)
    auto outputs = ttnn::prod_bw(grad, input, std::optional<int64_t>(0));
    ASSERT_EQ(outputs.size(), 1u);

    const auto out_host = ttnn::test_utils::to_float_vector(outputs[0]);
    const std::vector<float> expected = {0.0f, 8.0f, 0.0f};

    // Use strict element-wise comparison for this small, exact case
    ttnn::test_utils::expect_close(out_host, expected, 1e-6f, 1e-6f, ttnn::test_utils::kSkipCheck, ttnn::test_utils::kSkipCheck);
}

}  // namespace ttnn::operations::reduction::test
