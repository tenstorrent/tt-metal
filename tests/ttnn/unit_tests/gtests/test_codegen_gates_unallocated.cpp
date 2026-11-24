// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <optional>

#include "ttnn/device.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/gather/codegen/gather_codegen_supported.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"
#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_supported.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::test {

class CodegenGatesUnallocatedFixture : public TTNNFixtureWithSuiteDevice<CodegenGatesUnallocatedFixture> {};

// The codegen routing gates are queried before the op's own validation, so a deallocated device
// tensor must get an answer rather than an exception. pad and repeat_interleave decline it;
// untilize and gather accept it and leave the error to the op's validation.
TEST_F(CodegenGatesUnallocatedFixture, GatesAnswerForDeallocatedTensor) {
    auto& device = *device_;
    auto input = ttnn::zeros(ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto index = ttnn::zeros(ttnn::Shape({1, 1, 32, 64}), DataType::UINT32, ttnn::TILE_LAYOUT, device);
    input.deallocate();
    index.deallocate();
    ASSERT_FALSE(input.is_allocated());
    ASSERT_FALSE(index.is_allocated());

    EXPECT_FALSE(pad_codegen::in_codegen_scope(input, input.memory_config()));
    EXPECT_FALSE(repeat_interleave_codegen::supported_by_codegen(input, /*repeats=*/2, /*dim=*/-1, std::nullopt));
    EXPECT_TRUE(untilize_codegen::codegen_cb_plan_fits_live_l1(input, input.memory_config()));
    EXPECT_TRUE(gather::supported_by_codegen(input, /*dim=*/-1, index));
}

}  // namespace ttnn::operations::data_movement::test
