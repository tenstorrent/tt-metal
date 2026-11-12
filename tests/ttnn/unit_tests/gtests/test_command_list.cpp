// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Command lists need no warm-up run: each test opens a fresh device and first calls its op while recording.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "ttnn/command_list.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::experimental::test {

class CommandListTest : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Command lists require fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
    }

    Tensor make_input(const std::vector<float>& values) const {
        const TensorSpec spec(
            ttnn::Shape{1, 1, 32, 64}, TensorLayout(DataType::BFLOAT16, Layout::TILE, MemoryConfig{}));
        return Tensor::from_vector(values, spec, device_);
    }
};

TEST_F(CommandListTest, RecordBuildReplay) {
    std::vector<float> values(32 * 64);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % 17) - 8);
    }
    const Tensor input = make_input(values);

    CommandListBuilder builder(*device_);
    Tensor output;
    {
        auto recording = builder.record_scope();
        EXPECT_TRUE(builder.is_recording());
        output = ttnn::relu(input);
    }
    EXPECT_FALSE(builder.is_recording());

    CommandList command_list = builder.build(QueueId{0});
    builder.deallocate();
    EXPECT_EQ(*command_list.cq_id(), 0);

    command_list.enqueue(/*blocking=*/true);

    const std::vector<float> result = output.to_vector<float>();
    ASSERT_EQ(result.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(result[i], std::max(values[i], 0.0f)) << "at index " << i;
    }
    command_list.deallocate();
}

TEST_F(CommandListTest, LegacyOpIsRejected) {
    const Tensor input = make_input(std::vector<float>(32 * 64, 1.0f));
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    try {
        (void)ttnn::add(input, input);
        FAIL() << "Recording a legacy-factory op should throw";
    } catch (const std::exception& e) {
        EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("cannot be recorded into a command list"));
    }
    EXPECT_TRUE(builder.is_recording());
}

TEST_F(CommandListTest, BuilderLifetimeErrors) {
    CommandListBuilder builder(*device_);
    EXPECT_ANY_THROW(builder.build());
    EXPECT_ANY_THROW(CommandListBuilder second_builder(*device_));
    {
        auto recording = builder.record_scope();
        EXPECT_ANY_THROW(builder.begin_recording());
        EXPECT_ANY_THROW(builder.clear());
        EXPECT_ANY_THROW(builder.deallocate());
    }
    builder.deallocate();
    EXPECT_THAT(
        [&] { (void)builder.build(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("CommandListBuilder has been deallocated")));
}

}  // namespace ttnn::experimental::test
