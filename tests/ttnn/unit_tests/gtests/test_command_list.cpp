// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Command lists need no warm-up run: each test opens a fresh device and first calls its op while recording.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ttnn/command_list.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::command_list::test {

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

CmdListScalar real(double value) { return {value}; }
CmdListScalar integer(int64_t value) { return {value}; }

TEST(CommandListParamsTest, EncodersProduceArgumentWords) {
    EXPECT_EQ(encode_uint32(integer(7)).to_uint32(), 7u);
    EXPECT_EQ(encode_uint32(real(3.0)).to_uint32(), 3u);
    EXPECT_EQ(encode_uint32(integer(0xFFFFFFFF)).to_uint32(), 0xFFFFFFFFu);
    EXPECT_EQ(encode_int32_bits(integer(-1)).to_uint32(), 0xFFFFFFFFu);
    EXPECT_EQ(encode_float32_bits(real(1.0)).to_uint32(), 0x3F800000u);
    EXPECT_EQ(encode_float32_bits(integer(2)).to_uint32(), 0x40000000u);
    EXPECT_EQ(encode_float32_bits(real(-std::numeric_limits<double>::infinity())).to_uint32(), 0xFF800000u);
    EXPECT_EQ(encode_bfloat16_pair(real(1.0)).to_uint32(), 0x3F803F80u);
    EXPECT_EQ(encode_bfloat16_low(real(1.0)).to_uint32(), 0x00003F80u);
    EXPECT_EQ(encode_bfloat16_high(real(1.0)).to_uint32(), 0x3F800000u);
    EXPECT_EQ(encode_uint16_pair(integer(0x1234)).to_uint32(), 0x12341234u);
    EXPECT_EQ(encode_uint16_low(integer(0x1234)).to_uint32(), 0x00001234u);
    EXPECT_EQ(encode_uint16_high(integer(0x1234)).to_uint32(), 0x12340000u);
    EXPECT_EQ(encode_float8_e4m3_quad(real(1.0)).to_uint32(), 0x38383838u);
}

TEST(CommandListParamsTest, EncodersRejectValuesTheyCannotRepresent) {
    const auto rejects = [](const CmdListEncodeFn& encode, const CmdListScalar& scalar) {
        EXPECT_THAT(
            [&] { (void)encode(scalar); }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("needs")));
    };
    rejects(encode_uint32, integer(-1));
    rejects(encode_uint32, integer(int64_t{1} << 32));
    rejects(encode_uint32, real(1.5));
    rejects(encode_uint32, real(std::nan("")));
    rejects(encode_int32_bits, integer(int64_t{1} << 31));
    rejects(encode_uint16_low, integer(0x10000));
    rejects(encode_float32_bits, real(1e39));
    rejects(encode_bfloat16_pair, real(-1e39));
}

TEST_F(CommandListTest, BindsNeedARecordingBuilder) {
    EXPECT_THAT(
        [] { push_cmd_list_binds("op", {{.name = "a", .invoke_name = "x", .value = integer(1)}}); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("op: CmdListParams need a TTNN CommandListBuilder recording on this thread")));
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));

    CommandListBuilder builder(*device_);
    builder.begin_recording();
    push_cmd_list_binds("op", {{.name = "a", .invoke_name = "x", .value = integer(1)}});
    EXPECT_THAT(
        [&] { builder.end_recording(); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot end recording while op still has CmdListParam binds pushed")));
    pop_cmd_list_binds(false);
    builder.end_recording();
}

TEST_F(CommandListTest, FinishReportsBindsThatReachedNoProgram) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    CmdListBindScope scope("ttnn.relu", {{.name = "slope", .invoke_name = "slope", .value = real(0.5)}});
    EXPECT_THAT(
        [&] { scope.finish(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("ttnn.relu"),
            ::testing::HasSubstr("slope"),
            ::testing::HasSubstr("did not reach any recorded program field"))));
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

TEST_F(CommandListTest, BindScopesNest) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    {
        CmdListBindScope outer("outer_op", {{.name = "a", .invoke_name = "x", .value = integer(1)}});
        {
            CmdListBindScope inner("inner_op", {{.name = "b", .invoke_name = "y", .value = integer(2)}});
            EXPECT_THAT(
                [&] { inner.finish(); },
                ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
                    ::testing::HasSubstr("inner_op"), ::testing::HasSubstr("CmdListParam(s) 'b' did not reach"))));
        }
        CmdListBindScope empty("empty_op", {});
        EXPECT_NO_THROW(empty.finish());
    }
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

TEST_F(CommandListTest, PushRejectsInvalidBinds) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    EXPECT_ANY_THROW(push_cmd_list_binds("op", {{.name = "", .invoke_name = "x", .value = integer(1)}}));
    EXPECT_ANY_THROW(push_cmd_list_binds("op", {{.name = "a", .invoke_name = "", .value = integer(1)}}));
    const Tensor host_tensor = Tensor::from_vector(
        std::vector<float>(32 * 32, 1.0f),
        TensorSpec(ttnn::Shape{32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})));
    EXPECT_THAT(
        [&] { push_cmd_list_binds("op", {{.name = "t", .invoke_name = "input_tensor", .value = host_tensor}}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be an allocated device tensor")));
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

}  // namespace ttnn::command_list::test
