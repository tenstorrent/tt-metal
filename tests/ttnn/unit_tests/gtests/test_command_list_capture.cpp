// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <vector>

#include <tt-metalium/sub_device.hpp>

#include "command_list_capture_test_ops.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_utils.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::experimental::test {
namespace {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

class CommandListCaptureTest : public TTNNFixtureWithDevice {
protected:
    static constexpr uint32_t kCapacity = 12;
    static constexpr uint32_t kTilesPerRow = 3;
    static constexpr uint32_t kWidth = kTilesPerRow * TILE_WIDTH;
    static constexpr uint32_t kElementsPerTileRow = TILE_HEIGHT * kWidth;

    CommandListCaptureTest() : TTNNFixtureWithDevice(/*trace_region_size=*/8 << 20, DEFAULT_L1_SMALL_SIZE) {}

    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Command lists need fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
    }

    static tt::tt_metal::TensorSpec spec(uint32_t width = kWidth) {
        return tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, kCapacity * TILE_HEIGHT, width},
            tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    }

    static std::vector<uint32_t> iota_values(uint32_t width = kWidth) {
        std::vector<uint32_t> values(kCapacity * TILE_HEIGHT * width);
        for (uint32_t i = 0; i < values.size(); ++i) {
            values[i] = i + 1;
        }
        return values;
    }

    Tensor make_input(uint32_t width = kWidth) const {
        return Tensor::from_vector(iota_values(width), spec(width), device_);
    }

    void zero(Tensor& tensor) const {
        copy_to_device(Tensor::from_vector(std::vector<uint32_t>(kCapacity * kElementsPerTileRow, 0), spec()), tensor);
    }

    // The first `tile_rows` rows of tiles are the input's; the rest are still zero.
    static std::vector<uint32_t> expected_after(uint32_t tile_rows) {
        auto values = iota_values();
        std::fill(values.begin() + (tile_rows * kElementsPerTileRow), values.end(), 0);
        return values;
    }
};

TEST_F(CommandListCaptureTest, OneRecordingReplaysEveryLength) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    auto mid = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input}, kSharedTileRows);
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = mid}, kSharedTileRows);
    auto list = capture.build();
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);
    EXPECT_EQ(list.capacity("tile_rows"), kCapacity);

    // One patch updates both recorded ops; lengths below the core count leave some cores idle.
    for (uint32_t tile_rows : {1u, 5u, 2u, kCapacity, 0u}) {
        zero(mid);
        zero(out);
        list.set("tile_rows", tile_rows);
        list.replay(/*blocking=*/true);
        EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(tile_rows)) << tile_rows << " tile rows";
    }
}

// Eager launches and the recording share one cached program, and neither disturbs the other.
TEST_F(CommandListCaptureTest, SharesTheCachedProgramWithEagerLaunches) {
    const auto input = make_input();
    auto eager = device_operation::launch<CopyRowsOp>({.num_tile_rows = 2}, {.input = input});
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);

    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = 4}, {.input = input});
    auto list = capture.build();
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);

    device_->get_program_cache().set_cache_misses_allowed(false);
    eager = device_operation::launch<CopyRowsOp>({.num_tile_rows = 7}, {.input = input});
    zero(out);
    list.replay(/*blocking=*/true);
    device_->get_program_cache().set_cache_misses_allowed(true);

    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(4));
    const auto eager_values = eager.to_vector<uint32_t>();
    const auto input_values = iota_values();
    const auto copied = static_cast<std::ptrdiff_t>(7 * kElementsPerTileRow);
    EXPECT_TRUE(std::equal(eager_values.begin(), eager_values.begin() + copied, input_values.begin()));
}

TEST_F(CommandListCaptureTest, RejectsUnknownParametersAndValuesOverCapacity) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = 1}, {.input = input}, kSharedTileRows);
    auto list = capture.build();

    EXPECT_THAT(
        [&] { list.set("tile_rows", kCapacity + 1); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("exceeds its capacity")));
    EXPECT_THAT(
        [&] { list.set("rows", 1); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("no scalar parameter")));

    // The rejected patches wrote nothing: the recorded length still holds.
    zero(out);
    list.replay(/*blocking=*/true);
    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(1));
}

// A real Metal 2.0 op, unary MAC_TSS (x * scale + bias), feeds CopyRowsOp. One update patches the unary op's input and
// both scalars, the copy's length and its output.
TEST_F(CommandListCaptureTest, PatchesFiveParametersInOneUpdate) {
    using operations::unary::UnaryDeviceOperation;
    namespace unary = operations::unary;
    const tt::tt_metal::TensorSpec fp32_spec(
        ttnn::Shape{1, 1, kCapacity * TILE_HEIGHT, kWidth},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    const auto fp32_tensor = [&](uint32_t period, float offset) {
        std::vector<float> values(kCapacity * kElementsPerTileRow);
        for (uint32_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(i % period) + offset;
        }
        return Tensor::from_vector(values, fp32_spec, device_);
    };
    const auto zeros = [&] {
        return Tensor::from_vector(std::vector<float>(kCapacity * kElementsPerTileRow, 0.0f), fp32_spec, device_);
    };
    const auto mac = [](const Tensor& input, float scale, float bias) {
        const std::vector op_chain{
            unary::EltwiseUnaryWithParam{unary::UnaryOpType::MAC_TSS, std::vector<float>{scale, bias}}};
        return UnaryDeviceOperation::operation_attributes_t{
            .op_chain = op_chain,
            .output_dtype = DataType::FLOAT32,
            .memory_config = input.memory_config(),
            .fp32_dest_acc_en = true,
            .preserve_fp32_precision = true,
            .worker_grid = unary::get_worker_grid(op_chain, input, std::nullopt, input.memory_config(), std::nullopt),
        };
    };
    const auto scalar = [](float value) {
        return unary::utils::pack_scalar_runtime_arg_impl(value, DataType::FLOAT32);
    };
    // The eager result with only its first `tile_rows` rows of tiles kept, as the copy leaves a zeroed output.
    const auto eager_mac = [&](const Tensor& input, float scale, float bias, uint32_t tile_rows) {
        auto values = device_operation::launch<UnaryDeviceOperation>(mac(input, scale, bias), {.input = input})
                          .to_vector<float>();
        std::fill(values.begin() + (tile_rows * kElementsPerTileRow), values.end(), 0.0f);
        return values;
    };

    const auto first_input = fp32_tensor(/*period=*/1000, /*offset=*/0.0f);
    const auto second_input = fp32_tensor(/*period=*/777, /*offset=*/-300.0f);
    CommandListCapture capture(*device_);
    const auto scaled = capture.record<UnaryDeviceOperation>(
        mac(first_input, 2.0f, 1.0f),
        {.input = first_input},
        {{"input", "x"}, {"scalar1", "scale"}, {"scalar2", "bias"}});
    const auto first_out = capture.record<CopyRowsOp>(
        {.num_tile_rows = kCapacity}, {.input = scaled}, {{"tile_rows", "tile_rows"}, {"output", "out"}});
    auto list = capture.build();

    list.replay(/*blocking=*/true);
    const auto recorded = eager_mac(first_input, 2.0f, 1.0f, kCapacity);
    EXPECT_EQ(first_out.to_vector<float>(), recorded);

    auto second_out = zeros();
    list.update(CommandListPatch{}
                    .set("x", second_input)
                    .set("scale", scalar(3.0f))
                    .set("bias", scalar(-5.0f))
                    .set("tile_rows", 5)
                    .set("out", second_out));
    list.replay(/*blocking=*/true);
    EXPECT_EQ(second_out.to_vector<float>(), eager_mac(second_input, 3.0f, -5.0f, 5));
    EXPECT_EQ(first_out.to_vector<float>(), recorded);

    // Parameters left out of a patch keep their last values.
    copy_to_device(
        Tensor::from_vector(std::vector<float>(kCapacity * kElementsPerTileRow, 0.0f), fp32_spec), second_out);
    list.update(CommandListPatch{}.set("bias", scalar(7.0f)));
    list.replay(/*blocking=*/true);
    EXPECT_EQ(second_out.to_vector<float>(), eager_mac(second_input, 3.0f, 7.0f, 5));
}

TEST_F(CommandListCaptureTest, RejectsBindingsTheOpDoesNotDeclare) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    EXPECT_THAT(
        [&] { (void)capture.record<CopyRowsOp>({.num_tile_rows = 1}, {.input = input}, {{"length", "length"}}); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("no command list parameter 'length'")));
    EXPECT_THAT(
        [&] {
            (void)capture.record<CopyRowsOp>(
                {.num_tile_rows = 1}, {.input = input}, {{"tile_rows", "p"}, {"input", "p"}});
        },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("both a tensor and a scalar")));
}

// Metal validates the whole patch before writing any of it, so a bad tensor also drops the valid length beside it.
TEST_F(CommandListCaptureTest, RejectsATensorWithAnotherSpecAndWritesNothing) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>(
        {.num_tile_rows = 2}, {.input = input}, {{"tile_rows", "tile_rows"}, {"input", "in"}});
    auto list = capture.build();

    const auto narrow = make_input(/*width=*/TILE_WIDTH);
    EXPECT_THROW(list.update(CommandListPatch{}.set("tile_rows", 6).set("in", narrow)), std::exception);
    EXPECT_THAT(
        [&] { list.update(CommandListPatch{}.set("tile_rows", 6).set("input", input)); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("no tensor parameter 'input'")));

    zero(out);
    list.replay(/*blocking=*/true);
    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(2));
}

TEST_F(CommandListCaptureTest, RejectsLegacyPrograms) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    EXPECT_THAT(
        [&] { (void)capture.record<LegacyOp>({}, {.input = input}); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("only support Metal 2.0 programs")));
}

// The program cache may drop the recorded programs, and other programs may take over their cores, without breaking
// the list.
TEST_F(CommandListCaptureTest, ReplaysAfterTheProgramCacheDropsTheRecordedProgram) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = 3}, {.input = input}, kSharedTileRows);
    auto list = capture.build();

    device_->clear_program_cache();
    EXPECT_EQ(device_->get_program_cache().num_entries(), 0);
    // A narrower input changes the kernels' compile-time arguments, so this program has its own binaries.
    const auto other_input = make_input(/*width=*/TILE_WIDTH);
    auto other = device_operation::launch<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = other_input});
    EXPECT_EQ(other.to_vector<uint32_t>(), iota_values(TILE_WIDTH));

    zero(out);
    list.set("tile_rows", 5);
    list.replay(/*blocking=*/true);
    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(5));
}

// Under a manager with two sub-devices, ops on either one record into one list. Managers cannot change while a builder
// is alive, but a built list survives a swap and replays whenever its manager is loaded again.
TEST_F(CommandListCaptureTest, RecordsOpsOnTwoSubDevices) {
    const auto grid = device_->compute_with_storage_grid_size();
    ASSERT_GE(grid.y, 2 * CopyRowsOp::kCoreRows);
    const std::array first_cores{CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, CopyRowsOp::kCoreRows - 1}))};
    const std::array second_cores{CoreRangeSet(CoreRange({0, CopyRowsOp::kCoreRows}, {grid.x - 1, grid.y - 1}))};
    const std::array sub_devices{tt::tt_metal::SubDevice(first_cores), tt::tt_metal::SubDevice(second_cores)};
    const auto manager = device_->create_sub_device_manager(sub_devices, /*local_l1_size=*/0);
    device_->load_sub_device_manager(manager);
    {
        const auto input = make_input();
        Tensor first_out;
        Tensor second_out;
        std::optional<CapturedCommandList> list;
        {
            CommandListCapture capture(*device_);
            first_out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input}, kSharedTileRows);
            second_out = capture.record<CopyRowsOp>(
                {.num_tile_rows = kCapacity, .first_core_row = CopyRowsOp::kCoreRows},
                {.input = input},
                kSharedTileRows);
            EXPECT_THROW(device_->clear_loaded_sub_device_manager(), std::exception);
            list.emplace(capture.build());
        }
        const auto check = [&](uint32_t tile_rows) {
            zero(first_out);
            zero(second_out);
            list->set("tile_rows", tile_rows);
            list->replay(/*blocking=*/true);
            EXPECT_EQ(first_out.to_vector<uint32_t>(), expected_after(tile_rows)) << tile_rows << " tile rows";
            EXPECT_EQ(second_out.to_vector<uint32_t>(), expected_after(tile_rows)) << tile_rows << " tile rows";
        };
        check(3);
        check(kCapacity);

        device_->clear_loaded_sub_device_manager();
        EXPECT_THROW(list->replay(/*blocking=*/true), std::exception);
        device_->load_sub_device_manager(manager);
        check(7);
    }
    device_->clear_loaded_sub_device_manager();
    device_->remove_sub_device_manager(manager);
}

}  // namespace
}  // namespace ttnn::experimental::test
