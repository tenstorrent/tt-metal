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
    auto mid = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input});
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = mid});
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
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = 1}, {.input = input});
    auto list = capture.build();

    EXPECT_THAT(
        [&] { list.set("tile_rows", kCapacity + 1); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("exceeds its capacity")));
    EXPECT_THAT(
        [&] { list.set("rows", 1); }, ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("no parameter")));

    // The rejected patches wrote nothing: the recorded length still holds.
    zero(out);
    list.replay(/*blocking=*/true);
    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(1));
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
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = 3}, {.input = input});
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
            first_out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input});
            second_out = capture.record<CopyRowsOp>(
                {.num_tile_rows = kCapacity, .first_core_row = CopyRowsOp::kCoreRows}, {.input = input});
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
