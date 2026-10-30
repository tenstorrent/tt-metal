// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>

#include "command_list_capture_test_ops.hpp"
#include "ttnn/operations/trace.hpp"
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

// New allocations would take the freed input and intermediate if the list did not hold them.
TEST_F(CommandListCaptureTest, HoldsRecordedTensorsAfterTheCallerDropsThem) {
    std::optional<CapturedCommandList> list;
    Tensor out;
    {
        const auto input = make_input();
        CommandListCapture capture(*device_);
        const auto mid = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input}, kSharedTileRows);
        out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = mid}, kSharedTileRows);
        list.emplace(capture.build());
    }
    const std::vector<uint32_t> junk_values(kCapacity * kElementsPerTileRow, 0xDEADBEEF);
    std::vector<Tensor> junk;
    for (int i = 0; i < 4; ++i) {
        junk.push_back(Tensor::from_vector(junk_values, spec(), device_));
    }

    zero(out);
    list->set("tile_rows", 5);
    list->replay(/*blocking=*/true, /*check_dram_overlap=*/true);
    EXPECT_EQ(out.to_vector<uint32_t>(), expected_after(5));
}

TEST_F(CommandListCaptureTest, HoldsThePatchedTensorUntilTheNextPatch) {
    const auto input = make_input();
    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>(
        {.num_tile_rows = kCapacity}, {.input = input}, {{"tile_rows", "tile_rows"}, {"input", "in"}});
    auto list = capture.build();

    auto tripled = iota_values();
    for (auto& value : tripled) {
        value *= 3;
    }
    list.update(CommandListPatch{}.set("in", Tensor::from_vector(tripled, spec(), device_)).set("tile_rows", 4));
    const std::vector<uint32_t> junk_values(kCapacity * kElementsPerTileRow, 0xDEADBEEF);
    std::vector<Tensor> junk;
    for (int i = 0; i < 4; ++i) {
        junk.push_back(Tensor::from_vector(junk_values, spec(), device_));
    }

    zero(out);
    list.replay(/*blocking=*/true, /*check_dram_overlap=*/true);
    std::fill(tripled.begin() + (4 * kElementsPerTileRow), tripled.end(), 0);
    EXPECT_EQ(out.to_vector<uint32_t>(), tripled);
}

// Holding a tensor cannot stop a forced deallocate. The overlap check then catches an allocation over its memory.
TEST_F(CommandListCaptureTest, OverlapCheckCatchesMemoryFreedUnderTheList) {
    auto input = make_input();
    CommandListCapture capture(*device_);
    auto out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input}, kSharedTileRows);
    auto list = capture.build();
    list.replay(/*blocking=*/true, /*check_dram_overlap=*/true);

    input.deallocate(/*force=*/true);
    // An allocation exactly filling a recorded range passes as a rebound tensor; half of one is an overlap.
    const tt::tt_metal::TensorSpec half_spec(
        ttnn::Shape{1, 1, kCapacity / 2 * TILE_HEIGHT, kWidth},
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    const auto squatter = create_device_tensor(half_spec, device_);
    EXPECT_THAT(
        [&] { list.replay(/*blocking=*/true, /*check_dram_overlap=*/true); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("overlap")));
}

// Not a correctness test: prints the host cost of enqueueing, and the time to finish, one iteration of kOps tiny ops.
// Run with --gtest_also_run_disabled_tests.
TEST_F(CommandListCaptureTest, DISABLED_TimesReplayAgainstTraceAndEagerLaunches) {
    constexpr int kOps = 16;
    constexpr int kIterations = 1000;
    const auto input = make_input();
    auto& cq = device_->mesh_command_queue();
    const auto run_ops = [&] {
        Tensor current = input;
        for (int i = 0; i < kOps; ++i) {
            current = device_operation::launch<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = current});
        }
        return current;
    };
    struct Timing {
        double enqueue_us;
        double total_us;
    };
    const auto time = [&](const std::function<void()>& iteration) {
        iteration();
        tt::tt_metal::distributed::Finish(cq);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kIterations; ++i) {
            iteration();
        }
        const auto enqueued = std::chrono::steady_clock::now();
        tt::tt_metal::distributed::Finish(cq);
        const auto finished = std::chrono::steady_clock::now();
        const auto per_iteration = [](auto duration) {
            return std::chrono::duration<double, std::micro>(duration).count() / kIterations;
        };
        return Timing{per_iteration(enqueued - start), per_iteration(finished - start)};
    };

    const auto eager = time([&] { (void)run_ops(); });

    Timing trace{};
    Timing trace_blocking{};
    {
        const auto trace_id = ttnn::operations::trace::begin_trace_capture(device_, ttnn::QueueId(0));
        const auto traced_out = run_ops();
        ttnn::operations::trace::end_trace_capture(device_, trace_id, ttnn::QueueId(0));
        trace = time([&] { ttnn::operations::trace::execute_trace(device_, trace_id, ttnn::QueueId(0), false); });
        trace_blocking =
            time([&] { ttnn::operations::trace::execute_trace(device_, trace_id, ttnn::QueueId(0), true); });
        ttnn::operations::trace::release_trace(device_, trace_id);
    }

    std::optional<CapturedCommandList> list;
    {
        CommandListCapture capture(*device_);
        Tensor current = input;
        for (int i = 0; i < kOps; ++i) {
            const CommandListBindings bindings =
                i == 0 ? CommandListBindings{{"tile_rows", "tile_rows"}, {"input", "in"}} : kSharedTileRows;
            current = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = current}, bindings);
        }
        list.emplace(capture.build());
    }
    const auto other_input = make_input();
    const auto replay = time([&] { list->replay(); });
    uint32_t tile_rows = 0;
    const auto scalar_patch = time([&] {
        list->set("tile_rows", kCapacity - (tile_rows++ % 2));
        list->replay();
    });
    bool flip = false;
    const auto two_patches = time([&] {
        flip = !flip;
        list->update(CommandListPatch{}.set("in", flip ? other_input : input).set("tile_rows", kCapacity));
        list->replay();
    });
    const auto patch_only = time([&] { list->set("tile_rows", kCapacity - (tile_rows++ % 2)); });
    const auto tensor_patch_only = time([&] {
        flip = !flip;
        list->update(CommandListPatch{}.set("in", flip ? other_input : input));
    });
    // With the queue drained before each call, the call's duration is host work only.
    const auto host_cost = [&](const std::function<void()>& call) {
        constexpr int kCalls = 200;
        double total_us = 0;
        for (int i = 0; i < kCalls; ++i) {
            tt::tt_metal::distributed::Finish(cq);
            const auto start = std::chrono::steady_clock::now();
            call();
            total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        }
        tt::tt_metal::distributed::Finish(cq);
        return total_us / kCalls;
    };
    const double scalar_patch_host_us = host_cost([&] { list->set("tile_rows", kCapacity - (tile_rows++ % 2)); });
    const double tensor_patch_host_us = host_cost([&] {
        flip = !flip;
        list->update(CommandListPatch{}.set("in", flip ? other_input : input));
    });
    const double replay_host_us = host_cost([&] { list->replay(); });
    const auto replay_blocking = time([&] { list->replay(/*blocking=*/true); });
    const auto patch_replay_blocking = time([&] {
        list->set("tile_rows", kCapacity - (tile_rows++ % 2));
        list->replay(/*blocking=*/true);
    });

    const auto report = [](const char* mode, const Timing& timing) {
        std::printf(
            "%-36s enqueue %8.2f us/iteration, finished %8.2f us/iteration\n",
            mode,
            timing.enqueue_us,
            timing.total_us);
    };
    std::printf("%d CopyRowsOp launches per iteration, %d iterations\n", kOps, kIterations);
    report("eager launches", eager);
    report("trace replay", trace);
    report("command list replay", replay);
    report("command list 1 scalar patch + replay", scalar_patch);
    report("command list tensor+scalar + replay", two_patches);
    report("command list 1 scalar patch only", patch_only);
    report("command list first-op tensor patch only", tensor_patch_only);
    report("trace replay, blocking", trace_blocking);
    report("command list replay, blocking", replay_blocking);
    report("command list patch + replay, blocking", patch_replay_blocking);
    std::printf(
        "host cost per call: 32-word scalar patch %.2f us, first-op tensor patch %.2f us, replay %.2f us\n",
        scalar_patch_host_us,
        tensor_patch_host_us,
        replay_host_us);
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
