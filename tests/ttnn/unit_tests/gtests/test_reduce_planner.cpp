// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side sanity tests for the reduce planner (ttnn::kernel_lib::host::make_reduce_plan).
// Pure host code -- no device required.

#include <gtest/gtest.h>

#include <cstdint>

#include <tt-metalium/shape.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/host/reduce_host.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::reduction::test {

TEST(ReduceHostPlanner, PartialMaxUsesExistingScalerRecipe) {
    using namespace tt::tt_metal;
    using namespace ttnn::kernel_lib::host;
    namespace args = ttnn::kernel_lib::reduce_plan_args;
    for (const auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE, tt::ARCH::QUASAR}) {
        for (const bool fp32_dest : {false, true}) {
            const ReduceHardwareConfig hardware{
                .arch = arch,
                .fp32_dest_acc_en = fp32_dest,
                .dst_full_sync_en = false,
                .available_l1_bytes = 1U << 20,
            };
            for (const auto dtype : {DataType::BFLOAT16, DataType::FLOAT32}) {
                const auto spec = [dtype](const Shape& shape) {
                    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), MemoryConfig{}));
                };
                for (const auto dim : {ReduceOpDim::W, ReduceOpDim::H}) {
                    for (const uint32_t valid : {1U, 15U, 16U, 17U, 31U}) {
                        const auto input = spec(dim == ReduceOpDim::W ? Shape{32, 64 + valid} : Shape{64 + valid, 160});
                        const auto output = spec(dim == ReduceOpDim::W ? Shape{32, 1} : Shape{1, 160});
                        const auto plan = make_reduce_plan(
                            input, output, ReduceOpMath::MAX, dim, 1.0F, ReduceFp32Mode::Fast, hardware);
                        EXPECT_EQ(plan.partial_mode, compute_kernel_lib::ReducePartialMode::Scaler);
                        EXPECT_EQ(plan.partial_reduce_axis_elements, valid);
                        EXPECT_EQ(plan.algorithm, compute_kernel_lib::ReduceAlgorithm::ReduceTile);
                        ASSERT_EQ(plan.auxiliary_tiles.size(), 2U);
                        EXPECT_EQ(plan.auxiliary_tiles[0].num_valid_elements, 32U);
                        EXPECT_EQ(plan.auxiliary_tiles[1].num_valid_elements, valid);
                        EXPECT_EQ(
                            plan.auxiliary_tiles[1].type,
                            dim == ReduceOpDim::W ? ReduceAuxiliaryTileType::FirstRow
                                                  : ReduceAuxiliaryTileType::FirstRowPerFaceRow);
                        if (dim == ReduceOpDim::H) {
                            EXPECT_EQ(plan.chunk.output_tiles, fp32_dest ? 4U : 5U);
                        }
                        const auto words =
                            ReduceCallArgs(plan, {.input_cb_id = 0, .auxiliary_cb_id = 1, .output_cb_id = 16})
                                .get_compile_time_args();
                        EXPECT_EQ(
                            args::extract(
                                words[static_cast<uint32_t>(args::CallWord::Configuration)],
                                args::config::partial_mode_shift,
                                args::config::partial_mode_mask),
                            static_cast<uint32_t>(compute_kernel_lib::ReducePartialMode::Scaler));
                    }
                }
            }
        }
    }
}

TEST(ReduceHostPlanner, PartialMaxScalerRequiresNativeReduceTile) {
    using namespace tt::tt_metal;
    using namespace ttnn::kernel_lib::host;
    const ReduceHardwareConfig hardware{
        .arch = tt::ARCH::BLACKHOLE,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .available_l1_bytes = 1U << 20,
    };
    for (const auto dtype : {DataType::INT32, DataType::FLOAT32}) {
        const auto spec = [dtype](const Shape& shape) {
            return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), MemoryConfig{}));
        };
        for (const auto dim : {ReduceOpDim::W, ReduceOpDim::H}) {
            const auto input = spec(dim == ReduceOpDim::W ? Shape{32, 45} : Shape{45, 32});
            const auto output = spec(dim == ReduceOpDim::W ? Shape{32, 1} : Shape{1, 32});
            EXPECT_ANY_THROW(
                make_reduce_plan(input, output, ReduceOpMath::MAX, dim, 1.0F, ReduceFp32Mode::Accurate, hardware));
        }
    }
}

TEST(ReduceHostPlanner, BasicAlgorithmAndChunkSanity) {
    using namespace tt::tt_metal;
    using namespace ttnn::kernel_lib::host;

    const auto make_tiled_spec = [](const Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    };
    const ReduceHardwareConfig hardware{
        .arch = tt::ARCH::BLACKHOLE,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .available_l1_bytes = 1U << 20,
    };

    const auto output = make_tiled_spec(Shape{1, 1, 32, 1});
    const auto short_plan = make_reduce_plan(
        make_tiled_spec(Shape{1, 1, 32, 3 * 32}),
        output,
        ReduceOpMath::SUM,
        ReduceOpDim::W,
        1.0F,
        ReduceFp32Mode::Fast,
        hardware);
    EXPECT_EQ(short_plan.algorithm, compute_kernel_lib::ReduceAlgorithm::ReduceTile);

    const auto threshold_plan = make_reduce_plan(
        make_tiled_spec(Shape{1, 1, 32, 4 * 32}),
        output,
        ReduceOpMath::SUM,
        ReduceOpDim::W,
        0.25F,
        ReduceFp32Mode::Fast,
        hardware);
    EXPECT_EQ(threshold_plan.algorithm, compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd);
    EXPECT_FLOAT_EQ(threshold_plan.post_scale, 0.25F);

    const auto tile_bytes = tt::tt_metal::tile_size(DataType::BFLOAT16);
    const auto chunked_plan = make_reduce_plan(
        make_tiled_spec(Shape{1, 1, 32, 8 * 32}),
        output,
        ReduceOpMath::SUM,
        ReduceOpDim::W,
        1.0F,
        ReduceFp32Mode::Fast,
        hardware,
        4 * tile_bytes);
    EXPECT_EQ(chunked_plan.input_policy, compute_kernel_lib::ReduceInputPolicy::ChunkedWaitChunkedPop);
    EXPECT_EQ(chunked_plan.chunk.reduce_axis_tiles, 2U);
    ASSERT_NE(chunked_plan.find_cb(ReduceCbRole::Input), nullptr);
    EXPECT_EQ(chunked_plan.find_cb(ReduceCbRole::Input)->total_size_bytes, 4 * tile_bytes);

    const auto col_input = make_tiled_spec(Shape{1, 1, 8 * 32, 8 * 32});
    const auto col_output = make_tiled_spec(Shape{1, 1, 1, 8 * 32});
    const std::vector<ReduceCbConfig> reductions{
        {0U,
         ReduceCallConfig{
             .input_spec = col_input,
             .output_spec = col_output,
             .reduce_math = ReduceOpMath::SUM,
             .reduce_dim = ReduceOpDim::H,
             .scalar = 1.0F,
             .fp32_mode = ReduceFp32Mode::Fast,
             .max_input_cb_bytes = 32 * tile_bytes}},
        {1U,
         ReduceCallConfig{
             .input_spec = col_input,
             .output_spec = col_output,
             .reduce_math = ReduceOpMath::SUM,
             .reduce_dim = ReduceOpDim::H,
             .scalar = 1.0F,
             .fp32_mode = ReduceFp32Mode::Fast,
             .max_input_cb_bytes = 32 * tile_bytes}},
    };
    const auto sequence = make_reduce_sequence_plan(
        reductions, {.auxiliary_cb_id = 2U, .accumulator_cb_id = 4U, .output_cb_id = 3U}, hardware);
    ASSERT_EQ(sequence.calls.size(), 2U);
    ASSERT_EQ(sequence.auxiliary.tiles.size(), 1U);
    EXPECT_EQ(sequence.auxiliary.cb_id, 2U);
    EXPECT_EQ(sequence.calls[0].input_cb_id, 0U);
    EXPECT_EQ(sequence.calls[0].output_cb_id, 4U);
    EXPECT_EQ(sequence.calls[0].accumulator_cb_id, 4U);
    EXPECT_EQ(sequence.calls[0].accumulation_mode, ttnn::kernel_lib::ReduceAccumulationMode::Intermediate);
    EXPECT_EQ(sequence.calls[0].accumulation_index, 0U);
    EXPECT_EQ(sequence.calls[1].input_cb_id, 1U);
    EXPECT_EQ(sequence.calls[1].output_cb_id, 3U);
    EXPECT_EQ(sequence.calls[1].accumulator_cb_id, 4U);
    EXPECT_EQ(sequence.calls[1].accumulation_mode, ttnn::kernel_lib::ReduceAccumulationMode::Final);
    EXPECT_EQ(sequence.calls[1].accumulation_index, 1U);
    EXPECT_EQ(sequence.calls[0].auxiliary_tile_offset, 0U);
    EXPECT_EQ(sequence.calls[1].auxiliary_tile_offset, 0U);
    for (const auto& call : sequence.calls) {
        EXPECT_EQ(call.plan.algorithm, compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd);
        EXPECT_EQ(call.plan.input_policy, compute_kernel_lib::ReduceInputPolicy::ChunkedWaitChunkedPop);
        EXPECT_EQ(call.plan.chunk.reduce_axis_tiles, 2U);
        EXPECT_EQ(call.plan.chunk.output_tiles, 8U);
    }

    auto repeated_cb_reductions = reductions;
    repeated_cb_reductions[1].first = repeated_cb_reductions[0].first;
    const auto repeated_cb_sequence = make_reduce_sequence_plan(
        repeated_cb_reductions, {.auxiliary_cb_id = 2U, .accumulator_cb_id = 4U, .output_cb_id = 3U}, hardware);
    ASSERT_EQ(repeated_cb_sequence.calls.size(), 2U);
    EXPECT_EQ(repeated_cb_sequence.calls[0].input_cb_id, 0U);
    EXPECT_EQ(repeated_cb_sequence.calls[1].input_cb_id, 0U);
    EXPECT_EQ(repeated_cb_sequence.calls[0].accumulation_index, 0U);
    EXPECT_EQ(repeated_cb_sequence.calls[1].accumulation_index, 1U);
    EXPECT_EQ(repeated_cb_sequence.auxiliary.tiles.size(), 1U);

    const auto partial_input = make_tiled_spec(Shape{1, 1, 32, 5 * 32 + 7});
    const std::vector<ReduceCbConfig> partial_reductions{
        {0U,
         ReduceCallConfig{
             .input_spec = partial_input,
             .output_spec = output,
             .reduce_math = ReduceOpMath::SUM,
             .reduce_dim = ReduceOpDim::W,
             .scalar = 1.0F,
             .fp32_mode = ReduceFp32Mode::Fast}},
        {1U,
         ReduceCallConfig{
             .input_spec = partial_input,
             .output_spec = output,
             .reduce_math = ReduceOpMath::SUM,
             .reduce_dim = ReduceOpDim::W,
             .scalar = 1.0F,
             .fp32_mode = ReduceFp32Mode::Fast}},
    };
    const auto partial_sequence = make_reduce_sequence_plan(
        partial_reductions, {.auxiliary_cb_id = 2U, .accumulator_cb_id = 4U, .output_cb_id = 3U}, hardware);
    ASSERT_EQ(partial_sequence.calls.size(), 2U);
    ASSERT_EQ(partial_sequence.auxiliary.tiles.size(), 2U);
    EXPECT_EQ(partial_sequence.auxiliary.tiles[0].type, ReduceAuxiliaryTileType::FirstRow);
    EXPECT_EQ(partial_sequence.auxiliary.tiles[1].type, ReduceAuxiliaryTileType::Zero);
    EXPECT_EQ(partial_sequence.calls[0].auxiliary_tile_offset, 0U);
    EXPECT_EQ(partial_sequence.calls[1].auxiliary_tile_offset, 0U);
    EXPECT_EQ(partial_sequence.calls[0].plan.partial_mode, compute_kernel_lib::ReducePartialMode::Mask);
    EXPECT_EQ(partial_sequence.calls[1].plan.partial_mode, compute_kernel_lib::ReducePartialMode::Mask);
    ASSERT_EQ(partial_sequence.calls[0].plan.auxiliary_tiles.size(), 1U);
    EXPECT_EQ(partial_sequence.calls[0].plan.auxiliary_tiles[0].type, ReduceAuxiliaryTileType::FirstRow);
    EXPECT_EQ(partial_sequence.calls[0].plan.auxiliary_tiles[0].num_valid_elements, 7U);
    EXPECT_EQ(partial_sequence.calls[1].plan.reload_mode, compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair);
    ASSERT_EQ(partial_sequence.calls[1].plan.auxiliary_tiles.size(), 2U);
    EXPECT_EQ(partial_sequence.calls[1].plan.auxiliary_tiles[0].type, ReduceAuxiliaryTileType::FirstRow);
    EXPECT_EQ(partial_sequence.calls[1].plan.auxiliary_tiles[1].type, ReduceAuxiliaryTileType::Zero);
    ASSERT_NE(partial_sequence.calls[1].plan.find_cb(ReduceCbRole::Auxiliary), nullptr);
    EXPECT_EQ(partial_sequence.calls[1].plan.find_cb(ReduceCbRole::Auxiliary)->page_count, 2U);
    const auto first_serialized = ReduceCallArgs(partial_sequence.calls[0]).get_compile_time_args();
    const auto second_serialized = ReduceCallArgs(partial_sequence.calls[1]).get_compile_time_args();
    EXPECT_EQ(first_serialized.size(), ttnn::kernel_lib::reduce_plan_args::call_word_count);
    EXPECT_EQ(second_serialized.size(), ttnn::kernel_lib::reduce_plan_args::call_word_count);
    constexpr std::uint32_t kernel_owned_arg = 17U;
    std::vector<std::uint32_t> serialized{kernel_owned_arg};
    partial_sequence.append_to(serialized);
    namespace plan_args = ttnn::kernel_lib::reduce_plan_args;
    ASSERT_EQ(serialized.size(), 1 + plan_args::call_count_word_count + 2 * plan_args::call_word_count);
    EXPECT_EQ(serialized[0], kernel_owned_arg);
    EXPECT_EQ(serialized[1], 2U);
    const std::uint32_t first_offset = 1 + plan_args::call_count_word_count;
    const auto second_offset = first_offset + plan_args::call_word_count;
    const auto second_configuration =
        serialized[plan_args::call_word_offset(second_offset, plan_args::CallWord::Configuration)];
    EXPECT_EQ(
        plan_args::extract(
            second_configuration, plan_args::config::partial_mode_shift, plan_args::config::partial_mode_mask),
        static_cast<std::uint32_t>(compute_kernel_lib::ReducePartialMode::Mask));
    const auto second_cbs =
        serialized[plan_args::call_word_offset(second_offset, plan_args::CallWord::CircularBuffers)];
    EXPECT_EQ(
        plan_args::extract(second_cbs, plan_args::circular_buffers::input_shift, plan_args::circular_buffers::id_mask),
        1U);
    EXPECT_EQ(
        plan_args::extract(second_cbs, plan_args::circular_buffers::output_shift, plan_args::circular_buffers::id_mask),
        3U);
    const auto second_chunk_and_auxiliary =
        serialized[plan_args::call_word_offset(second_offset, plan_args::CallWord::ChunkAndAuxiliary)];
    EXPECT_EQ(
        plan_args::extract(
            second_chunk_and_auxiliary,
            plan_args::chunk_and_auxiliary::auxiliary_tile_offset_shift,
            plan_args::chunk_and_auxiliary::auxiliary_tile_offset_mask),
        0U);
    EXPECT_EQ(
        plan_args::extract(
            second_chunk_and_auxiliary,
            plan_args::chunk_and_auxiliary::auxiliary_tile_count_shift,
            plan_args::chunk_and_auxiliary::auxiliary_tile_count_mask),
        2U);

    constexpr std::uint32_t auxiliary_kernel_owned_arg = 23U;
    std::vector<std::uint32_t> auxiliary_serialized{auxiliary_kernel_owned_arg};
    partial_sequence.append_auxiliary_to(auxiliary_serialized);
    ASSERT_EQ(
        auxiliary_serialized.size(),
        1 + plan_args::auxiliary_header_word_count +
            partial_sequence.auxiliary.tiles.size() * plan_args::auxiliary_tile_word_count);
    const auto auxiliary_header = auxiliary_serialized[1];
    EXPECT_EQ(
        plan_args::extract(
            auxiliary_header, plan_args::auxiliary_header::cb_id_shift, plan_args::auxiliary_header::cb_id_mask),
        2U);
    EXPECT_EQ(
        plan_args::extract(
            auxiliary_header,
            plan_args::auxiliary_header::tile_count_shift,
            plan_args::auxiliary_header::tile_count_mask),
        2U);
    const auto auxiliary_tiles_offset = plan_args::auxiliary_tiles_offset(1);
    const auto zero_config = auxiliary_serialized[plan_args::auxiliary_tile_word_offset(
        auxiliary_tiles_offset, 1, plan_args::AuxiliaryTileWord::Configuration)];
    EXPECT_EQ(
        plan_args::extract(
            zero_config,
            plan_args::auxiliary_configuration::tile_type_shift,
            plan_args::auxiliary_configuration::tile_type_mask),
        static_cast<std::uint32_t>(ReduceAuxiliaryTileType::Zero));

    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    const MemoryConfig sharded_l1(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(shard_grid, {32, 4 * 32}, ShardOrientation::ROW_MAJOR));
    const auto sharded_input =
        TensorSpec(Shape{1, 1, 32, 4 * 32}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), sharded_l1));
    const auto alias_plan = make_reduce_plan(
        sharded_input, output, ReduceOpMath::SUM, ReduceOpDim::W, 1.0F, ReduceFp32Mode::Fast, hardware, 0);
    EXPECT_EQ(alias_plan.input_policy, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop);
    ASSERT_NE(alias_plan.find_cb(ReduceCbRole::Input), nullptr);
    EXPECT_EQ(alias_plan.find_cb(ReduceCbRole::Input)->alias, ReduceCbAlias::InputTensor);
    EXPECT_ANY_THROW(make_reduce_plan(
        make_tiled_spec(Shape{1, 1, 32, 4 * 32}),
        output,
        ReduceOpMath::SUM,
        ReduceOpDim::W,
        1.0F,
        ReduceFp32Mode::Fast,
        hardware,
        0));
}

}  // namespace ttnn::operations::reduction::test
