// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/transformer/concatenate_heads/concatenate_heads.hpp"
#include "ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using Parameters = std::map<std::string, ttnn::Tensor>;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

ttnn::MemoryConfig l1_memory_config =
    ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
ttnn::MemoryConfig dram_memory_config =
    ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

ttnn::Tensor encoder(
    ttnn::Tensor&& hidden_states,
    const ttnn::Tensor& attention_mask,
    const Parameters& parameters,
    std::size_t encoder_index,
    const std::uint32_t head_size) {
    auto batch_size = hidden_states.padded_shape()[0];

    auto fused_qkv_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .out_block_h = 12,
        .out_block_w = 8,
        .per_core_M = 12,
        .per_core_N = 8,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    ttnn::prim::MatmulParams attributes;
    attributes.program_config = fused_qkv_matmul_program_config;
    attributes.output_mem_config = l1_memory_config;

    auto fused_qkv_matmul_output = ttnn::prim::matmul(
                                       hidden_states,
                                       parameters.at(fmt::format("fused_qkv_weight_{}", encoder_index)),
                                       parameters.at(fmt::format("fused_qkv_bias_{}", encoder_index)),
                                       std::nullopt,
                                       attributes)
                                       .at(0);

    auto&& [query, key, value] = ttnn::experimental::split_query_key_value_and_split_heads(
        fused_qkv_matmul_output, CoreCoord{12, batch_size}, l1_memory_config);
    fused_qkv_matmul_output.deallocate();

    auto pre_softmax_bmm_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 1,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 12,
    };

    attributes.program_config = pre_softmax_bmm_program_config;
    attributes.output_mem_config = dram_memory_config;
    auto pre_softmax_bmm_matmul = ttnn::prim::matmul(
                                      query,
                                      key,
                                      /*bias=*/std::nullopt,
                                      /*optional_output_tensor=*/std::nullopt,
                                      attributes)
                                      .at(0);

    query.deallocate();
    key.deallocate();

    pre_softmax_bmm_matmul =
        ttnn::scale_mask_softmax_in_place(pre_softmax_bmm_matmul, 1.0f / std::sqrt(head_size), attention_mask);

    auto post_softmax_bmm_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 2,
        .out_subblock_h = 4,
        .out_subblock_w = 2,
        .per_core_M = 12,
        .per_core_N = 2,
    };

    attributes.program_config = post_softmax_bmm_program_config;
    attributes.output_mem_config = l1_memory_config;
    auto post_softmax_bmm_output = ttnn::prim::matmul(
                                       pre_softmax_bmm_matmul,
                                       value,
                                       /*bias=*/std::nullopt,
                                       /*optional_output_tensor=*/std::nullopt,
                                       attributes)
                                       .at(0);
    pre_softmax_bmm_matmul.deallocate();
    value.deallocate();

    auto concat_heads_output =
        ttnn::experimental::concatenate_heads(post_softmax_bmm_output, CoreCoord{12, batch_size}, l1_memory_config);
    post_softmax_bmm_output.deallocate();

    auto selfout_bmm_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .out_block_h = 12,
        .out_block_w = 3,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };

    attributes.program_config = selfout_bmm_program_config;
    attributes.output_mem_config = l1_memory_config;
    auto selfout_bmm_output = ttnn::prim::matmul(
                                  concat_heads_output,
                                  parameters.at(fmt::format("selfout_weight_{}", encoder_index)),
                                  parameters.at(fmt::format("selfout_bias_{}", encoder_index)),
                                  /*optional_output_tensor=*/std::nullopt,
                                  attributes)
                                  .at(0);
    concat_heads_output.deallocate();

    auto attention_layernorm_output = ttnn::layer_norm(
        hidden_states,
        1e-12,
        parameters.at(fmt::format("attention_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("attention_layernorm_bias_{}", encoder_index)),
        selfout_bmm_output,
        l1_memory_config);
    hidden_states.deallocate();
    selfout_bmm_output.deallocate();

    auto ff1_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .out_block_h = 12,
        .out_block_w = 11,
        .per_core_M = 12,
        .per_core_N = 11,
        .transpose_mcast = false,
        .fused_activation = UnaryWithParam(UnaryOpType::GELU, 1.0f),
    };

    attributes.program_config = ff1_matmul_program_config;
    attributes.output_mem_config = dram_memory_config;
    auto ff1_matmul_output = ttnn::prim::matmul(
                                 attention_layernorm_output,
                                 parameters.at(fmt::format("ff1_weight_{}", encoder_index)),
                                 parameters.at(fmt::format("ff1_bias_{}", encoder_index)),
                                 /*optional_output_tensor=*/std::nullopt,
                                 attributes)
                                 .at(0);

    auto ff2_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {12, batch_size},
        .in0_block_w = 4,
        .out_subblock_h = 6,
        .out_subblock_w = 1,
        .out_block_h = 12,
        .out_block_w = 3,
        .per_core_M = 12,
        .per_core_N = 3,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };

    attributes.program_config = ff2_matmul_program_config;
    attributes.output_mem_config = l1_memory_config;
    auto ff2_matmul_output = ttnn::prim::matmul(
                                 ff1_matmul_output,
                                 parameters.at(fmt::format("ff2_weight_{}", encoder_index)),
                                 parameters.at(fmt::format("ff2_bias_{}", encoder_index)),
                                 /*optional_output_tensor=*/std::nullopt,
                                 attributes)
                                 .at(0);
    ff1_matmul_output.deallocate();

    auto feedforward_layernorm_output = ttnn::layer_norm(
        attention_layernorm_output,
        1e-12,
        parameters.at(fmt::format("feedforward_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("feedforward_layernorm_bias_{}", encoder_index)),
        ff2_matmul_output,
        l1_memory_config);
    attention_layernorm_output.deallocate();
    ff2_matmul_output.deallocate();

    return feedforward_layernorm_output;
}

ttnn::Tensor qa_head(ttnn::Tensor&& hidden_states, const Parameters& parameters) {
    auto output = ttnn::prim::matmul(hidden_states, parameters.at("qa_head_weight"), /*bias=*/std::nullopt).at(0);
    hidden_states.deallocate();

    return ttnn::add(output, parameters.at("qa_head_bias"), std::nullopt, l1_memory_config);
}

void test_bert() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using tt::tt_metal::DataType;
    using tt::tt_metal::IDevice;

    using tt::tt_metal::Layout;
    using tt::tt_metal::Tensor;

    int device_id = 0;
    auto device_owner = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto* device = device_owner.get();
    CoreCoord compute_grid_size = device->compute_with_storage_grid_size();

    if (compute_grid_size.x * compute_grid_size.y == 88) {
        log_info(tt::LogTest, "Skipping test_bert for E75");
        return;
    }

    std::size_t num_iterations = 2;
    std::size_t num_encoders = 24;
    std::uint32_t batch_size = 9;
    std::uint32_t sequence_size = 384;
    std::uint32_t num_heads = 16;
    std::uint32_t head_size = 64;
    std::uint32_t hidden_size = num_heads * head_size;
    std::uint32_t intermediate_size = hidden_size * 4;

    auto attention_mask =
        ttnn::random::uniform(
            bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({batch_size, 1, TILE_HEIGHT, sequence_size}), Layout::TILE)
            .to_device(device, l1_memory_config);

    auto parameters = Parameters{};
    for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
        parameters.emplace(
            fmt::format("fused_qkv_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, hidden_size, hidden_size * 3}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("fused_qkv_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, hidden_size * 3}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("selfout_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, hidden_size, hidden_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("selfout_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, hidden_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("attention_layernorm_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), Layout::ROW_MAJOR)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("attention_layernorm_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), Layout::ROW_MAJOR)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("ff1_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, hidden_size, intermediate_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("ff1_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, intermediate_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("ff2_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, intermediate_size, hidden_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("ff2_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, hidden_size}), Layout::TILE)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("feedforward_layernorm_weight_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), Layout::ROW_MAJOR)
                .to_device(device, dram_memory_config));
        parameters.emplace(
            fmt::format("feedforward_layernorm_bias_{}", encoder_index),
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), Layout::ROW_MAJOR)
                .to_device(device, dram_memory_config));
    };
    parameters.emplace(
        "qa_head_weight",
        ttnn::random::uniform(
            bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, hidden_size, TILE_WIDTH}), Layout::TILE)
            .to_device(device, dram_memory_config));
    parameters.emplace(
        "qa_head_bias",
        ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({1, 1, 1, TILE_WIDTH}), Layout::TILE)
            .to_device(device, dram_memory_config));

    auto run_bert = [&]() {
        log_debug(tt::LogTest, "run_bert started");
        auto begin = std::chrono::steady_clock::now();
        auto hidden_states =
            ttnn::random::uniform(
                bfloat16(-1.0f), bfloat16(1.0f), ttnn::Shape({batch_size, 1, sequence_size, hidden_size}), Layout::TILE)
                .to_device(device, l1_memory_config);
        for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
            hidden_states = encoder(std::move(hidden_states), attention_mask, parameters, encoder_index, head_size);
        }
        auto output = qa_head(std::move(hidden_states), parameters).cpu();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        log_info(tt::LogTest, "run_bert finished in {} microseconds", duration);
        return duration;
    };

    auto run_loop = [&]() {
        auto total_duration = 0;
        for (int iteration = 0; iteration < num_iterations; iteration++) {
            total_duration += run_bert();
        }
        auto average_duration = total_duration / num_iterations;
        auto num_samples_per_second = 1e6 / average_duration * batch_size;
        log_info(tt::LogTest, "total duration: {} microseconds", total_duration);
        log_info(tt::LogTest, "average duration: {} average_duration", total_duration);
        log_info(tt::LogTest, "samples per second: {}", num_samples_per_second);
    };

    run_bert();
    run_loop();
}

int main() {
    test_bert();
    return 0;
}
