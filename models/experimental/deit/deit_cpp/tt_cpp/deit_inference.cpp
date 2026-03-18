// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "deit_inference.h"
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp>
#include <ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp>
#include <ttnn/operations/transformer/attention_softmax/attention_softmax.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/squeeze/squeeze.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

namespace deit_inference {

void update_model_config(deit_inference::DeiTConfig& config, int batch_size) {
    config.batch_size = batch_size;
    config.core_grid = ttnn::CoreCoord{3, static_cast<std::size_t>(batch_size)};

    std::size_t seqL_t = 224 / 32;                      // 7
    std::size_t dim_t = 192 / 32;                       // 6
    std::size_t dim_t__x = dim_t / config.core_grid.x;  // 2
    std::size_t head_num = 3;
    std::size_t head_seqL_t = head_num * seqL_t / config.core_grid.x;  // 7
    std::size_t head_size_t__x = dim_t / head_num;                     // 2
    std::size_t class__x = 1152 / 32 / config.core_grid.x;             // 3

    auto grid = config.core_grid;

    config.query_key_value_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = dim_t__x / 2,
        .out_subblock_h = 1,
        .out_subblock_w = 2 * dim_t__x / 2,
        .out_block_h = seqL_t,
        .out_block_w = 3 * dim_t__x,
        .per_core_M = seqL_t,
        .per_core_N = 3 * dim_t__x,
        .transpose_mcast = false,
        .fused_activation = std::nullopt};

    config.query_by_key_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = dim_t__x,
        .out_subblock_h = 1,
        .out_subblock_w = seqL_t,
        .per_core_M = seqL_t,
        .per_core_N = head_seqL_t};

    config.attention_probabilities_by_value_matmul_program_config =
        ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = grid,
            .in0_block_w = seqL_t,
            .out_subblock_h = 1,
            .out_subblock_w = head_size_t__x,
            .per_core_M = seqL_t,
            .per_core_N = head_size_t__x};

    config.self_output_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = dim_t__x / 2,
        .out_subblock_h = 1,
        .out_subblock_w = dim_t__x,
        .out_block_h = seqL_t,
        .out_block_w = dim_t__x,
        .per_core_M = seqL_t,
        .per_core_N = dim_t__x,
        .transpose_mcast = false,
        .fused_activation = std::nullopt};

    config.ff1_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = dim_t__x,
        .out_subblock_h = 1,
        .out_subblock_w = dim_t__x,
        .out_block_h = seqL_t,
        .out_block_w = 4 * dim_t__x,
        .per_core_M = seqL_t,
        .per_core_N = 4 * dim_t__x,
        .transpose_mcast = false,
        .fused_activation =
            ttnn::operations::unary::UnaryWithParam{
                ttnn::operations::unary::UnaryOpType::GELU, std::vector<float>{1.0f}}  // gelu=True
    };

    config.ff2_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = 2 * dim_t__x,
        .out_subblock_h = 1,
        .out_subblock_w = dim_t__x,
        .out_block_h = seqL_t,
        .out_block_w = dim_t__x,
        .per_core_M = seqL_t,
        .per_core_N = dim_t__x,
        .transpose_mcast = false,
        .fused_activation = std::nullopt};

    config.classifer_matmul_program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = dim_t__x,
        .out_subblock_h = 1,
        .out_subblock_w = class__x / 2,
        .out_block_h = seqL_t,
        .out_block_w = class__x,
        .per_core_M = seqL_t,
        .per_core_N = class__x,
        .transpose_mcast = false,
        .fused_activation = ttnn::operations::unary::UnaryWithParam{
            ttnn::operations::unary::UnaryOpType::GELU, std::vector<float>{1.0f}}};

    config.layernorm_program_config = ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = grid,
        .subblock_w = dim_t__x / 2,
        .block_h = seqL_t,
        .block_w = dim_t__x,
        .inplace = false};

    config.layernorm_after_output_program_config = ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = grid,
        .subblock_w = dim_t__x / 2,
        .block_h = seqL_t,
        .block_w = dim_t__x,
        .inplace = false};

    config.softmax_program_config = ttnn::SoftmaxShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = grid, .subblock_w = head_seqL_t, .block_h = seqL_t, .block_w = head_seqL_t};
}

ttnn::Tensor deit_patch_embeddings(
    const deit_inference::DeiTConfig& /*config*/,
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias) {
    uint32_t batch_size = pixel_values.logical_shape()[0];
    uint32_t img_h = pixel_values.logical_shape()[1];
    uint32_t img_c = pixel_values.logical_shape()[3];

    uint32_t patch_size = 16;
    uint32_t patch_count = img_h / patch_size;
    uint32_t embedding_dim = 192;
    uint32_t patch_count_all = patch_count * patch_count;

    auto pixel_values_interl = ttnn::to_memory_config(pixel_values, ttnn::L1_MEMORY_CONFIG, std::nullopt);
    auto folded_pixel_values =
        ttnn::reshape(pixel_values_interl, ttnn::Shape({batch_size, patch_count, patch_size, patch_count, img_c}));
    folded_pixel_values = ttnn::permute(folded_pixel_values, ttnn::SmallVector<int64_t>{0, 1, 3, 2, 4});
    folded_pixel_values =
        ttnn::reshape(folded_pixel_values, ttnn::Shape({batch_size, patch_count, patch_count, patch_size * img_c}));

    folded_pixel_values =
        ttnn::to_layout(folded_pixel_values, ttnn::TILE_LAYOUT, ttnn::DataType::BFLOAT8_B, ttnn::L1_MEMORY_CONFIG);

    auto patch_embedding_output = ttnn::linear(
        folded_pixel_values, proj_weight, proj_bias, false, false, ttnn::L1_MEMORY_CONFIG, ttnn::DataType::BFLOAT16);

    patch_embedding_output = ttnn::to_layout(patch_embedding_output, ttnn::ROW_MAJOR_LAYOUT);
    patch_embedding_output =
        ttnn::reshape(patch_embedding_output, ttnn::Shape({batch_size, patch_count_all, embedding_dim}));

    return patch_embedding_output;
}

ttnn::Tensor deit_embeddings(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor& cls_token,
    const ttnn::Tensor& distillation_token,
    const ttnn::Tensor& position_embeddings,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias) {
    auto patch_embeds = deit_patch_embeddings(config, pixel_values, proj_weight, proj_bias);
    auto embedding_output = ttnn::concat(
        std::vector<ttnn::Tensor>{cls_token, distillation_token, patch_embeds}, -2, ttnn::L1_MEMORY_CONFIG);
    embedding_output = ttnn::to_layout(embedding_output, ttnn::TILE_LAYOUT);
    embedding_output = ttnn::add(embedding_output, position_embeddings, std::nullopt, ttnn::L1_MEMORY_CONFIG);

    return embedding_output;
}

ttnn::Tensor deit_layernorm_before(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& layernorm_weight,
    const ttnn::Tensor& layernorm_bias) {
    return ttnn::layer_norm(
        hidden_states,
        config.layer_norm_eps,
        layernorm_weight,
        layernorm_bias,
        std::nullopt,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        config.layernorm_program_config);
}

ttnn::Tensor deit_layernorm_after(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& layernorm_weight,
    const ttnn::Tensor& layernorm_bias) {
    return ttnn::layer_norm(
        hidden_states,
        config.layer_norm_eps,
        layernorm_weight,
        layernorm_bias,
        std::nullopt,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        config.layernorm_after_output_program_config);
}

ttnn::Tensor deit_attention(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& attention_mask,
    const ttnn::Tensor& qkv_weight,
    const ttnn::Tensor& qkv_bias,
    const ttnn::Tensor& out_weight,
    const ttnn::Tensor& out_bias) {
    uint32_t num_heads = config.num_heads;
    uint32_t head_size = config.hidden_size / num_heads;

    auto query_key_value = ttnn::linear(
        hidden_states,
        qkv_weight,
        qkv_bias,
        false,
        false,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.query_key_value_matmul_program_config);

    if (query_key_value.logical_shape().rank() == 4 && query_key_value.logical_shape()[1] == 1) {
        query_key_value = ttnn::squeeze(query_key_value, 1);
    }

    auto [query, key, value] = ttnn::transformer::split_query_key_value_and_split_heads(
        query_key_value, std::nullopt, num_heads, std::nullopt, true, ttnn::L1_HEIGHT_SHARDED_MEMORY_CONFIG);

    query_key_value.deallocate();

    auto attention_scores = ttnn::matmul(
        query,
        key,
        false,
        false,
        ttnn::L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.query_by_key_matmul_program_config);

    query.deallocate();
    key.deallocate();

    auto attention_probs = ttnn::transformer::attention_softmax(
        attention_scores, head_size, attention_mask, config.softmax_program_config);

    auto context_layer = ttnn::matmul(
        attention_probs,
        value,
        false,
        false,
        ttnn::L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.attention_probabilities_by_value_matmul_program_config);

    attention_probs.deallocate();
    value.deallocate();

    context_layer = ttnn::transformer::concatenate_heads(context_layer, ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG);

    auto self_output = ttnn::linear(
        context_layer,
        out_weight,
        out_bias,
        false,
        false,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.self_output_matmul_program_config);

    context_layer.deallocate();

    return self_output;
}

ttnn::Tensor deit_intermediate(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias) {
    auto output = ttnn::linear(
        hidden_states,
        dense_weight,
        dense_bias,
        false,
        false,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.ff1_matmul_program_config);

    return output;
}

ttnn::Tensor deit_output(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias) {
    auto output = ttnn::linear(
        hidden_states,
        dense_weight,
        dense_bias,
        false,
        false,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B,
        config.ff2_matmul_program_config);

    output = ttnn::add(output, residual, std::nullopt, ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG);
    return output;
}

ttnn::Tensor deit_feedforward(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& attention_output,
    const ttnn::Tensor& intermediate_weight,
    const ttnn::Tensor& intermediate_bias,
    const ttnn::Tensor& output_weight,
    const ttnn::Tensor& output_bias) {
    auto intermediate = deit_intermediate(config, hidden_states, intermediate_weight, intermediate_bias);
    auto output = deit_output(config, intermediate, attention_output, output_weight, output_bias);
    return output;
}

ttnn::Tensor deit_layer(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& attention_mask,
    const ttnn::Tensor& layernorm_before_weight,
    const ttnn::Tensor& layernorm_before_bias,
    const ttnn::Tensor& qkv_weight,
    const ttnn::Tensor& qkv_bias,
    const ttnn::Tensor& attention_out_weight,
    const ttnn::Tensor& attention_out_bias,
    const ttnn::Tensor& layernorm_after_weight,
    const ttnn::Tensor& layernorm_after_bias,
    const ttnn::Tensor& intermediate_weight,
    const ttnn::Tensor& intermediate_bias,
    const ttnn::Tensor& output_weight,
    const ttnn::Tensor& output_bias) {
    auto layernorm_before_output =
        deit_layernorm_before(config, hidden_states, layernorm_before_weight, layernorm_before_bias);

    auto multi_head_attention_output = deit_attention(
        config,
        layernorm_before_output,
        attention_mask,
        qkv_weight,
        qkv_bias,
        attention_out_weight,
        attention_out_bias);

    multi_head_attention_output =
        ttnn::add(multi_head_attention_output, hidden_states, std::nullopt, ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG);

    auto layernorm_after_output =
        deit_layernorm_after(config, multi_head_attention_output, layernorm_after_weight, layernorm_after_bias);

    auto feedforward_output = deit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias);

    return feedforward_output;
}

ttnn::Tensor deit_encoder(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& embeddings,
    const std::vector<std::optional<ttnn::Tensor>>& head_masks,
    const std::unordered_map<std::string, ttnn::Tensor>& parameters) {
    uint32_t seqL_t = 224 / 32;
    uint32_t dim_t = 192 / 32;
    uint32_t dim_t__x = dim_t / config.core_grid.x;

    ttnn::CoreCoord grid = config.core_grid;
    tt::tt_metal::CoreRange cr{ttnn::CoreCoord{0, 0}, ttnn::CoreCoord{grid.x - 1, grid.y - 1}};
    tt::tt_metal::CoreRangeSet crs{cr};
    std::array<uint32_t, 2> shard_shape = {seqL_t * 32, dim_t__x * 32};
    tt::tt_metal::ShardSpec shard_spec{crs, shard_shape, tt::tt_metal::ShardOrientation::ROW_MAJOR};
    auto sharded_mem_config =
        ttnn::MemoryConfig{tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    auto encoder_input = ttnn::to_memory_config(embeddings, sharded_mem_config, std::nullopt);

    ttnn::Tensor encoder_output = encoder_input;

    for (int i = 0; i < config.num_layers; ++i) {
        std::string layer_prefix = "deit.encoder.layer." + std::to_string(i) + ".";

        encoder_output = deit_layer(
            config,
            encoder_output,
            head_masks[i],
            parameters.at(layer_prefix + "layernorm_before.weight"),
            parameters.at(layer_prefix + "layernorm_before.bias"),
            parameters.at(layer_prefix + "attention.attention.qkv.weight"),
            parameters.at(layer_prefix + "attention.attention.qkv.bias"),
            parameters.at(layer_prefix + "attention.output.dense.weight"),
            parameters.at(layer_prefix + "attention.output.dense.bias"),
            parameters.at(layer_prefix + "layernorm_after.weight"),
            parameters.at(layer_prefix + "layernorm_after.bias"),
            parameters.at(layer_prefix + "intermediate.dense.weight"),
            parameters.at(layer_prefix + "intermediate.dense.bias"),
            parameters.at(layer_prefix + "output.dense.weight"),
            parameters.at(layer_prefix + "output.dense.bias"));
    }

    return encoder_output;
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> deit(
    const deit_inference::DeiTConfig& config,
    const ttnn::Tensor& pixel_values,
    const std::vector<std::optional<ttnn::Tensor>>& head_masks,
    const ttnn::Tensor& cls_token,
    const ttnn::Tensor& distillation_token,
    const ttnn::Tensor& position_embeddings,
    const std::unordered_map<std::string, ttnn::Tensor>& parameters) {
    auto embeddings_output = deit_inference::deit_embeddings(
        config,
        pixel_values,
        cls_token,
        distillation_token,
        position_embeddings,
        parameters.at("deit.embeddings.patch_embeddings.projection.weight"),
        parameters.at("deit.embeddings.patch_embeddings.projection.bias"));

    auto hidden_states = deit_inference::deit_encoder(config, embeddings_output, head_masks, parameters);

    auto output = ttnn::layer_norm(
        hidden_states,
        config.layer_norm_eps,
        parameters.at("deit.layernorm.weight"),
        parameters.at("deit.layernorm.bias"),
        std::nullopt,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        config.layernorm_program_config);

    if (output.logical_shape().rank() == 4 && output.logical_shape()[1] == 1) {
        output = ttnn::squeeze(output, 1);
    }

    const auto& shape = output.logical_shape();
    uint32_t batch_size = shape[0];
    uint32_t hidden_dim = shape[shape.rank() - 1];

    // Dual classifier path (DeiTForImageClassificationWithTeacher)
    if (parameters.count("cls_classifier.weight") && parameters.count("distillation_classifier.weight")) {
        auto cls_token_output = ttnn::slice(
            output,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{batch_size, 1, hidden_dim},
            ttnn::SmallVector<uint32_t>{1, 1, 1},
            ttnn::L1_MEMORY_CONFIG);

        auto distillation_token_output = ttnn::slice(
            output,
            ttnn::SmallVector<uint32_t>{0, 1, 0},
            ttnn::SmallVector<uint32_t>{batch_size, 2, hidden_dim},
            ttnn::SmallVector<uint32_t>{1, 1, 1},
            ttnn::L1_MEMORY_CONFIG);

        output.deallocate();

        auto cls_logits = ttnn::linear(
            cls_token_output,
            parameters.at("cls_classifier.weight"),
            parameters.at("cls_classifier.bias"),
            false,
            false,
            ttnn::L1_MEMORY_CONFIG,
            ttnn::DataType::BFLOAT8_B);

        cls_token_output.deallocate();

        auto distillation_logits = ttnn::linear(
            distillation_token_output,
            parameters.at("distillation_classifier.weight"),
            parameters.at("distillation_classifier.bias"),
            false,
            false,
            ttnn::L1_MEMORY_CONFIG,
            ttnn::DataType::BFLOAT8_B);

        distillation_token_output.deallocate();

        auto logits = ttnn::multiply(ttnn::add(cls_logits, distillation_logits), 0.5f);
        return {logits, cls_logits, distillation_logits};
    }

    // Single classifier path
    auto classifier_output = ttnn::linear(
        output,
        parameters.at("classifier.weight"),
        parameters.at("classifier.bias"),
        false,
        false,
        ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT8_B);

    return {classifier_output, classifier_output, classifier_output};
}

}  // namespace deit_inference

std::unordered_map<std::string, ttnn::Tensor> custom_preprocessor(
    const ttnn::Tensor& patch_weight,
    const ttnn::Tensor& patch_bias,
    const ttnn::Tensor& q_w,
    const ttnn::Tensor& q_b,
    const ttnn::Tensor& k_w,
    const ttnn::Tensor& k_b,
    const ttnn::Tensor& v_w,
    const ttnn::Tensor& v_b,
    const ttnn::Tensor& classifier_weight,
    const ttnn::Tensor& classifier_bias) {
    std::unordered_map<std::string, ttnn::Tensor> parameters;

    // 1. Patch Embeddings
    const auto& pw_shape = patch_weight.logical_shape();
    // Assuming weight shape: [three_times_hidden_size, c, patch_h, patch_w]
    if (pw_shape.rank() == 4) {
        // Pad channel from 3 to 4, permute to (2,3,1,0), reshape
        // Note: For C++ we usually assume these weights are preprocessed offline via python,
        // but this demonstrates the equivalent logic structure.
        parameters["patch_embeddings.projection.weight"] = patch_weight;
        parameters["patch_embeddings.projection.bias"] = patch_bias;
    }

    // 2. QKV
    // In C++ we can concatenate Q, K, V weights
    auto qkv_w = ttnn::concat(std::vector<ttnn::Tensor>{q_w, k_w, v_w}, -1, std::nullopt);
    auto qkv_b = ttnn::concat(std::vector<ttnn::Tensor>{q_b, k_b, v_b}, -1, std::nullopt);
    parameters["query_key_value.weight"] = qkv_w;
    parameters["query_key_value.bias"] = qkv_b;

    // 3. Classifier
    const auto& cls_shape = classifier_weight.logical_shape();
    if (cls_shape.rank() >= 1 && cls_shape[0] == 1000) {
        // Pad from 1000 to 1152
        // auto padded_cls_shape = ttnn::Shape{1152, cls_shape[1]};
        // parameters["classifier.weight"] = ttnn::pad(classifier_weight, padded_cls_shape, tt::tt_metal::Array4D{0, 0,
        // 0, 0}, 0.0f);
        parameters["classifier.weight"] = classifier_weight;  // Placeholder
        parameters["classifier.bias"] = classifier_bias;
    } else {
        parameters["classifier.weight"] = classifier_weight;
        parameters["classifier.bias"] = classifier_bias;
    }

    return parameters;
}
