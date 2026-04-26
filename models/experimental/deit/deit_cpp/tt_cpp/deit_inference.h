// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace deit_inference {

struct DeiTConfig {
    int batch_size = 1;
    int image_size = 224;
    int patch_size = 16;
    int hidden_size = 192;
    int num_heads = 3;
    int num_layers = 12;
    int intermediate_size = 768;
    float layer_norm_eps = 1e-12f;

    ttnn::CoreCoord core_grid{3, 1};

    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig query_key_value_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig query_by_key_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig attention_probabilities_by_value_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig self_output_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig ff1_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig ff2_matmul_program_config;
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig classifer_matmul_program_config;

    ttnn::prim::LayerNormShardedMultiCoreProgramConfig layernorm_program_config;
    ttnn::prim::LayerNormShardedMultiCoreProgramConfig layernorm_after_output_program_config;
    ttnn::SoftmaxShardedMultiCoreProgramConfig softmax_program_config;
};

void update_model_config(DeiTConfig& config, int batch_size);

ttnn::Tensor deit_patch_embeddings(
    const DeiTConfig& config,
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias);

ttnn::Tensor deit_embeddings(
    const DeiTConfig& config,
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor& cls_token,
    const ttnn::Tensor& distillation_token,
    const ttnn::Tensor& position_emb,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias);

ttnn::Tensor deit_layernorm_before(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& layernorm_before_weight,
    const ttnn::Tensor& layernorm_before_bias);

ttnn::Tensor deit_layernorm_after(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& layernorm_after_weight,
    const ttnn::Tensor& layernorm_after_bias);

ttnn::Tensor deit_attention(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& attention_mask,
    const ttnn::Tensor& qkv_weight,
    const ttnn::Tensor& qkv_bias,
    const ttnn::Tensor& out_weight,
    const ttnn::Tensor& out_bias);

ttnn::Tensor deit_intermediate(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias);

ttnn::Tensor deit_output(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias);

ttnn::Tensor deit_feedforward(
    const DeiTConfig& config,
    const ttnn::Tensor& hidden_states,
    const ttnn::Tensor& attention_output,
    const ttnn::Tensor& intermediate_weight,
    const ttnn::Tensor& intermediate_bias,
    const ttnn::Tensor& output_weight,
    const ttnn::Tensor& output_bias);

ttnn::Tensor deit_layer(
    const DeiTConfig& config,
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
    const ttnn::Tensor& output_bias);

ttnn::Tensor deit_encoder(
    const DeiTConfig& config,
    const ttnn::Tensor& embeddings,
    const std::vector<std::optional<ttnn::Tensor>>& head_masks,
    const std::unordered_map<std::string, ttnn::Tensor>& parameters);

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> deit(
    const DeiTConfig& config,
    const ttnn::Tensor& pixel_values,
    const std::vector<std::optional<ttnn::Tensor>>& head_masks,
    const ttnn::Tensor& cls_token,
    const ttnn::Tensor& distillation_token,
    const ttnn::Tensor& position_embeddings,
    const std::unordered_map<std::string, ttnn::Tensor>& parameters);

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
    const ttnn::Tensor& classifier_bias);
