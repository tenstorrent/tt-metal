// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include "common/constants.hpp"

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"



namespace tt {

namespace tt_metal {

std::vector<Tensor> addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_add_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> addcdiv_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> add_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> exp_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> sqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> binary_assign_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> min_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> embedding_bw(const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// bw = grad(1 - tanh(x) ** 2)
std::vector<Tensor> tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> sigmoid_bw(const Tensor& grad, const Tensor& esinput, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> tan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> fill_zero_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> sub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_sub_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> rsub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> binary_le_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> rsqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> neg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> relu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> lt_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> gt_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> ne_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> clamp_bw(const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> clamp_min_bw(const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> clamp_max_bw(const Tensor& grad, const Tensor& input, float max, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> atan2_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> hypot_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> exp2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> expm1_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> gelu_bw(const Tensor& grad, const Tensor& input, string approximate, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> bias_gelu_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, string approximate, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> bias_gelu_unary_bw(const Tensor& grad, const Tensor& input, float bias, string approximate, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> squared_difference_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight, weight is float
std::vector<Tensor> lerp_bw(const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight, weight is tensor
std::vector<Tensor> lerp_bw(const Tensor& grad, const Tensor& input, const Tensor& end, const Tensor& weight, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> ldexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> xlogy_bw(const Tensor& grad, const Tensor& input,  const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> logaddexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

} //namespace tt_metal

} //namespace tt
