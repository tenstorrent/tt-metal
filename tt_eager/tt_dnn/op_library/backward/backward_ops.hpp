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

std::vector<Tensor> exp_bw(const Tensor& grad, const Tensor& exp_result, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> sqrt_bw(const Tensor& grad, const Tensor& sqrt_result, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// bw = grad(1 - tanh(x) ** 2)
std::vector<Tensor> tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> tan_bw(const Tensor& grad, const Tensor& tan_result, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> fill_zero_bw(const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

} //namespace tt_metal

} //namespace tt
