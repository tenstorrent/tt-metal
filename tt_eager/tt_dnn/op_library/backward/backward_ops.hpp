// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "tensor/host_buffer/functions.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> unary_mul_bw(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<std::optional<Tensor>> unary_pow_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    float exponent,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> unary_pow_bw(
    const Tensor& grad,
    const Tensor& input,
    float exponent,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> exp_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> exp_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> sqrt_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> sqrt_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<Tensor> unary_div_bw(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    string round_mode,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> rdiv_bw(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    string round_mode,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// bw = grad(1 - tanh(x) ** 2)
std::vector<std::optional<Tensor>> tanh_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> tanh_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true},
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<Tensor> binary_le_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_abs_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> lt_bw(
    const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> gt_bw(
    const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> ne_bw(
    const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> exp2_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> expm1_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> gelu_bw(
    const Tensor& grad,
    const Tensor& input,
    string approximate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> bias_gelu_unary_bw(
    const Tensor& grad,
    const Tensor& input,
    float bias,
    string approximate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> softplus_bw(
    const Tensor& grad,
    const Tensor& input,
    float beta = 1.0,
    float threshold = 20.0,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> polygamma_bw(
    const Tensor& grad,
    const Tensor& input,
    int n,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> erfinv_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> hardtanh_bw(
    const Tensor& grad,
    const Tensor& input,
    float min,
    float max,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> angle_bw(
    const Tensor& grad,
    const Tensor& input,
    bool is_complextensor = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> erf_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> digamma_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> deg2rad_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> reciprocal_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> prod_bw(
    const Tensor& grad,
    const Tensor& input,
    bool all_dimensions,
    int64_t dim,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> threshold_bw(
    const Tensor& grad,
    const Tensor& input,
    float threshold,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> ge_bw(
    const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> le_bw(
    const Tensor& grad, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


std::vector<Tensor> conj_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_recip_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> imag_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> real_bw(
    const Tensor& grad,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_mul_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_div_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> polar_bw(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_add_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha = 1.0,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> complex_sub_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha = 1.0,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<Tensor> repeat_bw(
    const Tensor& grad, const Tensor& input, const Shape& shape, const MemoryConfig& output_mem_config);

Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);
}  // namespace tt_metal

}  // namespace tt
