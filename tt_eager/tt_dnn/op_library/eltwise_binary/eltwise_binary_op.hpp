/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

namespace tt {

namespace tt_metal {

enum class BinaryOpType {
    ADD = 0, SUB = 1, MUL = 2, GT = 3, LT = 4, LTE = 5, GTE = 6, EQ = 7, NE = 8, SQUARED_DIFFERENCE = 9
};

enum class BinaryOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

operation::ProgramWithCallbacks eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations);
operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations);

struct EltwiseBinary {
    const BinaryOpType op_type;
    const std::optional<std::vector<UnaryWithParam>> fused_activations;
    const MemoryConfig output_mem_config;

    BinaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

template <BinaryOpType binary_op_type>
struct make_eltwise_binary {
     Tensor operator()(const Tensor& input_tensor_a, const Tensor& input_tensor_b, std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
         TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
         return operation::run_with_autoformat(EltwiseBinary{binary_op_type, fused_activations, output_mem_config}, {input_tensor_a, input_tensor_b}).at(0);
     }
 };

inline Tensor add_without_autoformat(const Tensor& input_tensor_a, const Tensor& input_tensor_b, std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_without_autoformat(EltwiseBinary{BinaryOpType::ADD, fused_activations, output_mem_config}, {input_tensor_a, input_tensor_b}).at(0);
}

 // arithmetic binary ops
 constexpr auto add = make_eltwise_binary<BinaryOpType::ADD>{};
 constexpr auto sub = make_eltwise_binary<BinaryOpType::SUB>{};
 constexpr auto mul = make_eltwise_binary<BinaryOpType::MUL>{};
 constexpr auto squared_difference = make_eltwise_binary<BinaryOpType::SQUARED_DIFFERENCE>{};

 // comparative binary ops
 constexpr auto lt = make_eltwise_binary<BinaryOpType::LT>{};
 constexpr auto gt = make_eltwise_binary<BinaryOpType::GT>{};
 constexpr auto lte = make_eltwise_binary<BinaryOpType::LTE>{};
 constexpr auto gte = make_eltwise_binary<BinaryOpType::GTE>{};
 constexpr auto eq = make_eltwise_binary<BinaryOpType::EQ>{};
 constexpr auto ne = make_eltwise_binary<BinaryOpType::NE>{};

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(BinaryOpType op_typee, const std::optional<std::vector<UnaryWithParam>> fused_activations);

}  // namespace eltwise_binary_op_utils
