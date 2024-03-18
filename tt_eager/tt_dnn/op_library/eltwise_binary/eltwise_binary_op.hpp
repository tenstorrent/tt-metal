// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class BinaryOpType {

    ADD = 0,
    SUB = 1,
    MUL = 2,
    GT = 3,
    LT = 4,
    LTE = 5,
    GTE = 6,
    EQ = 7,
    NE = 8,
    SQUARED_DIFFERENCE = 9,
    BIAS_GELU = 10,
    LOGADDEXP = 11,
    LOGICAL_AND = 12,
    LOGICAL_OR = 13,
    LDEXP = 14,
    LOGADDEXP2 = 15
};

enum class BinaryOpParallelizationStrategy { MULTI_CORE = 0, SINGLE_CORE = 1 };

operation::ProgramWithCallbacks eltwise_binary_single_core(
    const Tensor &a,
    const Tensor &b,
    const Tensor &output_tensor,
    BinaryOpType op_type,
    const std::optional<std::vector<UnaryWithParam>> fused_activations);
operation::ProgramWithCallbacks eltwise_binary_multi_core(
    const Tensor &a,
    const Tensor &b,
    const Tensor &output_tensor,
    BinaryOpType op_type,
    const std::optional<std::vector<UnaryWithParam>> fused_activations);

struct EltwiseBinary {
    const BinaryOpType op_type;
    const std::optional<std::vector<UnaryWithParam>> fused_activations;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const bool in_place;

    BinaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const {
        // GS specific parameters
        // 80 B/cycle unpacker BW shared
        // 128 datums per cycle math, but unpacker cant keep up
        constexpr int num_cores = 9 * 12;

        int total_bytes = 0;
        for (const auto &t : input_tensors) {
            total_bytes += t.volume() * t.element_size();
        }
        int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

        operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "Eltwise PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
        return result;
    }
    static constexpr auto attribute_names =
        std::make_tuple("op_type", "fused_activations", "output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->op_type),
            std::cref(this->fused_activations),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype),
            std::cref(this->in_place));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

template <BinaryOpType binary_op_type>
struct make_eltwise_binary {
    Tensor operator()(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt,
        const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::optional<const DataType> output_dtype = std::nullopt) const {
        TT_FATAL(
            input_tensor_a.get_legacy_shape() == input_tensor_b.get_legacy_shape(), "Input shapes must be the same!");
        return operation::run_with_autoformat(
                   EltwiseBinary{
                       binary_op_type,
                       fused_activations,
                       output_mem_config,
                       output_dtype.value_or(input_tensor_a.get_dtype()),
                       false},
                   {input_tensor_a, input_tensor_b})
            .at(0);
    }
};

// arithmetic binary ops
constexpr auto add = make_eltwise_binary<BinaryOpType::ADD>{};
constexpr auto sub = make_eltwise_binary<BinaryOpType::SUB>{};
constexpr auto mul = make_eltwise_binary<BinaryOpType::MUL>{};
constexpr auto squared_difference = make_eltwise_binary<BinaryOpType::SQUARED_DIFFERENCE>{};
constexpr auto bias_gelu = make_eltwise_binary<BinaryOpType::BIAS_GELU>{};
constexpr auto logaddexp = make_eltwise_binary<BinaryOpType::LOGADDEXP>{};
constexpr auto ldexp = make_eltwise_binary<BinaryOpType::LDEXP>{};
constexpr auto logaddexp2 = make_eltwise_binary<BinaryOpType::LOGADDEXP2>{};

// comparative binary ops
constexpr auto lt = make_eltwise_binary<BinaryOpType::LT>{};
constexpr auto gt = make_eltwise_binary<BinaryOpType::GT>{};
constexpr auto lte = make_eltwise_binary<BinaryOpType::LTE>{};
constexpr auto gte = make_eltwise_binary<BinaryOpType::GTE>{};
constexpr auto eq = make_eltwise_binary<BinaryOpType::EQ>{};
constexpr auto ne = make_eltwise_binary<BinaryOpType::NE>{};

// logical ops
constexpr auto logical_and = make_eltwise_binary<BinaryOpType::LOGICAL_AND>{};
constexpr auto logical_or = make_eltwise_binary<BinaryOpType::LOGICAL_OR>{};
}  // namespace tt_metal

namespace operations {

namespace primary {

// TODO: in_place should not take output args
inline Tensor add(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    bool in_place = false) {
    TT_FATAL(input_tensor_a.get_legacy_shape() == input_tensor_b.get_legacy_shape(), "Input shapes must be the same!");
    auto output = operation::run(
        EltwiseBinary{
            BinaryOpType::ADD,
            fused_activations,
            output_mem_config,
            output_dtype.value_or(input_tensor_a.get_dtype()),
            in_place},
        {input_tensor_a, input_tensor_b});
    if (in_place) {
        return input_tensor_a;
    } else {
        return output.at(0);
    }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(
    BinaryOpType op_typee, const std::optional<std::vector<UnaryWithParam>> fused_activations);

}  // namespace eltwise_binary_op_utils
