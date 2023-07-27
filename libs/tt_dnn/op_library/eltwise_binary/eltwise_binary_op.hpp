#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2, GT = 3, LT = 4, LTE = 5, GTE = 6, EQ = 7, NE = 8 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

struct BinaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

operation::ProgramWithCallbacks eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);
operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);

struct EltwiseBinary {
    const BinaryOpType::Enum op_type;

    BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;

    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

template <BinaryOpType::Enum binary_op_type>
Tensor run_eltwise_binary(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_with_autoformat(EltwiseBinary{binary_op_type}, {input_tensor_a, input_tensor_b}).at(0);
};

inline Tensor add_without_autoformat(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_without_autoformat(EltwiseBinary{BinaryOpType::ADD}, {input_tensor_a, input_tensor_b}).at(0);
}

// arithmetic binary ops
constexpr auto add = run_eltwise_binary<BinaryOpType::ADD>;
constexpr auto sub = run_eltwise_binary<BinaryOpType::SUB>;
constexpr auto mul = run_eltwise_binary<BinaryOpType::MUL>;

// comparative binary ops
constexpr auto lt = run_eltwise_binary<BinaryOpType::LT>;
constexpr auto gt = run_eltwise_binary<BinaryOpType::GT>;
constexpr auto lte = run_eltwise_binary<BinaryOpType::LTE>;
constexpr auto gte = run_eltwise_binary<BinaryOpType::GTE>;
constexpr auto eq = run_eltwise_binary<BinaryOpType::EQ>;
constexpr auto ne = run_eltwise_binary<BinaryOpType::NE>;

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel *eltwise_binary_kernel, BinaryOpType::Enum op_type);

}  // namespace eltwise_binary_op_utils
