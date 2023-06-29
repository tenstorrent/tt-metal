#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"


#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

struct BinaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

operation::ProgramWithCallbacks eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);
operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);

struct EltwiseBinary {
    const BinaryOpType::Enum op_type;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

std::ostream& operator<<(std::ostream& os, const EltwiseBinary& op);

inline Tensor add(const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_with_autoformat(EltwiseBinary{BinaryOpType::ADD}, input_tensor_a, input_tensor_b);
}
inline Tensor sub(const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_with_autoformat(EltwiseBinary{BinaryOpType::SUB}, input_tensor_a, input_tensor_b);
}
inline Tensor mul(const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    return operation::run_with_autoformat(EltwiseBinary{BinaryOpType::MUL}, input_tensor_a, input_tensor_b);
}

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type);

} // namespace eltwise_binary_op_utils
