#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"


#include "tt_dnn/op_library/operation.hpp"

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

Program eltwise_binary_single_core (const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);
Program eltwise_binary_multi_core (const Tensor &a, const Tensor &b, Tensor &output_tensor, BinaryOpType::Enum op_type);

struct EltwiseBinary : Operation {
    const BinaryOpType::Enum op_type;

    EltwiseBinary(BinaryOpType::Enum op_type) : op_type{op_type} {}

    EltwiseBinary(const EltwiseBinary&) = delete;
    EltwiseBinary& operator=(const EltwiseBinary&) = delete;
    ~EltwiseBinary() {}

    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

Tensor eltwise_binary(const EltwiseBinary& op, const Tensor &a, const Tensor &b);
inline Tensor add(const Tensor &a, const Tensor &b) { return eltwise_binary(EltwiseBinary(BinaryOpType::ADD), a, b); }
inline Tensor sub(const Tensor &a, const Tensor &b) { return eltwise_binary(EltwiseBinary(BinaryOpType::SUB), a, b); }
inline Tensor mul(const Tensor &a, const Tensor &b) { return eltwise_binary(EltwiseBinary(BinaryOpType::MUL), a, b); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type);

BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

} // namespace eltwise_binary_op_utils
