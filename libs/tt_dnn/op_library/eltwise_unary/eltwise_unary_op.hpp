#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH }; }
};

struct UnaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

Program eltwise_unary_single_core (const Tensor &input_tensor, Tensor &output_tensor, UnaryOpType::Enum op_type);
Program eltwise_unary_multi_core (const Tensor &input_tensor, Tensor &output_tensor, UnaryOpType::Enum op_type);

struct EltwiseUnary : Operation {
    const UnaryOpType::Enum op_type;

    EltwiseUnary(UnaryOpType::Enum op_type) : op_type{op_type} {}

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const override;
    Program create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

inline Tensor sqrt(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::SQRT).run({input_tensor}).at(0); }
inline Tensor exp(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::EXP).run({input_tensor}).at(0); }
inline Tensor recip(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::RECIP).run({input_tensor}).at(0); }
inline Tensor gelu(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::GELU).run({input_tensor}).at(0); }
inline Tensor relu(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::RELU).run({input_tensor}).at(0); }
inline Tensor sigmoid(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::SIGMOID).run({input_tensor}).at(0); }
inline Tensor log(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::LOG).run({input_tensor}).at(0); }
inline Tensor tanh(const Tensor &input_tensor) { return EltwiseUnary(UnaryOpType::TANH).run({input_tensor}).at(0); }


}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

string get_op_name(UnaryOpType::Enum op_type);

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type);

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &input_tensor);

} // namespace eltwise_unary_op_utils
