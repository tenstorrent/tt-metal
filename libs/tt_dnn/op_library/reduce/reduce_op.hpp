#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {
struct ReduceOpMath {
    enum Enum { SUM = 0, MAX = 1 };
    static const vector<Enum> all() { return { SUM, MAX }; }
};

struct ReduceOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const vector<Enum> all() { return { H, W, HW }; }
};

struct ReduceOpParallelizationStrategy {
    enum Enum { MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE }; }
};

// TODO: Accept parallelization
operation::ProgramWithCallbacks reduce_single_core(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_h(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_w(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);

struct Reduce {
    const ReduceOpMath::Enum math_op;
    const ReduceOpDim::Enum dim;
    const float scaler;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    ReduceOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

};

Tensor reduce(const Tensor &input_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);

}  // namespace tt_metal

}  // namespace tt

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim::Enum reduce_dim, ReduceOpMath::Enum reduce_op);

void add_defines(ComputeKernel * reduce_kernel, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim);

} // namespace reduce_op_utils
