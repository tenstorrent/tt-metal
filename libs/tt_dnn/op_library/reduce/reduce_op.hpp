#pragma once

#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

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
Tensor reduce(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
Tensor reduce_single_core(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
Tensor reduce_multi_core_h(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
Tensor reduce_multi_core_w(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);
Tensor reduce_multi_core_hw(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);

}  // namespace tt_metal

}  // namespace tt

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim::Enum reduce_dim, ReduceOpMath::Enum reduce_op);

void add_defines(ComputeKernel * reduce_kernel, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim);

ReduceOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, ReduceOpDim::Enum reduce_dim);

} // namespace reduce_op_utils
