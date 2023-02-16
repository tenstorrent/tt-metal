#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {
struct ReduceOpMath {
    enum Enum { SUM = 0, MAX = 1 };
    static const vector<Enum> all() { return { SUM, MAX }; }
};

struct ReduceOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const vector<Enum> all() { return { H, W, HW }; }
};

// TODO: Accept parallelization
Tensor reduce(const Tensor &a, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler = 1.0f);

}  // namespace ll_buda

}  // namespace tt
