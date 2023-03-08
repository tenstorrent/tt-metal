#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

struct BcastOpMath {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

struct BcastOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const vector<Enum> all() { return { H, W, HW }; }
};

// TODO: Accept parallelization
Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);

// TODO(AP): make 9 of these?
inline Tensor bcast_add_h(const Tensor &a, const Tensor &b) { return bcast(a, b, BcastOpMath::ADD, BcastOpDim::H); }

}  // namespace tt_metal

}  // namespace tt
