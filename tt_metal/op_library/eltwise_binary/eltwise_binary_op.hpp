#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

// TODO: Accept parallelization

Tensor eltwise_binary (const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type);

inline Tensor add     (const Tensor &a, const Tensor &b) { return eltwise_binary(a, b, BinaryOpType::ADD); }
inline Tensor sub     (const Tensor &a, const Tensor &b) { return eltwise_binary(a, b, BinaryOpType::SUB); }
inline Tensor mul     (const Tensor &a, const Tensor &b) { return eltwise_binary(a, b, BinaryOpType::MUL); }

}  // namespace tt_metal

}  // namespace tt
