#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH }; }
};

// TODO: Accept parallelization

Tensor eltwise_unary (const Tensor &a, UnaryOpType::Enum op_type);

inline Tensor exp     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::EXP); }
inline Tensor recip   (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RECIP); }
inline Tensor gelu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::GELU); }
inline Tensor relu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RELU); }
inline Tensor sqrt    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::SQRT); }
inline Tensor sigmoid (const Tensor &a) { return eltwise_unary(a, UnaryOpType::SIGMOID); }
inline Tensor log     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::LOG); }
inline Tensor tanh     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::TANH); }

}  // namespace ll_buda

}  // namespace tt
