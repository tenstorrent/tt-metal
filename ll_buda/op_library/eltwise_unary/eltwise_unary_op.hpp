#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT }; }
};

// TODO: Accept parallelization

Tensor eltwise_unary (const Tensor &a, UnaryOpType::Enum op_type);

inline Tensor exp     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::EXP); }
inline Tensor recip   (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RECIP); }
inline Tensor gelu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::GELU); }
inline Tensor relu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RELU); }
inline Tensor sqrt    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::SQRT); }

}  // namespace ll_buda

}  // namespace tt
