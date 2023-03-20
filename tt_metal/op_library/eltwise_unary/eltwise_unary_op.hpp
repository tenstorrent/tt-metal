#pragma once

#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

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

Tensor eltwise_unary (const Tensor &a, UnaryOpType::Enum op_type);
Tensor eltwise_unary_single_core (const Tensor &a, UnaryOpType::Enum op_type);
Tensor eltwise_unary_multi_core (const Tensor &a, UnaryOpType::Enum op_type);

inline Tensor exp     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::EXP); }
inline Tensor recip   (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RECIP); }
inline Tensor gelu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::GELU); }
inline Tensor relu    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::RELU); }
inline Tensor sqrt    (const Tensor &a) { return eltwise_unary(a, UnaryOpType::SQRT); }
inline Tensor sigmoid (const Tensor &a) { return eltwise_unary(a, UnaryOpType::SIGMOID); }
inline Tensor log     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::LOG); }
inline Tensor tanh     (const Tensor &a) { return eltwise_unary(a, UnaryOpType::TANH); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

string get_op_name(UnaryOpType::Enum op_type);

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type);

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a);

} // namespace eltwise_unary_op_utils
