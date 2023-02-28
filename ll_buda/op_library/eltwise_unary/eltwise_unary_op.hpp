#pragma once

#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

namespace tt {

namespace ll_buda {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH }; }
};

string get_op_name(UnaryOpType::Enum op_type);
void set_compute_kernel_defines(ll_buda::ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type);

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

}  // namespace ll_buda

}  // namespace tt

namespace eltwise_unary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};

} // namespace eltwise_unary
