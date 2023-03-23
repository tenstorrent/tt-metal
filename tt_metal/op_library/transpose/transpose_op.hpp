#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

struct TransposeOpDim {
    enum Enum { WH = 0, HC = 1 };
    static const vector<Enum> all() { return { WH, HC }; }
};

struct TransposeOpParallelizationStrategy {
    enum Enum { MULTI_CORE_WH = 0, MULTI_CORE_HC = 1, SINGLE_CORE = 2 };
    static const vector<Enum> all() { return { MULTI_CORE_WH, MULTI_CORE_HC, SINGLE_CORE }; }
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim=TransposeOpDim::WH);
inline Tensor transpose(const Tensor &a) { return transpose_(a, TransposeOpDim::WH); }
inline Tensor transpose_wh(const Tensor &a) { return transpose_(a, TransposeOpDim::WH); }
inline Tensor transpose_hc(const Tensor &a) { return transpose_(a, TransposeOpDim::HC); }

Tensor transpose_single_core(const Tensor &a, TransposeOpDim::Enum transpose_dim);
Tensor transpose_wh_multi_core(const Tensor &a);
Tensor transpose_hc_multi_core(const Tensor &a);

}  // namespace tt_metal

}  // namespace tt

namespace transpose_op_utils {

using namespace tt::tt_metal;

TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, TransposeOpDim::Enum transpose_dim);

} // namespace transpose_op_utils
