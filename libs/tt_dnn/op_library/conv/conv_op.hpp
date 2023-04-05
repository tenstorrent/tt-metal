#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct ConvOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, MULTI_CORE_REUSE = 1, MULTI_CORE_REUSE_MCAST = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE }; }
};

Tensor conv_as_large_bmm_single_core(const Tensor& A, const Tensor& B); // Tilizes a, untilizes b
Tensor conv_as_large_bmm_single_core_single_block(const Tensor& A, const Tensor& B, bool untilize_out, bool use_single_bank_reader); // Allows support for tilizing a, untilize b

}  // namespace tt_metal

}  // namespace tt
