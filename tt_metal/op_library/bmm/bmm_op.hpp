#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct BmmOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

Tensor matmul (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B
Tensor matmul_single_core  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm_single_core     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B
Tensor matmul_multi_core  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm_multi_core     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B

}  // namespace tt_metal

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

}
