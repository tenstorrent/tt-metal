#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
struct BmmOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, MULTI_CORE_REUSE = 1, MULTI_CORE_REUSE_MCAST = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE }; }
};

Tensor matmul (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B
Tensor large_bmm(const Tensor& A, const Tensor& B, bool tilize_a, bool untilize_out); // Allows support for tilizing a, untilize b
Tensor matmul_single_core  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm_single_core     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B
Tensor large_bmm_single_core(const Tensor& A, const Tensor& B, bool tilize_a, bool untilize_out); // Allows support for tilizing a, untilize b
Tensor matmul_multi_core  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm_multi_core     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B
Tensor matmul_multi_core_reuse  (const Tensor &A, const Tensor &B); // Only supports 2D matmul expects N=1 for now
Tensor bmm_multi_core_reuse  (const Tensor &A, const Tensor &B); // Only supports 2D matmul expects N=1 for now
Tensor matmul_multi_core_reuse_mcast  (const Tensor &A, const Tensor &B); // Only supports 2D matmul expects N=1 for now
Tensor bmm_multi_core_reuse_mcast  (const Tensor &A, const Tensor &B); // Only supports 2D matmul expects N=1 for now

}  // namespace tt_metal

}  // namespace tt

namespace bmm_op_utils {
using namespace tt::tt_metal;

tt_xy_pair get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

}
