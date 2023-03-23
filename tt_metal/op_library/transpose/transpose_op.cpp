#include "tt_metal/op_library/transpose/transpose_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace transpose_op_utils {

using namespace tt::tt_metal;

TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, TransposeOpDim::Enum transpose_dim){
    auto ashape = a.shape();
    uint32_t num_tiles = a.volume() / TILE_HW;
    if (transpose_dim == TransposeOpDim::WH && num_tiles > 1) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (transpose_dim == TransposeOpDim::HC && num_tiles > 1) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else {
        return TransposeOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace transpose_op_utils

namespace tt {

namespace tt_metal {

Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim) {
    switch (transpose_op_utils::get_parallelization_strategy(a, transpose_dim)){
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            return transpose_wh_multi_core(a);
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            return transpose_hc_multi_core(a);
            break;
        case TransposeOpParallelizationStrategy::SINGLE_CORE:
        default:
            return transpose_single_core(a, transpose_dim);
    }
}

}  // namespace tt_metal

}  // namespace tt
