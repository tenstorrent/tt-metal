#include "tt_metal/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;

namespace bmm_op_utils {
using namespace tt::tt_metal;

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b){
    const auto& ashape = a.shape(), bshape = b.shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW; // Output M x N
    if (num_output_tiles > 1) {
        return BmmOpParallelizationStrategy::MULTI_CORE;
    }else {
        return BmmOpParallelizationStrategy::SINGLE_CORE;
    }
}

}

namespace tt {

namespace tt_metal {


Tensor matmul(const Tensor& a, const Tensor& b) {
    switch (bmm_op_utils::get_parallelization_strategy(a, b)){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return matmul_multi_core(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return matmul_single_core(a, b);
    }
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    switch (bmm_op_utils::get_parallelization_strategy(a, b)){
        case BmmOpParallelizationStrategy::MULTI_CORE:
            return bmm_multi_core(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bmm_single_core(a, b);
    }
}

}  // namespace tt_metal

}  // namespace tt
