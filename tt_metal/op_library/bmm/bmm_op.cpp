#include "tt_metal/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;

namespace bmm_op_utils {
using namespace tt::tt_metal;

BmmOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b){
    const auto& ashape = a.shape(), bshape = b.shape();
    uint32_t num_output_tiles = ashape[0] * ashape[1] * ashape[2] * bshape[3] / TILE_HW; // Output M x N

    // Parameters for large matmul with reuse
    uint32_t B = ashape[0] * ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t in0_block_w = 2;
    uint32_t per_core_M = 16;
    uint32_t per_core_N = 16;

    tt::tt_metal::Device *device = a.device();
    auto logical_grid_size = device->logical_grid_size();
    uint32_t num_cores_x = logical_grid_size.x;
    uint32_t num_cores_y = logical_grid_size.y;
    uint32_t num_blocks_total = (Mt / per_core_M) * (Nt / per_core_N);

    if (
        B == 1 and
        Mt % per_core_M == 0 and
        Nt % per_core_N == 0 and
        Kt % in0_block_w == 0 and
        num_blocks_total <= num_cores_x * num_cores_y
    ) {
        return BmmOpParallelizationStrategy::MULTI_CORE_REUSE;
    }
    else if (num_output_tiles > 1) {
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
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return matmul_multi_core_reuse(a, b);
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
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE:
            return bmm_multi_core_reuse(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bmm_single_core(a, b);
    }
}

}  // namespace tt_metal

}  // namespace tt
