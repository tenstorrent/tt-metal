#include "tt_metal/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;

namespace bmm_op_utils {
using namespace tt::tt_metal;

tt_xy_pair get_core_range(uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    tt_xy_pair core_range(0, 0);
    if (!(num_blocks_rows == 1 && num_blocks_cols == 1) && num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

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
    tt_xy_pair core_range = get_core_range((Mt / per_core_M), (Nt / per_core_N), num_cores_y, num_cores_x);

    if (
        Mt % per_core_M == 0 and
        Nt % per_core_N == 0 and
        Kt % in0_block_w == 0 and
        num_blocks_total <= num_cores_x * num_cores_y
    ) {
        if (core_range.y > 0) {
            return BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST;
        }
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
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return matmul_multi_core_reuse_mcast(a, b);
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
        case BmmOpParallelizationStrategy::MULTI_CORE_REUSE_MCAST:
            return bmm_multi_core_reuse_mcast(a, b);
            break;
        case BmmOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bmm_single_core(a, b);
    }
}

Tensor large_bmm(const Tensor& a, const Tensor& b, bool tilize_a, bool untilize_out) {
    // TT_ASSERT(
    //     bmm_op_utils::get_parallelization_strategy(a, b) == BmmOpParallelizationStrategy::SINGLE_CORE,
    //     "Only single core large_bmm supported so far");
    if (bmm_op_utils::get_parallelization_strategy(a, b) != BmmOpParallelizationStrategy::SINGLE_CORE) {
        std::cout << "WARNING: Only single core mode supported for large_bmm. Falling back to single core." << std::endl;
    }
    return large_bmm_single_core(a, b, tilize_a, untilize_out);
}

}  // namespace tt_metal

}  // namespace tt
