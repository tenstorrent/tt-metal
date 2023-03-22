#include "tt_metal/op_library/reduce/reduce_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim::Enum reduce_dim, ReduceOpMath::Enum reduce_op){
    string kernel_name;
    switch(reduce_dim){
        case ReduceOpDim::H: kernel_name = "tt_metal/kernels/compute/reduce_h.cpp"; break;
        case ReduceOpDim::W: kernel_name = "tt_metal/kernels/compute/reduce_w.cpp"; break;
        case ReduceOpDim::HW: kernel_name = "tt_metal/kernels/compute/reduce_hw.cpp"; break;
        default: TT_ASSERT(false && "Undefined dim");
    }
    return kernel_name;
}

void add_defines(ComputeKernel * reduce_kernel, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim){
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == ReduceOpMath::MAX;
    reduce_kernel->add_define("REDUCE_OP", do_max ? "PoolType::MAX" : "PoolType::SUM");
    switch(reduce_dim) {
        case ReduceOpDim::W: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_ROW"); break;
        case ReduceOpDim::H: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_COL"); break;
        case ReduceOpDim::HW: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"); break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    return;
}

ReduceOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, ReduceOpDim::Enum reduce_dim){
    uint32_t num_tiles = a.volume() / TILE_HW;
    auto shape = a.shape();
    uint32_t Wt = shape[3]/TILE_WIDTH;
    uint32_t Ht = shape[2]/TILE_HEIGHT;
    uint32_t NC = shape[1]*shape[0];
    if(NC * Wt > 1 and reduce_dim == ReduceOpDim::H){
        return ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }else if(NC * Ht > 1 and reduce_dim == ReduceOpDim::W){
        return ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }else if(num_tiles > 1 and reduce_dim == ReduceOpDim::HW){
        return ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    }else{
        return ReduceOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace reduce_op_utils
namespace tt {
namespace tt_metal {

Tensor reduce (const Tensor &a, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {

    switch (reduce_op_utils::get_parallelization_strategy(a, reduce_dim)){
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return reduce_multi_core_h(a, reduce_op, reduce_dim, scaler);
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return reduce_multi_core_w(a, reduce_op, reduce_dim, scaler);
        case ReduceOpParallelizationStrategy::MULTI_CORE_HW:
            return reduce_multi_core_hw(a, reduce_op, reduce_dim, scaler);
        case ReduceOpParallelizationStrategy::SINGLE_CORE:
        default:
            return reduce_single_core(a, reduce_op, reduce_dim, scaler);
    }

}

}  // namespace tt_metal

}  // namespace tt
