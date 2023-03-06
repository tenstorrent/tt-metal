#include "tt_metal/op_library/bcast/bcast_op.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "constants.hpp"

// TODO(AP): duplication
namespace bcast_op_utils {

using namespace tt::tt_metal;
using namespace tt::constants;

// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy) {
		if (bcast_parallelization_strategy == BcastOpParallelizationStrategy::SINGLE_CORE) {
        if (bcast_dim == BcastOpDim::H) {
            return "kernels/dataflow/reader_bcast_h_8bank.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "kernels/dataflow/reader_bcast_w_8bank.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "kernels/dataflow/reader_bcast_hw_8bank.cpp";
        }
    }
    else {
        if (bcast_dim == BcastOpDim::H) {
            return "kernels/dataflow/reader_bcast_h_8bank_input_rows_partitioned.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "kernels/dataflow/reader_bcast_w_8bank_input_cols_partitioned.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "kernels/dataflow/reader_bcast_hw_8bank_partitioned.cpp";
        }
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "kernels/compute/bcast_hw.cpp";
        default:           TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

const char* get_math_to_op_define(BcastOpMath::Enum bcast_math) {
    switch (bcast_math) {
        case BcastOpMath::ADD:  return "add_tiles_bcast";
        case BcastOpMath::SUB:  return "sub_tiles_bcast";
        case BcastOpMath::MUL:  return "mul_tiles_bcast";
        default:           TT_ASSERT(false && "Unexpected bcast_math!");
    }
    return "";
}

void set_compute_kernel_defines(ComputeKernel * bcast_kernel, BcastOpMath::Enum bcast_math){
    bcast_kernel->add_define("BCAST_OP", get_math_to_op_define(bcast_math));
    return;
}

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, BcastOpDim::Enum bcast_dim){
    uint32_t num_tiles = a.volume() / TILE_HW;
    uint32_t Ht = a.shape()[2] / TILE_HEIGHT;
    uint32_t Wt = a.shape()[3] / TILE_WIDTH;

    if(Ht > 1 and bcast_dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and bcast_dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and bcast_dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace bcast_op_utils


using namespace tt::tt_metal;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace tt_metal {


Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
    const auto ashape = a.shape();
    const auto bshape = b.shape();
    u32 N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    u32 bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    u32 NC = N*C;
    u32 HW = H*W;

    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.volume() % TILE_HW == 0);

    TT_ASSERT((bN*bC == 1 || (bN == N && bC == C)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");
    // validate input dimensions
    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(H == bH && bW == TILE_WIDTH);
    if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(W == bW && bH == TILE_HEIGHT);
    if (bcast_dim == BcastOpDim::HW)
        TT_ASSERT(bW == TILE_WIDTH && bH == TILE_HEIGHT);

    switch (bcast_op_utils::get_parallelization_strategy(a, bcast_dim)){
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            return bcast_multi_core_h(a, b, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            return bcast_multi_core_w(a, b, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            return bcast_multi_core_hw(a, b, bcast_math, bcast_dim);
            break;
        case BcastOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bcast_single_core(a, b, bcast_math, bcast_dim);
    }
}

}  // namespace tt_metal

}  // namespace tt
