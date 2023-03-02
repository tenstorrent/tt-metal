#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

string get_op_name(BinaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

void set_compute_kernel_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type){
    string op_name = get_op_name(op_type);
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    return;
}

BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b){
    uint32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // eltwise_binary_op_utils

namespace tt {

namespace tt_metal {

Tensor eltwise_binary(const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type) {

    switch (eltwise_binary_op_utils::get_parallelization_strategy(a, b)){
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(a, b, op_type);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_binary_single_core(a, b, op_type);
    }

}

}  // namespace tt_metal

}  // namespace tt
