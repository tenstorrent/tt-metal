#include "tt_metal/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

string get_op_name(UnaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case UnaryOpType::EXP: op_name = "hlk_sfpu_exponential(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RECIP: op_name = "hlk_sfpu_reciprocal(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::GELU: op_name = "hlk_sfpu_gelu(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RELU: op_name = "hlk_pack_relu_tile_to_stream(nullptr, 0, CB::c_out0);"; break;
        case UnaryOpType::SQRT: op_name = "hlk_sfpu_sqrt(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::SIGMOID: op_name = "hlk_sfpu_sigmoid(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG: op_name = "hlk_sfpu_log(nullptr, 0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::TANH: op_name = "hlk_sfpu_tanh(nullptr, 0); pack_tile(0, CB::c_out0);"; break;

        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

void set_compute_kernel_defines(tt_metal::ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type){
    string op_name = get_op_name(op_type);
    eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", op_name);
    bool is_relu = (op_type == UnaryOpType::RELU);
    eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "hlk_relu_config(nullptr, 1);" : "");
    eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "hlk_relu_config(nullptr, 0);" : "");
    return;
}

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a){
    int32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return UnaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type) {

    switch (get_parallelization_strategy(a)){
        case UnaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_unary_multi_core(a, op_type);
            break;
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_unary_single_core(a, op_type);
    }

}

}  // namespace tt_metal

}  // namespace tt
