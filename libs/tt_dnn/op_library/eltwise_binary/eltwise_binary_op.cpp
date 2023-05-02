#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type){
    string op_name, op_code;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_code = "0"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_code = "1"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_code = "2"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
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

Tensor eltwise_binary_(const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type) {
    switch (eltwise_binary_op_utils::get_parallelization_strategy(a, b)){
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(a, b, op_type);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_binary_single_core(a, b, op_type);
    }
}
Tensor eltwise_binary(const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type) {

    Device * device;

    // Get the device
    if (a.on_host() && b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!a.on_host()){
        device = a.device();
    } else {
        device = b.device();
    }
    TT_ASSERT(a.shape() == b.shape() && "Operand to eltwise binary need to be the same size!");

    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape());
    auto b_pad_shape = AutoPad::pad_to_tile_shape(b.shape());
    auto out_shape = a.shape();
    auto no_pad_a = AutoPad::check_input_tensor_format(a, a_pad_shape);
    auto no_pad_b = AutoPad::check_input_tensor_format(b, b_pad_shape);
    if (no_pad_a && no_pad_b) {
        return eltwise_binary_(a, b, op_type);
    } else if (no_pad_a) {
        auto output = eltwise_binary_(a, AutoPad::format_input_tensor(b, device, b_pad_shape, 0), op_type);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else if (no_pad_b) {
        auto output = eltwise_binary_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), b, op_type);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    } else {
        auto output = eltwise_binary_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), AutoPad::format_input_tensor(b, device, b_pad_shape, 0), op_type);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    }

}

}  // namespace tt_metal

}  // namespace tt
