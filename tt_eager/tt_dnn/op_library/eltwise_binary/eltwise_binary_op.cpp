#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {
    std::map<string, string> defines;
    string op_name = "sub_tiles";
    string op_code = "1";
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_code = "0";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_code = "1";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_code = "2";
            break;
        case BinaryOpType::GT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ)); break;
        case BinaryOpType::LT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LTZ)); break;
        case BinaryOpType::GTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GEZ)); break;
        case BinaryOpType::LTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LEZ)); break;
        case BinaryOpType::EQ: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EQZ)); break;
        case BinaryOpType::NE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ)); break;
        case BinaryOpType::SQUARED_DIFFERENCE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::SQUARE)); break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_CODE"] = op_code.c_str();
    if (fused_activations.has_value()) {
        defines.merge(eltwise_unary_op_utils::get_block_defines(fused_activations.value()));
    }
    return defines;
}



}  // namespace eltwise_binary_op_utils

namespace tt {

namespace tt_metal {


void EltwiseBinary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE), "Inputs to eltwise binary must be tilized");
    TT_ASSERT(input_tensor_a.dtype() == input_tensor_b.dtype());
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);
}

std::vector<Shape> EltwiseBinary::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseBinary::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseBinary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default: return eltwise_binary_single_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
    }
}


BinaryOpParallelizationStrategy EltwiseBinary::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    if(num_tiles > 1){
           return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
       return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes EltwiseBinary::attributes() const {
    return {
        {"op_type", this->op_type},
        {"fused_activations", this->fused_activations},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
