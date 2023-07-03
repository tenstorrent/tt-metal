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

void add_defines(ComputeKernel* eltwise_binary_kernel, BinaryOpType::Enum op_type) {
    string op_name = "sub_tiles";
    string op_code = "1";
    string compare = "1";
    string compare_init = "";
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_code = "0";
            compare = "0";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_code = "1";
            compare = "0";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_code = "2";
            compare = "0";
            break;
        case BinaryOpType::GT: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::GTZ); break;
        case BinaryOpType::LT: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::LTZ); break;
        case BinaryOpType::GTE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::GEZ); break;
        case BinaryOpType::LTE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::LEZ); break;
        case BinaryOpType::EQ: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::EQZ); break;
        case BinaryOpType::NE: compare_init = eltwise_unary_op_utils::get_op_name(UnaryOpType::NEZ); break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
    if ( compare == "1" ) {
      eltwise_binary_kernel->add_define("ELTWISE_COMPARE_BINARY_OP", compare);
      eltwise_binary_kernel->add_define("SFPU_OP_AND_PACK", compare_init);
    }
}



}  // namespace eltwise_binary_op_utils

namespace tt {

namespace tt_metal {


void EltwiseBinary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
}

std::vector<Shape> EltwiseBinary::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseBinary::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

operation::ProgramWithCallbacks EltwiseBinary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    switch (get_parallelization_strategy(input_tensor_a, input_tensor_b)) {
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default: return eltwise_binary_single_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type);
    }
}

operation::Hash EltwiseBinary::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    return fmt::format(
        "eltwise_binary_{}_{}_{}",
         magic_enum::enum_name(this->op_type),
         operation::hash_tensor(input_tensor_a),
         operation::hash_tensor(input_tensor_b)
    );
}


BinaryOpParallelizationStrategy::Enum EltwiseBinary::get_parallelization_strategy(const Tensor& input_tensor_a,const Tensor& input_tensor_b) const {

    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    if(num_tiles > 1){
           return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
       return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

std::ostream& operator<<(std::ostream& os, const EltwiseBinary& op) {
    os << boost::core::demangle(typeid(op).name());
    os << "{";
    os << ".op_type=" << magic_enum::enum_name(op.op_type);
    os << "}";
    return os;
}

// eltwise binary arith ops
eltwise_binop_t add = MakeEltwiseBinary<BinaryOpType::ADD>::call;
eltwise_binop_t sub = MakeEltwiseBinary<BinaryOpType::SUB>::call;
eltwise_binop_t mul = MakeEltwiseBinary<BinaryOpType::MUL>::call;

// eltwise comparative binary ops
eltwise_binop_t lt = MakeEltwiseBinary<BinaryOpType::LT>::call;
eltwise_binop_t gt = MakeEltwiseBinary<BinaryOpType::GT>::call;
eltwise_binop_t lte = MakeEltwiseBinary<BinaryOpType::LTE>::call;
eltwise_binop_t gte = MakeEltwiseBinary<BinaryOpType::GTE>::call;
eltwise_binop_t eq = MakeEltwiseBinary<BinaryOpType::EQ>::call;
eltwise_binop_t ne = MakeEltwiseBinary<BinaryOpType::NE>::call;

}  // namespace tt_metal

}  // namespace tt
