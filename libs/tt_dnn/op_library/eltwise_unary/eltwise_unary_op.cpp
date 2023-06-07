#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

template <typename T>
bool is_parameterized_type(T val) {
  return val == UnaryOpType::RELU_MAX || val == UnaryOpType::RELU_MIN || val == UnaryOpType::POWER;
}


/**
SFPU ops in BRISC

IMPL DONE
  tanh,
  gelu,
  exponential,
  sigmoid,
  reciprocal,
  sqrt,
  log,
  log_with_base,
  sine,
  cosine,
  relu_min,
  relu_max,

  abs,
  sign,
  square,

  equal_zero,
  not_equal_zero,
  less_than_zero,
  greater_than_equal_zero,
  less_than_equal_zero,
  greater_than_zero,
  power,

WIP
  hardtanh,
  exp_with_base,
  lrelu,
  tanh_derivative,

  clamp,
  gelu_derivative,

  dropout,
  max,
*/
union Converter {
public:
  float f;
  uint32_t u;

  Converter(float f_) : f(f_) {};

  static
  std::string to_hex(float f_) {
    Converter obj(f_);
    std::stringstream ss;
    ss << "0x" << std::hex << obj.u;
    return std::move(ss.str());
  }
};

inline
string get_op_name_parameterized(UnaryOpType::Enum op_type,float param0) {
    string op_name;
    TT_ASSERT( is_parameterized_type(op_type) && "operator should support one parameter" );

    switch (op_type) {
        case UnaryOpType::RELU_MAX: op_name = "relu_max_tile_init(); relu_max_tile(0,"+Converter::to_hex(param0)+"); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RELU_MIN: op_name = "relu_min_tile_init(); relu_min_tile(0,"+Converter::to_hex(param0)+"); pack_tile(0, CB::c_out0);"; break;
    case UnaryOpType::POWER: op_name = "power_tile_init(); power_tile(0," + std::to_string( (uint32_t) param0) + " ); pack_tile(0, CB::c_out0);"; break;
        default:
	  TT_ASSERT( false && "unexpected parameterized type");
    };
    return op_name;
}

inline
string get_op_name_default(UnaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case UnaryOpType::EXP: op_name = "exp_tile_init(); exp_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RECIP: op_name = "recip_tile_init(); recip_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::GELU: op_name = "gelu_tile_init(); gelu_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RELU: op_name = "pack_relu_tile_to_stream(0, CB::c_out0);"; break;
        case UnaryOpType::SQRT: op_name = "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::SIGMOID: op_name = "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG: op_name = "log_tile_init(); log_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::TANH: op_name = "tanh_tile_init(); tanh_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::SIN: op_name = "sin_tile_init(); sin_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::COS: op_name = "cos_tile_init(); cos_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x36f3); pack_tile(0,CB::c_out0);";
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_name = "log_with_base_tile_init(); log_with_base_tile(0,0x3dc5); pack_tile(0,CB::c_out0);";
            break;
        case UnaryOpType::ABS:
            op_name = "abs_tile_init(); abs_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::SIGN:
            op_name = "sign_tile_init(); sign_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::SQUARE:
            op_name = "square_tile_init(); square_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::EQZ:
            op_name = "eqz_tile_init(); eqz_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::NEZ:
            op_name = "nez_tile_init(); nez_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::LTZ:
            op_name = "ltz_tile_init(); ltz_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::GTZ:
            op_name = "gtz_tile_init(); gtz_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::LEZ:
            op_name = "lez_tile_init(); lez_tile(0); pack_tile(0,CB::c_out0);"; break;
        case UnaryOpType::GEZ:
            op_name = "gez_tile_init(); gez_tile(0); pack_tile(0,CB::c_out0);"; break;

        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

bool get_op_approx_mode(UnaryOpType::Enum op_type) {
    switch (op_type) {
        case UnaryOpType::GELU:
            return true;
        default:
            return false;
    }
}

static
void add_defines_impl(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type, std::string op_name){
    eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", op_name);
    bool is_relu = (op_type == UnaryOpType::RELU);
    eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "pack_relu_config(1);" : "");
    eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "pack_relu_config(0);" : "");
    return;
}

string get_op_name(UnaryOpType::Enum op_type,std::optional<float> param0) {
    return param0.has_value() ? get_op_name_parameterized(op_type, param0.value()) : get_op_name_default(op_type);
}

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type,std::optional<float> param0) {
    std::string op_name = get_op_name(op_type,param0);
    add_defines_impl(eltwise_unary_kernel,op_type,op_name);
    return;
}



UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a){
    uint32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return UnaryOpParallelizationStrategy::SINGLE_CORE;
    }
}


} // namespace eltwise_unary_op_utils

namespace tt {

namespace tt_metal {

void EltwiseUnary::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {}

std::vector<Shape> EltwiseUnary::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    return {input_tensor.shape()};
}

std::vector<Tensor> EltwiseUnary::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

operation::ProgramWithCallbacks EltwiseUnary::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = eltwise_unary_op_utils::get_parallelization_strategy(input_tensor);

    op_profiler::set_preferred_name(this->op_type);
    op_profiler::set_parallelization_strategy(parallelization_strategy);

    switch (parallelization_strategy){
        case UnaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_unary_multi_core(input_tensor, output_tensor, this->op_type,param);
            break;
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_unary_single_core(input_tensor, output_tensor, this->op_type,param);
    }
}

operation::Hash EltwiseUnary::compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();

    return fmt::format(
        "eltwise_unary_{}_{}",
         magic_enum::enum_name(this->op_type),
         operation::hash_tensor(input_tensor)
    );
}

}  // namespace tt_metal

}  // namespace tt
