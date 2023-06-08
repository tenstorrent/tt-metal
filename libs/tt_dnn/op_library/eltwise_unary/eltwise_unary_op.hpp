#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7, LOG2 = 8, LOG10 = 9 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH, LOG2, LOG10 }; }
    static UnaryOpType::Enum str2enum(std::string value_) {
      std::string value(value_.size(),'\0');
      for(int i = 0; i < value_.size(); i++) value[i] = toupper(value_[i]);
      if ( value == "EXP" ) return EXP;
      if ( value == "RECIP" ) return RECIP;
      if ( value == "GELU" ) return GELU;
      if ( value == "RELU" ) return RELU;
      if ( value == "SQRT" ) return SQRT;
      if ( value == "SIGMOID" ) return SQRT;
      if ( value == "LOG" ) return LOG;
      if ( value == "TANH" ) return TANH;
      if ( value == "LOG2" ) return LOG2;
      if ( value == "LOG10" ) return LOG10;
      TT_ASSERT(false,"string does not match any known operator");
      return LOG10;
    }
};

struct UnaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

Program eltwise_unary_single_core (const Tensor &input_tensor, Tensor &output_tensor, UnaryOpType::Enum op_type);
Program eltwise_unary_multi_core (const Tensor &input_tensor, Tensor &output_tensor, UnaryOpType::Enum op_type);

struct EltwiseUnary : Operation {
    const UnaryOpType::Enum op_type;

    EltwiseUnary(UnaryOpType::Enum op_type) : op_type{op_type} {}

    EltwiseUnary(const EltwiseUnary&) = delete;
    EltwiseUnary& operator=(const EltwiseUnary&) = delete;
    ~EltwiseUnary() {}

    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor &input_tensor);
inline Tensor sqrt(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::SQRT), input_tensor); }
inline Tensor exp(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::EXP), input_tensor); }
inline Tensor recip(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::RECIP), input_tensor); }
inline Tensor gelu(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::GELU), input_tensor); }
inline Tensor relu(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::RELU), input_tensor); }
inline Tensor sigmoid(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::SIGMOID), input_tensor); }
inline Tensor log(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::LOG), input_tensor); }
inline Tensor tanh(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::TANH), input_tensor); }
inline Tensor log2(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::LOG2), input_tensor); }
inline Tensor log10(const Tensor &input_tensor) { return eltwise_unary(EltwiseUnary(UnaryOpType::LOG10), input_tensor); }


}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

string get_op_name(UnaryOpType::Enum op_type);

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type);

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &input_tensor);

} // namespace eltwise_unary_op_utils
