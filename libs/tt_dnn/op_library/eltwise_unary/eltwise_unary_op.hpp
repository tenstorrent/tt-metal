#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7, LOG2 = 8, LOG10 = 9, SIN = 10, COS = 11,
                ABS=12, SIGN=13, SQUARE=14, EQZ = 15, NEZ = 16, GTZ = 17, LTZ = 18, GEZ = 19, LEZ = 20, RELU_MAX = 21, RELU_MIN = 22, POWER = 23  };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH, LOG2, LOG10, SIN, COS, ABS, SIGN, SQUARE,
                EQZ , NEZ , GTZ , LTZ , GEZ , LEZ , RELU_MAX , RELU_MIN, POWER }; }
};

struct UnaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

struct EltwiseUnary {
    const UnaryOpType::Enum op_type;
    std::optional<float> param;

    explicit EltwiseUnary(UnaryOpType::Enum op_type,std::optional<float> param_={}) : op_type{op_type}, param(param_) {}

    ProgramHash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor &input_tensor);
Program eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});
Program eltwise_unary_single_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});

inline Tensor sqrt(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::SQRT), input_tensor); }
inline Tensor exp(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::EXP), input_tensor); }
inline Tensor recip(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::RECIP), input_tensor); }
inline Tensor gelu(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::GELU), input_tensor); }
inline Tensor relu(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::RELU), input_tensor); }
inline Tensor sigmoid(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::SIGMOID), input_tensor); }
inline Tensor log(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::LOG), input_tensor); }
inline Tensor tanh(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::TANH), input_tensor); }
inline Tensor log2(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::LOG2), input_tensor); }
inline Tensor log10(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::LOG10), input_tensor); }


inline Tensor sin(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::SIN), input_tensor); }
inline Tensor cos(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::COS), input_tensor); }
inline Tensor abs(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::ABS), input_tensor); }
inline Tensor sign(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::SIGN), input_tensor); }
inline Tensor square(const Tensor &input_tensor) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::SQUARE), input_tensor); }

inline Tensor eqz(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::EQZ),a); }
inline Tensor nez(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::NEZ),a); }
inline Tensor gez(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::GEZ),a); }
inline Tensor lez(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::LEZ),a); }
inline Tensor gtz(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::GTZ),a); }
inline Tensor ltz(const Tensor &a) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::LTZ),a); }

inline Tensor relu_max(const Tensor& a,float upper_limit) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::RELU_MAX,upper_limit),a); }
inline Tensor relu_min(const Tensor& a,float lower_limit) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::RELU_MIN,lower_limit),a); }
inline Tensor power(const Tensor& a,uint32_t exponent) { return operation::run_with_autopad(EltwiseUnary(UnaryOpType::POWER,exponent),a); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType::Enum op_type);
string get_op_name(UnaryOpType::Enum op_type,std::optional<float> param={});
void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type,std::optional<float> param={});

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &input_tensor);

} // namespace eltwise_unary_op_utils
