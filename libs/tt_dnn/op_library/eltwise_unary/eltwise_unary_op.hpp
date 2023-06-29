#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7, LOG2 = 8, LOG10 = 9, SIN = 10, COS = 11,
                ABS=12, SIGN=13, SQUARE=14, EQZ = 15, NEZ = 16, GTZ = 17, LTZ = 18, GEZ = 19, LEZ = 20, RELU_MAX = 21, RELU_MIN = 22, POWER = 23};
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH, LOG2, LOG10, SIN, COS, ABS, SIGN, SQUARE,
                EQZ , NEZ , GTZ , LTZ , GEZ , LEZ , RELU_MAX , RELU_MIN, POWER}; }
};

struct UnaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

struct EltwiseUnary {
    const UnaryOpType::Enum op_type;
    const std::optional<float> param;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

std::ostream& operator<<(std::ostream& os, const EltwiseUnary& op);

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor &input_tensor);

operation::ProgramWithCallbacks eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});
operation::ProgramWithCallbacks eltwise_unary_single_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});

inline Tensor sqrt(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::SQRT}, input_tensor); }
inline Tensor exp(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::EXP}, input_tensor); }
inline Tensor recip(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::RECIP}, input_tensor); }
inline Tensor gelu(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::GELU}, input_tensor); }
inline Tensor relu(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::RELU}, input_tensor); }
inline Tensor sigmoid(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::SIGMOID}, input_tensor); }
inline Tensor log(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::LOG}, input_tensor); }
inline Tensor tanh(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::TANH}, input_tensor); }
inline Tensor log2(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::LOG2}, input_tensor); }
inline Tensor log10(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::LOG10}, input_tensor); }


inline Tensor sin(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::SIN}, input_tensor); }
inline Tensor cos(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::COS}, input_tensor); }
inline Tensor abs(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::ABS}, input_tensor); }
inline Tensor sign(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::SIGN}, input_tensor); }
inline Tensor square(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::SQUARE}, input_tensor); }

inline Tensor eqz(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::EQZ}, input_tensor); }
inline Tensor nez(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::NEZ}, input_tensor); }
inline Tensor gez(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::GEZ}, input_tensor); }
inline Tensor lez(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::LEZ}, input_tensor); }
inline Tensor gtz(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::GTZ}, input_tensor); }
inline Tensor ltz(const Tensor &input_tensor) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::LTZ}, input_tensor); }

inline Tensor relu_max(const Tensor& input_tensor, float upper_limit) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::RELU_MAX, upper_limit}, input_tensor); }
inline Tensor relu_min(const Tensor& input_tensor, float lower_limit) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::RELU_MIN, lower_limit}, input_tensor); }
inline Tensor power(const Tensor& input_tensor, uint32_t exponent) { return operation::run_with_autoformat(EltwiseUnary{UnaryOpType::POWER, exponent}, input_tensor); }

// binop with tied inputs.
Tensor sub_unary(const Tensor& input_tensor, float value);
Tensor sub_unary(float value, const Tensor& input_tensor);

Tensor add_unary(const Tensor& input_tensor, float value);
Tensor add_unary(float value, const Tensor& input_tensor);

Tensor mul_unary(const Tensor& input_tensor, float value);
Tensor mul_unary(float value, const Tensor& input_tensor);

Tensor div_unary(const Tensor& input_tensor, float value);
Tensor div_unary(float value, const Tensor& input_tensor);

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType::Enum op_type);
string get_op_name(UnaryOpType::Enum op_type, std::optional<float> param={});
void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type, std::optional<float> param={});

} // namespace eltwise_unary_op_utils
