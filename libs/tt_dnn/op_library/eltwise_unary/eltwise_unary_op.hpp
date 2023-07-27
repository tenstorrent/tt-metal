#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7, LOG2 = 8, LOG10 = 9, SIN = 10, COS = 11,
                ABS=12, SIGN=13, SQUARE=14, EQZ = 15, NEZ = 16, GTZ = 17, LTZ = 18, GEZ = 19, LEZ = 20, RELU_MAX = 21, RELU_MIN = 22, POWER = 23, LEAKY_RELU = 24, ELU = 25, EXP2 = 26, HEAVISIDE = 27,
                EXPM1 = 28, SIGNBIT = 29, ASIN = 30, ACOS = 31, RSQRT = 32, RELU6 = 33 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

template <typename T>
bool is_parametrized_type(T val) {
    switch ( val ) {
    case UnaryOpType::RELU_MAX:
    case UnaryOpType::RELU_MIN:
    case UnaryOpType::POWER:
    case UnaryOpType::LEAKY_RELU:
    case UnaryOpType::ELU:
    case UnaryOpType::GELU:
    case UnaryOpType::RSQRT:
    case UnaryOpType::HEAVISIDE:
        return true;
    default:
        return false;
    }
    return false;
}

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
    tt::stl::reflection::Attributes attributes() const;
};

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor &input_tensor);

operation::ProgramWithCallbacks eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});
operation::ProgramWithCallbacks eltwise_unary_single_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});

template <UnaryOpType::Enum unary_op_type>
Tensor run_eltwise_unary(const Tensor& input_tensor, std::optional<float> param = std::nullopt) {
    // TODO: Replace padding and target/output layouts to match input when RM/CL support is merged
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(EltwiseUnary{unary_op_type, param}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}

inline Tensor relu_without_autoformat(const Tensor& input_tensor) {
    return operation::run_without_autoformat(EltwiseUnary{UnaryOpType::RELU, std::nullopt}, {input_tensor}).at(0);
}

inline Tensor sqrt(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SQRT>(input_tensor); }
inline Tensor exp(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::EXP>(input_tensor); }
inline Tensor recip(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::RECIP>(input_tensor); }
inline Tensor relu(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::RELU>(input_tensor); }
inline Tensor relu6(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::RELU6>(input_tensor); }
inline Tensor sigmoid(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SIGMOID>(input_tensor); }
inline Tensor log(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::LOG>(input_tensor); }
inline Tensor tanh(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::TANH>(input_tensor); }
inline Tensor log2(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::LOG2>(input_tensor); }
inline Tensor log10(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::LOG10>(input_tensor); }
inline Tensor exp2(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::EXP2>(input_tensor); }
inline Tensor expm1(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::EXPM1>(input_tensor); }


inline Tensor sin(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SIN>(input_tensor); }
inline Tensor cos(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::COS>(input_tensor); }
inline Tensor asin(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::ASIN>(input_tensor); }
inline Tensor acos(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::ACOS>(input_tensor); }
inline Tensor abs(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::ABS>(input_tensor); }
inline Tensor sign(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SIGN>(input_tensor); }
inline Tensor signbit(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SIGNBIT>(input_tensor); }
inline Tensor square(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::SQUARE>(input_tensor); }

inline Tensor eqz(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::EQZ>(input_tensor); }
inline Tensor nez(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::NEZ>(input_tensor); }
inline Tensor gez(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::GEZ>(input_tensor); }
inline Tensor lez(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::LEZ>(input_tensor); }
inline Tensor gtz(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::GTZ>(input_tensor); }
inline Tensor ltz(const Tensor &input_tensor) { return run_eltwise_unary<UnaryOpType::LTZ>(input_tensor); }

inline Tensor relu_max(const Tensor& input_tensor, float upper_limit) { return run_eltwise_unary<UnaryOpType::RELU_MAX>(input_tensor, upper_limit); }
inline Tensor relu_min(const Tensor& input_tensor, float lower_limit) { return run_eltwise_unary<UnaryOpType::RELU_MIN>(input_tensor, lower_limit); }
inline Tensor power(const Tensor& input_tensor, uint32_t exponent) { return run_eltwise_unary<UnaryOpType::POWER>(input_tensor, exponent); }
inline Tensor leaky_relu(const Tensor& input_tensor, float slope) { return run_eltwise_unary<UnaryOpType::LEAKY_RELU>(input_tensor, slope); }
inline Tensor elu(const Tensor& input_tensor, float slope) { return run_eltwise_unary<UnaryOpType::ELU>(input_tensor, slope); }
inline Tensor gelu(const Tensor &input_tensor,bool fast_and_approx=true) { return run_eltwise_unary<UnaryOpType::GELU>(input_tensor, static_cast<float>(fast_and_approx)); }
inline Tensor rsqrt(const Tensor &input_tensor,bool fast_and_approx=true) { return run_eltwise_unary<UnaryOpType::RSQRT>(input_tensor, static_cast<float>(fast_and_approx)); }
inline Tensor heaviside(const Tensor& input_tensor, float value) { return run_eltwise_unary<UnaryOpType::HEAVISIDE>(input_tensor, value); }

// binop with tied inputs.
Tensor sub_unary(const Tensor& input_tensor, float value);
Tensor sub_unary(float value, const Tensor& input_tensor);

Tensor add_unary(const Tensor& input_tensor, float value);
Tensor add_unary(float value, const Tensor& input_tensor);

Tensor mul_unary(const Tensor& input_tensor, float value);
Tensor mul_unary(float value, const Tensor& input_tensor);

Tensor div_unary(const Tensor& input_tensor, float value);
Tensor div_unary(float value, const Tensor& input_tensor);

//deg2rad(a) using scale pi/180.
inline Tensor deg2rad(const Tensor &input_tensor) { return mul_unary(input_tensor, (float)(M_PI/180.0)); }

//rad2deg(a) using scale 180/pi.
inline Tensor rad2deg(const Tensor &input_tensor) { return mul_unary(input_tensor, (float)(180.0/M_PI)); }

// Function neg
//use transformation y = -1 * x by broadcast
inline Tensor neg(const Tensor &input_tensor) { return mul_unary(input_tensor, -1.0f); }

//add 1
//use transformation y = 1.0 + x by broadcast
inline Tensor add1(const Tensor &input_tensor) { return add_unary(input_tensor, 1.0f); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType::Enum op_type);
string get_op_name(UnaryOpType::Enum op_type, std::optional<float> param={});
void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type, std::optional<float> param={});

} // namespace eltwise_unary_op_utils
