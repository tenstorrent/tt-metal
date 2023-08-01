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
                EXPM1 = 28, SIGNBIT = 29, ASIN = 30, ACOS = 31, RSQRT = 32, RELU6 = 33, ATAN = 34};
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
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor &input_tensor);

operation::ProgramWithCallbacks eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});
operation::ProgramWithCallbacks eltwise_unary_single_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param = {});

template <UnaryOpType::Enum unary_op_type>
Tensor run_eltwise_unary(const Tensor& input_tensor, std::optional<float> param = std::nullopt, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) {
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(EltwiseUnary{unary_op_type, param, output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}

template <UnaryOpType::Enum unary_op_type>
struct make_eltwise_unary_with_param {
    Tensor operator()(const Tensor& input_tensor, float param, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) const {
        return run_eltwise_unary<unary_op_type>(input_tensor, param, output_mem_config);
    }
};

template <UnaryOpType::Enum unary_op_type>
struct make_eltwise_unary {
    Tensor operator()(const Tensor& input_tensor, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) const {
        return run_eltwise_unary<unary_op_type>(input_tensor, std::nullopt, output_mem_config);
    }
};

inline Tensor relu_without_autoformat(const Tensor& input_tensor, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) {
    return operation::run_without_autoformat(EltwiseUnary{UnaryOpType::RELU, std::nullopt, output_mem_config}, {input_tensor}).at(0);
}

constexpr auto sqrt = make_eltwise_unary<UnaryOpType::SQRT>{};
constexpr auto exp = make_eltwise_unary<UnaryOpType::EXP>{};
constexpr auto recip = make_eltwise_unary<UnaryOpType::RECIP>{};
constexpr auto relu = make_eltwise_unary<UnaryOpType::RELU>{};
constexpr auto relu6 = make_eltwise_unary<UnaryOpType::RELU6>{};
constexpr auto sigmoid = make_eltwise_unary<UnaryOpType::SIGMOID>{};
constexpr auto log = make_eltwise_unary<UnaryOpType::LOG>{};
constexpr auto tanh = make_eltwise_unary<UnaryOpType::TANH>{};
constexpr auto log2 = make_eltwise_unary<UnaryOpType::LOG2>{};
constexpr auto log10 = make_eltwise_unary<UnaryOpType::LOG10>{};
constexpr auto exp2 = make_eltwise_unary<UnaryOpType::EXP2>{};
constexpr auto expm1 = make_eltwise_unary<UnaryOpType::EXPM1>{};
constexpr auto sin = make_eltwise_unary<UnaryOpType::SIN>{};
constexpr auto cos = make_eltwise_unary<UnaryOpType::COS>{};
constexpr auto asin = make_eltwise_unary<UnaryOpType::ASIN>{};
constexpr auto acos = make_eltwise_unary<UnaryOpType::ACOS>{};
constexpr auto abs = make_eltwise_unary<UnaryOpType::ABS>{};
constexpr auto sign = make_eltwise_unary<UnaryOpType::SIGN>{};
constexpr auto signbit = make_eltwise_unary<UnaryOpType::SIGNBIT>{};
constexpr auto square = make_eltwise_unary<UnaryOpType::SQUARE>{};
constexpr auto eqz = make_eltwise_unary<UnaryOpType::EQZ>{};
constexpr auto nez = make_eltwise_unary<UnaryOpType::NEZ>{};
constexpr auto gez = make_eltwise_unary<UnaryOpType::GEZ>{};
constexpr auto lez = make_eltwise_unary<UnaryOpType::LEZ>{};
constexpr auto gtz = make_eltwise_unary<UnaryOpType::GTZ>{};
constexpr auto ltz = make_eltwise_unary<UnaryOpType::LTZ>{};

constexpr auto relu_max = make_eltwise_unary_with_param<UnaryOpType::RELU_MAX>{};
constexpr auto relu_min = make_eltwise_unary_with_param<UnaryOpType::RELU_MIN>{};
constexpr auto power = make_eltwise_unary_with_param<UnaryOpType::POWER>{};
constexpr auto leaky_relu = make_eltwise_unary_with_param<UnaryOpType::LEAKY_RELU>{};
constexpr auto elu = make_eltwise_unary_with_param<UnaryOpType::ELU>{};
constexpr auto heaviside = make_eltwise_unary_with_param<UnaryOpType::HEAVISIDE>{};
constexpr auto atan = make_eltwise_unary<UnaryOpType::ATAN>{};
inline Tensor gelu(const Tensor &input_tensor, bool fast_and_approx=true, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) {
    return make_eltwise_unary_with_param<UnaryOpType::GELU>{}(input_tensor, static_cast<float>(fast_and_approx), output_mem_config);
}
inline Tensor rsqrt(const Tensor &input_tensor, bool fast_and_approx=true, const MemoryConfig& output_mem_config = MemoryConfig{.interleaved = true}) {
    return make_eltwise_unary_with_param<UnaryOpType::RSQRT>{}(input_tensor, static_cast<float>(fast_and_approx), output_mem_config);
}

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
std::map<string, string> get_defines(UnaryOpType::Enum op_type, std::optional<float> param={});

} // namespace eltwise_unary_op_utils
