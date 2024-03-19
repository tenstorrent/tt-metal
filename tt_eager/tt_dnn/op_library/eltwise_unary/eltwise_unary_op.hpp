// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class UnaryOpType {
    EXP = 0,
    RECIP = 1,
    GELU = 2,
    RELU = 3,
    SQRT = 4,
    SIGMOID = 5,
    LOG = 6,
    TANH = 7,
    LOG2 = 8,
    LOG10 = 9,
    SIN = 10,
    COS = 11,
    ABS = 12,
    SIGN = 13,
    SQUARE = 14,
    EQZ = 15,
    NEZ = 16,
    GTZ = 17,
    LTZ = 18,
    GEZ = 19,
    LEZ = 20,
    RELU_MAX = 21,
    RELU_MIN = 22,
    POWER = 23,
    LEAKY_RELU = 24,
    ELU = 25,
    EXP2 = 26,
    HEAVISIDE = 27,
    EXPM1 = 28,
    SIGNBIT = 29,
    ASIN = 30,
    ACOS = 31,
    RSQRT = 32,
    RELU6 = 33,
    ATAN = 34,
    ERF = 35,
    ERFC = 36,
    ISINF = 37,
    ISPOSINF = 38,
    ISNEGINF = 39,
    ISNAN = 40,
    LOGICAL_NOT_UNARY = 41,
    ISFINITE = 42,
    ERFINV = 43,
    I0 = 44,
    TAN = 45,
    RSUB = 46,
    RDIV = 47,
    SILU = 48,
    IDENTITY = 49,
    NEG = 50,
    ADD_UNARY = 51,
    SUB_UNARY = 52,
    MUL_UNARY = 53,
    DIV_UNARY = 54,
    ADD_UNARY_SFPU = 55,
    SUB_UNARY_SFPU = 56,
    MUL_UNARY_SFPU = 57,
    DIV_UNARY_SFPU = 58
};

template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::POWER:
        case UnaryOpType::LEAKY_RELU:
        case UnaryOpType::ELU:
        case UnaryOpType::GELU:
        case UnaryOpType::RSQRT:
        case UnaryOpType::HEAVISIDE:
        case UnaryOpType::ERF:
        case UnaryOpType::ERFC:
        case UnaryOpType::RSUB:
        case UnaryOpType::RDIV:
        case UnaryOpType::EXP:
        case UnaryOpType::ADD_UNARY:
        case UnaryOpType::SUB_UNARY:
        case UnaryOpType::MUL_UNARY:
        case UnaryOpType::DIV_UNARY:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU: return true;
        default: return false;
    }
    return false;
}

struct UnaryWithParam {
    UnaryOpType op_type;
    std::optional<float> param = std::nullopt;

    static constexpr auto attribute_names = std::make_tuple("op_type", "param");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->op_type), std::cref(this->param)); }
};

enum class UnaryOpParallelizationStrategy { MULTI_CORE = 0, SINGLE_CORE = 1 };

struct EltwiseUnary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UnaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("op_chain", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->op_chain), std::cref(this->output_mem_config));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

Tensor eltwise_unary(const EltwiseUnary& op, const Tensor& input_tensor);

operation::ProgramWithCallbacks eltwise_unary_multi_core(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain);
operation::ProgramWithCallbacks eltwise_unary_single_core(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain);

inline Tensor run_eltwise_unary(
    const Tensor& input_tensor,
    std::vector<UnaryWithParam> ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    FormatParams input_format_params = {.pad_shape = pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
    return operation::run_with_autoformat(
               EltwiseUnary{ops_chain, output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE})
        .at(0);
}

inline Tensor run_eltwise_unary(
    CommandQueue& queue,
    const Tensor& input_tensor,
    std::vector<UnaryWithParam> ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    FormatParams input_format_params = {.pad_shape = pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
    return operation::run(
               queue,
               tt::tt_metal::operation::DeviceOperation(EltwiseUnary{ops_chain, output_mem_config}),
               {input_tensor})
        .at(0);
}

template <UnaryOpType unary_op_type, typename T = float>
struct make_eltwise_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor,
            {UnaryWithParam{.op_type = unary_op_type, .param = static_cast<float>(param)}},
            output_mem_config);
    }
};

template <UnaryOpType unary_op_type>
struct make_eltwise_unary {
    Tensor operator()(
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(input_tensor, {UnaryWithParam{.op_type = unary_op_type}}, output_mem_config);
    }
};

template <UnaryOpType unary_op_type, typename T = float>
struct make_eltwise_symmetric_binop_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor,
            {UnaryWithParam{.op_type = unary_op_type, .param = static_cast<float>(param)}},
            output_mem_config);
    }
    Tensor operator()(
        T param,
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor,
            {UnaryWithParam{.op_type = unary_op_type, .param = static_cast<float>(param)}},
            output_mem_config);
    }
};

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type, typename T = float>
struct make_eltwise_asymmetric_binop_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor,
            {UnaryWithParam{.op_type = unary_op_type, .param = static_cast<float>(param)}},
            output_mem_config);
    }
    Tensor operator()(
        T param,
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        return run_eltwise_unary(
            input_tensor,
            {UnaryWithParam{.op_type = unary_op_rev_type, .param = static_cast<float>(param)}},
            output_mem_config);
    }
};

inline Tensor sqrt(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(input_tensor, {UnaryWithParam{.op_type = UnaryOpType::SQRT}}, output_mem_config);
}

inline Tensor sqrt(
    CommandQueue& queue,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(queue, input_tensor, {UnaryWithParam{.op_type = UnaryOpType::SQRT}}, output_mem_config);
}

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
constexpr auto isfinite = make_eltwise_unary<UnaryOpType::ISFINITE>{};
constexpr auto isinf = make_eltwise_unary<UnaryOpType::ISINF>{};
constexpr auto isposinf = make_eltwise_unary<UnaryOpType::ISPOSINF>{};
constexpr auto isneginf = make_eltwise_unary<UnaryOpType::ISNEGINF>{};
constexpr auto isnan = make_eltwise_unary<UnaryOpType::ISNAN>{};
constexpr auto sign = make_eltwise_unary<UnaryOpType::SIGN>{};
constexpr auto signbit = make_eltwise_unary<UnaryOpType::SIGNBIT>{};
constexpr auto square = make_eltwise_unary<UnaryOpType::SQUARE>{};
constexpr auto atan = make_eltwise_unary<UnaryOpType::ATAN>{};
constexpr auto eqz = make_eltwise_unary<UnaryOpType::EQZ>{};
constexpr auto nez = make_eltwise_unary<UnaryOpType::NEZ>{};
constexpr auto gez = make_eltwise_unary<UnaryOpType::GEZ>{};
constexpr auto lez = make_eltwise_unary<UnaryOpType::LEZ>{};
constexpr auto gtz = make_eltwise_unary<UnaryOpType::GTZ>{};
constexpr auto ltz = make_eltwise_unary<UnaryOpType::LTZ>{};
constexpr auto logical_not_unary = make_eltwise_unary<UnaryOpType::LOGICAL_NOT_UNARY>{};
constexpr auto i0 = make_eltwise_unary<UnaryOpType::I0>{};
constexpr auto erfinv = make_eltwise_unary<UnaryOpType::ERFINV>{};
constexpr auto tan = make_eltwise_unary<UnaryOpType::TAN>{};
constexpr auto neg = make_eltwise_unary<UnaryOpType::NEG>{};
constexpr auto relu_max = make_eltwise_unary_with_param<UnaryOpType::RELU_MAX>{};
constexpr auto relu_min = make_eltwise_unary_with_param<UnaryOpType::RELU_MIN>{};
constexpr auto power = make_eltwise_unary_with_param<UnaryOpType::POWER, uint32_t>{};
constexpr auto leaky_relu = make_eltwise_unary_with_param<UnaryOpType::LEAKY_RELU>{};
constexpr auto prelu = leaky_relu;
constexpr auto elu = make_eltwise_unary_with_param<UnaryOpType::ELU>{};
constexpr auto heaviside = make_eltwise_unary_with_param<UnaryOpType::HEAVISIDE>{};
constexpr auto rsub = make_eltwise_unary_with_param<UnaryOpType::RSUB>{};
constexpr auto silu = make_eltwise_unary<UnaryOpType::SILU>{};
constexpr auto identity = make_eltwise_unary<UnaryOpType::IDENTITY>{};
constexpr auto add_unary_sfpu = make_eltwise_symmetric_binop_unary_with_param<UnaryOpType::ADD_UNARY_SFPU>{};
constexpr auto mul_unary_sfpu = make_eltwise_symmetric_binop_unary_with_param<UnaryOpType::MUL_UNARY_SFPU>{};
constexpr auto sub_unary_sfpu =
    make_eltwise_asymmetric_binop_unary_with_param<UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::RSUB>{};
constexpr auto div_unary_sfpu =
    make_eltwise_asymmetric_binop_unary_with_param<UnaryOpType::DIV_UNARY_SFPU, UnaryOpType::RDIV>{};

inline Tensor exp(
    const Tensor& input_tensor,
    bool fast_and_approx,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::EXP>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor exp(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return exp(input_tensor, false, output_mem_config);
}

inline Tensor erf(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::ERF>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor erfc(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::ERFC>{}(input_tensor, fast_and_approx, output_mem_config);
}

inline Tensor gelu(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::GELU>{}(input_tensor, fast_and_approx, output_mem_config);
}
inline Tensor rsqrt(
    const Tensor& input_tensor,
    bool fast_and_approx = true,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return make_eltwise_unary_with_param<UnaryOpType::RSQRT>{}(input_tensor, fast_and_approx, output_mem_config);
}

inline Tensor log_sigmoid(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(
        input_tensor,
        {UnaryWithParam{.op_type = UnaryOpType::SIGMOID}, UnaryWithParam{.op_type = UnaryOpType::LOG}},
        output_mem_config);
}
inline Tensor unary_chain(
    const Tensor& input_tensor,
    std::vector<UnaryWithParam> ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return run_eltwise_unary(input_tensor, ops_chain, output_mem_config);
}

Tensor sub_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor sub_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor add_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor add_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor mul_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mul_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor div_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor div_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// relops with unary argument
Tensor lte_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor lte_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor gte_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor gte_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor eq_unary(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor eq_unary(
    float value,
    const Tensor& input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// same as div_unary(value,tensor)
Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// deg2rad(a) using scale pi/180.
inline Tensor deg2rad(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return mul_unary(input_tensor, (float)(M_PI / 180.0), output_mem_config);
}

// rad2deg(a) using scale 180/pi.
inline Tensor rad2deg(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return mul_unary(input_tensor, (float)(180.0 / M_PI), output_mem_config);
}

// add 1
// use transformation y = 1.0 + x by broadcast
inline Tensor add1(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return add_unary(input_tensor, 1.0f, output_mem_config);
}

}  // namespace tt_metal

namespace operations {

namespace primary {

inline Tensor relu(
    const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run(
               EltwiseUnary{{UnaryWithParam{.op_type = UnaryOpType::RELU}}, output_mem_config}, {input_tensor})
        .at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType op_type);
std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, std::optional<float> param = {}, string idst = "0");
std::map<string, string> get_defines(
    UnaryOpType op_type, std::optional<float> param = {}, string id = "0", string idst = "0");
std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam> op_chain, string block_id = "0", string idst = "0");
}  // namespace eltwise_unary_op_utils
