// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/types.hpp"

namespace tt {

namespace tt_metal {

// These operations have a corresponding LLK available
enum class UnaryOpType {
    EXP,
    RECIP,
    GELU,
    RELU,
    SQRT,
    SIGMOID,
    LOG,
    TANH,
    LOG2,
    LOG10,
    SIN,
    COS,
    ABS,
    SIGN,
    SQUARE,
    EQZ,
    NEZ,
    GTZ,
    LTZ,
    GEZ,
    LEZ,
    RELU_MAX,
    RELU_MIN,
    POWER,
    LEAKY_RELU,
    ELU,
    EXP2,
    HEAVISIDE,
    EXPM1,
    SIGNBIT,
    ASIN,
    ACOS,
    RSQRT,
    RELU6,
    ATAN,
    ERF,
    ERFC,
    ISINF,
    ISPOSINF,
    ISNEGINF,
    ISNAN,
    LOGICAL_NOT_UNARY,
    ISFINITE,
    ERFINV,
    I0,
    TAN,
    RSUB,
    RDIV,
    SILU,
    SOFTPLUS,
    IDENTITY,
    NEG,
    ADD_UNARY_SFPU,
    SUB_UNARY_SFPU,
    MUL_UNARY_SFPU,
    DIV_UNARY_SFPU,
    IDENTITY_UINT32,
    UNARY_NE,
    UNARY_GT,
    UNARY_LT,
    TILED_PROD,
    TYPECAST,
    BITWISE_XOR,
    BITWISE_NOT,
    BITWISE_AND,
    BITWISE_OR,
    RIGHT_SHIFT,
    FLOOR,
    LEFT_SHIFT,
    REMAINDER,
    FMOD,
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
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::TYPECAST:
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
        case UnaryOpType::RIGHT_SHIFT:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FMOD: return true;
        default: return false;
    }
    return false;
}

struct UnaryWithParam {
    UnaryOpType op_type;
    std::vector<float> params;

    UnaryWithParam(UnaryOpType op_type, const std::vector<float>& params) : op_type{op_type}, params{params} {}
    UnaryWithParam(UnaryOpType op_type, float param) : op_type{op_type}, params{param} {}
    UnaryWithParam(UnaryOpType op_type) : op_type{op_type} {}

    bool has_parameter() const { return params.size() > 0; }

    static constexpr auto attribute_names = std::make_tuple("op_type", "param");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->op_type), std::cref(this->params)); }
};

inline UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "relu")
        return UnaryWithParam(UnaryOpType::RELU);
    else if (name == "gelu")
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    else if (name == "silu")
        return UnaryWithParam(UnaryOpType::SILU);
    else if (name == "sigmoid")
        return UnaryWithParam(UnaryOpType::SIGMOID);
    else if (name == "sqrt")
        return UnaryWithParam(UnaryOpType::SQRT);
    else if (name == "exp")
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    else if (name == "recip")
        return UnaryWithParam(UnaryOpType::RECIP);
    else if (name == "log")
        return UnaryWithParam(UnaryOpType::LOG);
    else if (name == "tanh")
        return UnaryWithParam(UnaryOpType::TANH);
    else if (name == "log2")
        return UnaryWithParam(UnaryOpType::LOG2);
    else if (name == "log10")
        return UnaryWithParam(UnaryOpType::LOG10);
    else if (name == "sin")
        return UnaryWithParam(UnaryOpType::SIN);
    else if (name == "cos")
        return UnaryWithParam(UnaryOpType::COS);
    else if (name == "abs")
        return UnaryWithParam(UnaryOpType::ABS);
    else if (name == "sign")
        return UnaryWithParam(UnaryOpType::SIGN);
    else if (name == "square")
        return UnaryWithParam(UnaryOpType::SQUARE);
    else if (name == "softplus")
        return UnaryWithParam(UnaryOpType::SOFTPLUS);
    TT_THROW("Unknown unary op: " + name);
}

enum class UnaryOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct EltwiseUnary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;
    bool fp32_dest_acc_en;
    bool preserve_fp32_precision;
    DataType output_dtype;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &optional_output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UnaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

operation::ProgramWithCallbacks eltwise_unary_sharded(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain, bool fp32_dest_acc_en, bool preserve_fp32_precision);
operation::ProgramWithCallbacks eltwise_unary_multi_core(
    const Tensor& a, Tensor& output, const std::vector<UnaryWithParam> op_chain, bool fp32_dest_acc_en, bool preserve_fp32_precision);

inline Tensor run_eltwise_unary_with_output_tensor(
    uint8_t cq_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor = std::nullopt) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    DataType output_dtype = (ops_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(ops_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (ops_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en =
        preserve_fp32_precision or
        output_dtype == DataType::UINT32 or
        output_dtype == DataType::INT32 or
        output_dtype == DataType::FLOAT32 or
        input_tensor.get_dtype() == DataType::UINT32 or
        input_tensor.get_dtype() ==
            DataType::INT32;  // MT: Currently only uint32/int32 is moved to DST directly, fp32 is converted to fp16b
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    if (output_mem_config.is_sharded()) {
        operation::launch_op(
            [ops_chain, output_mem_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype, output_tensor, cq_id](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run_without_autoformat(
                    EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype}, input_tensors, {}, {output_tensor}, cq_id);
            },
            {input_tensor},
            output_tensors,
            {},
            {output_tensor});
    } else {
        operation::launch_with_autoformat(
            [ops_chain, output_mem_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype, output_tensor, cq_id](
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                Tensor input_tensor = input_tensors.at(0);
                Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                FormatParams input_format_params = {
                    .pad_shape = pad_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
                return operation::run_with_autoformat(
                    EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype},
                    {input_tensor},
                    {input_format_params},
                    {Layout::TILE},
                    {},
                    {},
                    {output_tensor},
                    cq_id
                    );
            },
            {input_tensor},
            output_tensors,
            {},
            {output_tensor});
    }
    return output_tensors.at(0);
}

inline Tensor run_eltwise_unary(
    uint8_t cq_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor = std::nullopt) {
    TT_FATAL(ops_chain.size() > 0, "At least 1 unary op must be specified");
    DataType output_dtype = (ops_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(ops_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (ops_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en =
        preserve_fp32_precision or
        output_dtype == DataType::UINT32 or
        output_dtype == DataType::INT32 or
        output_dtype == DataType::FLOAT32 or
        input_tensor.get_dtype() == DataType::UINT32 or
        input_tensor.get_dtype() ==
            DataType::INT32;  // MT: Currently only uint32/int32 is moved to DST directly, fp32 is converted to fp16b
    return operation::run(
               EltwiseUnary{ops_chain, output_mem_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype},
               {input_tensor}, {}, {output_tensor}, cq_id)
        .at(0);
}

template <UnaryOpType unary_op_type, typename T = float>
struct make_eltwise_unary_with_param {
    Tensor operator()(
        const Tensor& input_tensor,
        T param,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        uint8_t default_queue_id = 0;
        return run_eltwise_unary_with_output_tensor(
            default_queue_id, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, output_mem_config);
    }
};

template <UnaryOpType unary_op_type>
struct make_eltwise_unary {
    Tensor operator()(
        const Tensor& input_tensor,
        const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        uint8_t default_queue_id = 0;
        return run_eltwise_unary_with_output_tensor(
            default_queue_id, input_tensor, {UnaryWithParam(unary_op_type)}, output_mem_config);
    }
};


inline Tensor unary_chain(
    const Tensor& input_tensor,
    std::vector<UnaryWithParam> ops_chain,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    uint8_t default_queue_id = 0;
    return run_eltwise_unary_with_output_tensor(
        default_queue_id, input_tensor, ops_chain, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

bool get_op_approx_mode(UnaryOpType op_type);
std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, std::vector<float> params = {}, string idst = "0");
std::map<string, string> get_defines(
    UnaryOpType op_type, std::optional<std::vector<float>> params = std::nullopt, string id = "0", string idst = "0");
std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, string block_id = "0", string idst = "0");
}  // namespace eltwise_unary_op_utils
