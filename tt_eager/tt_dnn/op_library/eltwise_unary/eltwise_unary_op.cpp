// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"

using namespace tt::constants;

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

union Converter {
   public:
    float f;
    uint32_t u;

    Converter(float f_) : f(f_){};

    static std::string to_hex(float f_) {
        Converter obj(f_);
        std::stringstream ss;
        ss << "0x" << std::hex << obj.u;
        return ss.str();
    }
};

// update split eltwise ops include macros
void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines) {
    switch (op_type) {
        case UnaryOpType::EXP: defines["SFPU_OP_EXP_INCLUDE"] = "1"; break;
        case UnaryOpType::GELU: defines["SFPU_OP_GELU_INCLUDE"] = "1"; break;
        case UnaryOpType::RECIP: defines["SFPU_OP_RECIP_INCLUDE"] = "1"; break;
        case UnaryOpType::SQRT: defines["SFPU_OP_SQRT_INCLUDE"] = "1"; break;
        case UnaryOpType::ERFINV: defines["SFPU_OP_ERFINV_INCLUDE"] = "1"; break;
        case UnaryOpType::ERFC:
        case UnaryOpType::ERF: defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1"; break;
        case UnaryOpType::ELU: defines["SFPU_OP_ELU_INCLUDE"] = "1"; break;
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::LEAKY_RELU: defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1"; break;
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU: defines["SFPU_OP_BINOP_WITH_SCALAR_INCLUDE"] = "1"; break;
        case UnaryOpType::IDENTITY:
        case UnaryOpType::IDENTITY_UINT32: defines["SFPU_OP_IDENTITY_INCLUDE"] = "1"; break;
        case UnaryOpType::RDIV: break;
        case UnaryOpType::RSUB: defines["SFPU_OP_REVERSE_FAMILY_INCLUDE"] = "1";
        case UnaryOpType::ISINF:
        case UnaryOpType::ISNAN:
        case UnaryOpType::ISNEGINF:
        case UnaryOpType::ISPOSINF:
        case UnaryOpType::ISFINITE: defines["SFPU_OP_ISINF_ISNAN_INCLUDE"] = "1"; break;
        case UnaryOpType::LOGICAL_NOT_UNARY: defines["SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE"] = "1"; break;
        case UnaryOpType::I0: defines["SFPU_OP_I0_INCLUDE"] = "1"; break;
        case UnaryOpType::COS:
        case UnaryOpType::SIN:
        case UnaryOpType::TAN: defines["SFPU_OP_TRIG_FAMILY_INCLUDE"] = "1"; break;
        case UnaryOpType::NEG: defines["SFPU_OP_NEG_INCLUDE"] = "1"; break;
        case UnaryOpType::SOFTPLUS: defines["SFPU_OP_SOFTPLUS_INCLUDE"] = "1"; break;
        case UnaryOpType::TYPECAST: defines["SFPU_OP_TYPECAST_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_XOR: defines["SFPU_OP_BITWISE_XOR_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_NOT: defines["SFPU_OP_BITWISE_NOT_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_AND: defines["SFPU_OP_BITWISE_AND_INCLUDE"] = "1"; break;
        case UnaryOpType::RIGHT_SHIFT: defines["SFPU_OP_RIGHT_SHIFT_INCLUDE"] = "1"; break;
        case UnaryOpType::FLOOR: defines["SFPU_OP_FLOOR_INCLUDE"] = "1"; break;
        case UnaryOpType::LEFT_SHIFT: defines["SFPU_OP_LEFT_SHIFT_INCLUDE"] = "1"; break;
        case UnaryOpType::REMAINDER: defines["SFPU_OP_REMAINDER_INCLUDE"] = "1"; break;
        case UnaryOpType::FMOD: defines["SFPU_OP_FMOD_INCLUDE"] = "1"; break;
        default: defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1"; break;
    };
}

std::pair<string, string> get_op_init_and_func_parameterized(
    UnaryOpType op_type, std::vector<float> params, string idst) {
    std::pair<string, string> op_init_and_name;
    TT_FATAL(is_parametrized_type(op_type) && "operator should support at least one parameter");
    float param0 = params[0];
    switch (op_type) {
        case UnaryOpType::RELU_MAX:
            op_init_and_name = {
                "relu_max_tile_init();", fmt::format("relu_max_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::RELU_MIN:
            op_init_and_name = {
                "relu_min_tile_init();", fmt::format("relu_min_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::POWER:
            op_init_and_name = {
                "power_tile_init();", fmt::format("power_tile({}, {}u);", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::LEAKY_RELU:
            op_init_and_name = {
                "leaky_relu_tile_init();", fmt::format("leaky_relu_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::ELU:
            op_init_and_name = {"elu_tile_init();", fmt::format("elu_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::GELU:
            op_init_and_name = {
                fmt::format("gelu_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("gelu_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::RSQRT:
            op_init_and_name = {
                fmt::format("rsqrt_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("rsqrt_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::HEAVISIDE:
            op_init_and_name = {
                "heaviside_tile_init();", fmt::format("heaviside_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::BITWISE_XOR:
            op_init_and_name = {
                "bitwise_xor_tile_init();", fmt::format("bitwise_xor_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::BITWISE_NOT:
            op_init_and_name = {
                "bitwise_not_tile_init();", fmt::format("bitwise_not_tile({}, {}u);", idst, std::to_string((uint)param0))};
        case UnaryOpType::BITWISE_AND:
            op_init_and_name = {
                "bitwise_and_tile_init();", fmt::format("bitwise_and_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::RIGHT_SHIFT:
            op_init_and_name = {
                "right_shift_tile_init();",
                fmt::format("right_shift_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::LEFT_SHIFT:
            op_init_and_name = {
                "left_shift_tile_init();",
                fmt::format("left_shift_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::REMAINDER:
            op_init_and_name = {
                "remainder_tile_init();",
                fmt::format("remainder_tile({}, {}u, {}u);", idst, Converter::to_hex(param0), Converter::to_hex(1.0f/param0))};
            break;
        case UnaryOpType::FMOD:
            op_init_and_name = {
                "fmod_tile_init();",
                fmt::format("fmod_tile({}, {}u, {}u);", idst, Converter::to_hex(param0), Converter::to_hex(1.0f/param0))};
            break;
        case UnaryOpType::EXP:
            op_init_and_name = {
                fmt::format("exp_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("exp_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::ERF:
            op_init_and_name = {
                fmt::format("erf_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("erf_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::ERFC:
            op_init_and_name = {
                fmt::format("erfc_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("erfc_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::RDIV: op_init_and_name = {}; break;
        case UnaryOpType::RSUB:
            op_init_and_name = {
                "rsub_tile_init();", fmt::format("rsub_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::SUB_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("sub_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::ADD_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("add_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::MUL_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("mul_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::DIV_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("div_unary_tile({}, {}u);", idst, Converter::to_hex(1.0f / param0))};
            break;
        case UnaryOpType::UNARY_NE:
            op_init_and_name = {
                "unary_ne_tile_init();", fmt::format("unary_ne_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::UNARY_GT:
            op_init_and_name = {
                "unary_gt_tile_init();", fmt::format("unary_gt_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::UNARY_LT:
            op_init_and_name = {
                "unary_lt_tile_init();", fmt::format("unary_lt_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::SOFTPLUS: {
            TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
            float param1 = params[1];
            op_init_and_name = {
                "softplus_tile_init();",
                fmt::format(
                    "softplus_tile({}, {}u, {}u, {}u);",
                    idst,
                    Converter::to_hex(param0),
                    Converter::to_hex(1.0f / param0),  // Pass reciprocal to avoid doing it on device
                    Converter::to_hex(param1))};
            break;
        }
        case UnaryOpType::TYPECAST:
            TT_ASSERT(params.size() == 2, "Expected eltwise_typecast to take 2 parameters");
            op_init_and_name = {
                "typecast_tile_init();",
                fmt::format(
                    "typecast_tile<{1}u, {2}u>({0});",
                    idst,
                    std::to_string((uint32_t)datatype_to_dataformat_converter((DataType)params[0])),
                    std::to_string((uint32_t)datatype_to_dataformat_converter((DataType)params[1])))};
            break;
        default: TT_ASSERT(false && "unexpected parameterized type");
    };
    return op_init_and_name;
}

std::pair<string, string> get_op_init_and_func_default(UnaryOpType op_type, string idst) {
    std::pair<string, string> op_init_and_name;
    switch (op_type) {
        case UnaryOpType::RECIP: op_init_and_name = {"recip_tile_init();", fmt::format("recip_tile({});", idst)}; break;
        case UnaryOpType::RELU: op_init_and_name = {"relu_tile_init();", fmt::format("relu_tile({});", idst)}; break;
        case UnaryOpType::SQRT: op_init_and_name = {"sqrt_tile_init();", fmt::format("sqrt_tile({});", idst)}; break;
        case UnaryOpType::SIGMOID:
            op_init_and_name = {"sigmoid_tile_init();", fmt::format("sigmoid_tile({});", idst)};
            break;
        case UnaryOpType::LOG: op_init_and_name = {"log_tile_init();", fmt::format("log_tile({});", idst)}; break;
        case UnaryOpType::TANH: op_init_and_name = {"tanh_tile_init();", fmt::format("tanh_tile({});", idst)}; break;
        case UnaryOpType::SIGNBIT:
            op_init_and_name = {"signbit_tile_init();", fmt::format("signbit_tile({});", idst)};
            break;
        case UnaryOpType::FLOOR: op_init_and_name = {"floor_tile_init();", fmt::format("floor_tile({});", idst)}; break;
        case UnaryOpType::SIN: op_init_and_name = {"sin_tile_init();", fmt::format("sin_tile({});", idst)}; break;
        case UnaryOpType::COS: op_init_and_name = {"cos_tile_init();", fmt::format("cos_tile({});", idst)}; break;
        case UnaryOpType::ISFINITE:
            op_init_and_name = {"isfinite_tile_init();", fmt::format("isfinite_tile({});", idst)};
            break;
        case UnaryOpType::ISINF: op_init_and_name = {"isinf_tile_init();", fmt::format("isinf_tile({});", idst)}; break;
        case UnaryOpType::ISPOSINF:
            op_init_and_name = {"isposinf_tile_init();", fmt::format("isposinf_tile({});", idst)};
            break;
        case UnaryOpType::ISNEGINF:
            op_init_and_name = {"isneginf_tile_init();", fmt::format("isneginf_tile({});", idst)};
            break;
        case UnaryOpType::ISNAN: op_init_and_name = {"isnan_tile_init();", fmt::format("isnan_tile({});", idst)}; break;
        case UnaryOpType::LOGICAL_NOT_UNARY:
            op_init_and_name = {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile({});", idst)};
            break;
        case UnaryOpType::I0: op_init_and_name = {"i0_tile_init();", fmt::format("i0_tile({});", idst)}; break;
        case UnaryOpType::ERFINV:
            op_init_and_name = {"erfinv_tile_init();", fmt::format("erfinv_tile({});", idst)};
            break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x36f3u);", idst)};
            break;
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3dc5u);", idst)};
            break;
            break;
        case UnaryOpType::ABS: op_init_and_name = {"abs_tile_init();", fmt::format("abs_tile({});", idst)}; break;
        case UnaryOpType::SIGN: op_init_and_name = {"sign_tile_init();", fmt::format("sign_tile({});", idst)}; break;
        case UnaryOpType::SQUARE:
            op_init_and_name = {"square_tile_init();", fmt::format("square_tile({});", idst)};
            break;
        case UnaryOpType::TILED_PROD:
            op_init_and_name = {"tiled_prod_tile_init();", fmt::format("tiled_prod_tile({});", idst)};
            break;
        case UnaryOpType::EQZ: op_init_and_name = {"eqz_tile_init();", fmt::format("eqz_tile({});", idst)}; break;
        case UnaryOpType::NEZ: op_init_and_name = {"nez_tile_init();", fmt::format("nez_tile({});", idst)}; break;
        case UnaryOpType::LTZ: op_init_and_name = {"ltz_tile_init();", fmt::format("ltz_tile({});", idst)}; break;
        case UnaryOpType::GTZ: op_init_and_name = {"gtz_tile_init();", fmt::format("gtz_tile({});", idst)}; break;
        case UnaryOpType::LEZ: op_init_and_name = {"lez_tile_init();", fmt::format("lez_tile({});", idst)}; break;
        case UnaryOpType::GEZ: op_init_and_name = {"gez_tile_init();", fmt::format("gez_tile({});", idst)}; break;
        case UnaryOpType::EXP2: op_init_and_name = {"exp2_tile_init();", fmt::format("exp2_tile({});", idst)}; break;
        case UnaryOpType::EXPM1: op_init_and_name = {"expm1_tile_init();", fmt::format("expm1_tile({});", idst)}; break;
        case UnaryOpType::ASIN: op_init_and_name = {"asin_tile_init();", fmt::format("asin_tile({});", idst)}; break;
        case UnaryOpType::ACOS: op_init_and_name = {"acos_tile_init();", fmt::format("acos_tile({});", idst)}; break;
        case UnaryOpType::ATAN: op_init_and_name = {"atan_tile_init();", fmt::format("atan_tile({});", idst)}; break;
        case UnaryOpType::TAN: op_init_and_name = {"tan_tile_init();", fmt::format("tan_tile({});", idst)}; break;
        case UnaryOpType::SILU: op_init_and_name = {"silu_tile_init();", fmt::format("silu_tile({});", idst)}; break;
        case UnaryOpType::IDENTITY:
            op_init_and_name = {"identity_tile_init();", fmt::format("identity_tile({});", idst)};
            break;
        case UnaryOpType::IDENTITY_UINT32:
            op_init_and_name = {"identity_tile_init();", fmt::format("identity_tile_uint32({});", idst)};
            break;
        case UnaryOpType::RELU6:
            op_init_and_name = {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
            break;
        case UnaryOpType::NEG:
            op_init_and_name = {"negative_tile_init();", fmt::format("negative_tile({});", idst)};
            break;
        default: TT_ASSERT(false && "Undefined non-parametrized op type");
    }
    return op_init_and_name;
}

bool get_op_approx_mode(UnaryOpType op_type) {
    switch (op_type) {
        default: return false;
    }
}

std::map<string, string> get_defines_impl(
    UnaryOpType op_type,
    const std::vector<float>& params,
    std::string idst,
    std::string init_def,
    std::string func_def) {
    std::pair<string, string> op_init_and_name = get_op_init_and_func(op_type, params, idst);
    std::map<string, string> defines = {{init_def, op_init_and_name.first}, {func_def, op_init_and_name.second}};
    update_macro_defines(op_type, defines);
    return defines;
}

std::map<string, string> get_defines(
    UnaryOpType op_type, std::optional<std::vector<float>> params, std::string id, std::string idst) {
    std::string init_def = fmt::format("SFPU_OP_INIT_{}", id);
    std::string func_def = fmt::format("SFPU_OP_FUNC_{}", id);
    return get_defines_impl(
        op_type, params.has_value() ? params.value() : std::vector<float>{}, idst, init_def, func_def);
}

std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, std::vector<float> params, std::string idst) {
    return params.size() > 0 ? get_op_init_and_func_parameterized(op_type, params, idst)
                             : get_op_init_and_func_default(op_type, idst);
}

std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, std::string block_id, std::string idst) {
    std::vector<std::pair<string, string>> op_init_and_name;
    std::map<string, string> block_defines;
    std::string block_define = "";
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        block_define += init_def + " " + func_def + " ";
        block_defines.merge(get_defines_impl(op_chain[i].op_type, op_chain[i].params, idst, init_def, func_def));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

}  // namespace eltwise_unary_op_utils

namespace tt {

namespace tt_metal {

inline void validate_supported_arch_dtype(tt::ARCH arch, DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FLOOR:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::RIGHT_SHIFT:
            TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
            break;
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
            TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
            TT_FATAL(input_datatype == DataType::INT32, "Data type is not supported for Bitwise operations");
            TT_FATAL(output_datatype == DataType::INT32, "Data type is not supported for Bitwise operations");
            break;
        case UnaryOpType::FMOD:
            TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
            TT_FATAL(input_datatype == DataType::BFLOAT16, "Data type is not supported for Fmod operations");
            TT_FATAL(output_datatype == DataType::BFLOAT16, "Data type is not supported for Fmod operations");
            break;
        default:
            return;
    }
}

void EltwiseUnary::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &optional_output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto out_mem_config = (!optional_output_tensors.empty() && optional_output_tensors.at(0).has_value()) ? optional_output_tensors.at(0).value().memory_config() : this->output_mem_config;
    auto output_datatype = output_dtype;
    if(!optional_output_tensors.empty() && optional_output_tensors.at(0).has_value()){
        const auto& out = optional_output_tensors.at(0);
        output_datatype = out->get_dtype();
    }
    auto arch = input_tensor_a.device()->arch();
    auto input_datatype = input_tensor_a.get_dtype();
    for (const auto& unary_op : this->op_chain) {
        validate_supported_arch_dtype(arch, input_datatype, output_datatype, unary_op.op_type);
    }
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to eltwise unary need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr, "Operands to eltwise unary need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == out_mem_config.memory_layout,
        "Input and output memory layout must match");
    if (!input_tensor_a.is_sharded()) {
        TT_FATAL((input_tensor_a.get_layout() == Layout::TILE), "Inputs to eltwise unary must be tilized");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Interleaved memory layout supported");
    }
    if(!optional_output_tensors.empty() && optional_output_tensors.at(0).has_value()){
        const auto output_shape_required = this->compute_output_shapes(input_tensors);
        const auto& out_tensor = optional_output_tensors.at(0).value();
        TT_FATAL(out_tensor.get_legacy_shape() == output_shape_required.at(0), fmt::format("The input tensors need a shape of {}, however the output tensor is only {}", output_shape_required,  out_tensor.get_legacy_shape()));
    }
}

std::vector<Shape> EltwiseUnary::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> EltwiseUnary::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        return {output_tensors.at(0).value()};
    }

    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        Shape output_shape = compute_output_shapes(input_tensors).at(0);
        return {create_device_tensor(
            output_shape,
            this->output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config)};
    }
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseUnary::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    switch (parallelization_strategy) {
        case UnaryOpParallelizationStrategy::SHARDED_MULTI_CORE:
            return eltwise_unary_sharded(input_tensor, output_tensor, this->op_chain, this->fp32_dest_acc_en, this->preserve_fp32_precision);
        case UnaryOpParallelizationStrategy::MULTI_CORE:
        default: return eltwise_unary_multi_core(input_tensor, output_tensor, this->op_chain, this->fp32_dest_acc_en, this->preserve_fp32_precision);
    }
}

UnaryOpParallelizationStrategy EltwiseUnary::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.is_sharded())
        return UnaryOpParallelizationStrategy::SHARDED_MULTI_CORE;
    else {
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
}

const operation::Hash EltwiseUnary::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.legacy_shape();

    operation::Hash hash = operation::hash_operation<EltwiseUnary>(
        compute_volume(input_shape),
        input_tensor.dtype(),
        std::get<DeviceStorage>(input_tensor.storage()).memory_config(),
        this->output_mem_config);

    for (const auto& unary_with_param_op : this->op_chain) {
        hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.op_type);
        if (unary_with_param_op.has_parameter()) {
            hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.params);
        }
    }
    return hash;
}

// unary op version tie
template <BcastOpMath OP>
Tensor tie_binop_to_unary(
    uint8_t queue_id,
    const Tensor& input_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    const DataType& dtype = output_tensor.has_value() ? output_tensor.value().get_dtype() : input_tensor.get_dtype();
    Tensor t_value = ttnn::operations::creation::create_scalar(value, dtype, Layout::TILE, input_tensor.device());
    return bcast(queue_id, input_tensor, t_value, OP, BcastOpDim::HW, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
}

Tensor lte_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return lez(queue_id, sub_unary_sfpu(input_tensor, value, output_mem_config), output_mem_config, output_tensor);
}
Tensor lte_unary(uint8_t queue_id, float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return lez(queue_id, sub_unary_sfpu(value, input_tensor, output_mem_config), output_mem_config, output_tensor);
}
Tensor gte_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return gez(queue_id, sub_unary_sfpu(input_tensor, value, output_mem_config), output_mem_config, output_tensor);
}
Tensor gte_unary(uint8_t queue_id, float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return gez(queue_id, sub_unary_sfpu(value, input_tensor, output_mem_config), output_mem_config, output_tensor);
}
Tensor lte_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return lez(default_queue_id, sub_unary_sfpu(input_tensor, value, output_mem_config), output_mem_config, output_tensor);
}
Tensor lte_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return lez(default_queue_id, sub_unary_sfpu(value, input_tensor, output_mem_config), output_mem_config, output_tensor);
}
Tensor gte_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return gez(default_queue_id, sub_unary_sfpu(input_tensor, value, output_mem_config), output_mem_config, output_tensor);
}
Tensor gte_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return gez(default_queue_id, sub_unary_sfpu(value, input_tensor, output_mem_config), output_mem_config, output_tensor);
}
Tensor eq_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config) {
    return eqz(sub_unary_sfpu(input_tensor, value, output_mem_config), output_mem_config);
}

Tensor eq_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    return eq_unary(input_tensor, value, output_mem_config);
}

Tensor div_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return tie_binop_to_unary<BcastOpMath::MUL>(default_queue_id, input_tensor, 1.0f / value, output_mem_config, output_tensor);
}
Tensor div_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return tie_binop_to_unary<BcastOpMath::MUL>(queue_id, input_tensor, 1.0f / value, output_mem_config, output_tensor);
}

Tensor div_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    if(output_tensor.has_value()){
        recip(input_tensor, output_mem_config, output_tensor.value());
    }
    else{
        output_tensor = recip(input_tensor, output_mem_config);
    }
    return tie_binop_to_unary<BcastOpMath::MUL>(default_queue_id, output_tensor.value(), value, output_mem_config, output_tensor);
}
Tensor div_unary(uint8_t queue_id, float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    if(output_tensor.has_value()){
        recip(queue_id, input_tensor, output_mem_config, output_tensor.value());
    }
    else{
        output_tensor = recip(queue_id, input_tensor, output_mem_config);
    }
    return tie_binop_to_unary<BcastOpMath::MUL>(queue_id, output_tensor.value(), value, output_mem_config, output_tensor);
}

// same as div_unary(value,tensor)
Tensor rdiv(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config) {
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor result = div_unary(value, input_tensor, output_mem_config);
    result = where(eqz(input_tensor, output_mem_config), t_inf, result, output_mem_config);
    return result;
}

Tensor mul_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return tie_binop_to_unary<BcastOpMath::MUL>(default_queue_id, input_tensor, value, output_mem_config, output_tensor);
}
Tensor mul_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return tie_binop_to_unary<BcastOpMath::MUL>(queue_id, input_tensor, value, output_mem_config, output_tensor);
}

Tensor sub_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config) {
    uint8_t default_queue_id = 0;
    return tie_binop_to_unary<BcastOpMath::SUB>(default_queue_id, input_tensor, value, output_mem_config);
}

Tensor sub_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    return add_unary(value, neg(input_tensor, output_mem_config), output_mem_config);
}

Tensor add_unary(const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return tie_binop_to_unary<BcastOpMath::ADD>(default_queue_id, input_tensor, value, output_mem_config, output_tensor);
}
Tensor add_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return tie_binop_to_unary<BcastOpMath::ADD>(queue_id, input_tensor, value, output_mem_config, output_tensor);
}

// symmetric
Tensor add_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return add_unary(input_tensor, value, output_mem_config, output_tensor);
}
Tensor add_unary(uint8_t queue_id, float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return add_unary(queue_id, input_tensor, value, output_mem_config, output_tensor);
}

Tensor mul_unary(float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mul_unary(default_queue_id, input_tensor, value, output_mem_config, output_tensor);
}
Tensor mul_unary(uint8_t queue_id, float value, const Tensor& input_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return mul_unary(queue_id, input_tensor, value, output_mem_config, output_tensor);
}

}  // namespace tt_metal

}  // namespace tt
