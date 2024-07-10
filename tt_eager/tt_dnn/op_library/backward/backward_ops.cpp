// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/backward/backward_ops.hpp"

#include "tt_dnn/op_library/complex/complex_ops.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/cpp/ttnn/operations/embedding/embedding/embedding.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace tt {

namespace tt_metal {

// unary_pow:
// grad_input = grad * exponent * torch.pow(input, exponent - 1)
std::vector<std::optional<Tensor>> _unary_pow_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    TT_FATAL(are_required_outputs.at(0) , "input_grad derivative is required output");

    const float ZERO_THRESHOLD = std::numeric_limits<float>::epsilon() * 10.0f;
    TT_FATAL(exponent >= 0.0, "negative exponents are not supported; use recip(pow(input,abs(exponent)))");
    if (std::abs(exponent) < ZERO_THRESHOLD) {
        if(input_grad.has_value()){
            zeros_like(queue_id, input, output_mem_config, input_grad);
        } else {
        input_grad = zeros_like(queue_id, input, output_mem_config);
        }
        grad_tensor.emplace_back(input_grad);
        return grad_tensor;
    }

    Tensor power_input = ttnn::power(queue_id,input, fabs(exponent - 1.0f), output_mem_config);
    if (exponent < 1.0f) {
        power_input = ttnn::reciprocal(queue_id, power_input, output_mem_config);
    }

    Tensor result = ttnn::multiply(queue_id, power_input, exponent, std::nullopt, output_mem_config);
    power_input.deallocate();
    Tensor final_result = ttnn::multiply(queue_id, result, grad, std::nullopt, output_mem_config);
    result.deallocate();
    Tensor temp = where(queue_id, ttnn::le(queue_id, final_result, -3.4e+38, std::nullopt, output_mem_config), -std::numeric_limits<float>::infinity(), final_result, output_mem_config);
    if(input_grad.has_value()){
        where(queue_id, ttnn::ge(queue_id, final_result, 3.4e+38, std::nullopt, output_mem_config), std::numeric_limits<float>::infinity(), temp, output_mem_config, input_grad);
    } else {
        input_grad = where(queue_id, ttnn::ge(queue_id, final_result, 3.4e+38, std::nullopt, output_mem_config), std::numeric_limits<float>::infinity(), temp, output_mem_config);
    }
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}
std::vector<std::optional<Tensor>> unary_pow_bw(uint8_t queue_id,  const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    return operation::decorate_as_composite(__func__, _unary_pow_bw)(queue_id, grad, input, exponent, output_mem_config, are_required_outputs, input_grad);
}
std::vector<std::optional<Tensor>> unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _unary_pow_bw)(default_queue_id, grad, input, exponent, output_mem_config, are_required_outputs, input_grad);
}

std::vector<std::optional<Tensor>> _exp_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    TT_FATAL(are_required_outputs.at(0), "input_grad derivative is a required output");

    float t_inf = std::numeric_limits<float>::infinity();
    Tensor exp_result = ttnn::exp(queue_id, input, false, output_mem_config);
    Tensor result = ttnn::multiply(queue_id, grad, exp_result, std::nullopt, output_mem_config);
    result = where(queue_id, ttnn::ge(queue_id, result, 1e+38, std::nullopt, output_mem_config), t_inf, result, output_mem_config);
    result = where(queue_id, ttnn::ge(queue_id, result, -1e+38, std::nullopt,  output_mem_config), -t_inf, result, output_mem_config);
    if(input_grad.has_value()){
        where(queue_id,
        ttnn::logical_and(
            ttnn::ge(queue_id, ttnn::abs(queue_id, exp_result, output_mem_config), 1e+38, std::nullopt, output_mem_config),
            ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), -t_inf, result, output_mem_config, input_grad);
    } else {
    input_grad = where(queue_id,
        ttnn::logical_and(
            ttnn::ge(queue_id, ttnn::abs(queue_id, exp_result, output_mem_config), 1e+38, std::nullopt, output_mem_config),
            ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), -t_inf, result, output_mem_config);
    }
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}
std::vector<std::optional<Tensor>> exp_bw(uint8_t queue_id,  const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    return operation::decorate_as_composite(__func__, _exp_bw)(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}
std::vector<std::optional<Tensor>> exp_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _exp_bw)(default_queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}

// sqrt_bw
std::vector<std::optional<Tensor>> _sqrt_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    TT_FATAL(are_required_outputs.at(0), "input_grad derivative is required output");

    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();

    if(input_grad.has_value()){
        ttnn::sqrt(queue_id, input, output_mem_config, input_grad);
        ttnn::multiply(queue_id, grad, ttnn::reciprocal(queue_id, ttnn::multiply(queue_id, input_grad.value(), 2.0, std::nullopt, output_mem_config), output_mem_config),std::nullopt,output_mem_config, input_grad);
        where(queue_id, ttnn::lez(queue_id, input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
        where(queue_id,ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), -t_inf,input_grad.value(),output_mem_config,input_grad);
        where(queue_id, ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::gtz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config), t_inf,input_grad.value(),output_mem_config,input_grad);
    } else {
    Tensor sqrt_result = ttnn::sqrt(queue_id, input, output_mem_config);
    Tensor result = ttnn::multiply(queue_id, grad, recip(queue_id, ttnn::multiply(queue_id, sqrt_result, 2.0, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    sqrt_result.deallocate();
    input_grad = where(queue_id, ttnn::lez(queue_id, input, output_mem_config), t_nan, result, output_mem_config);
    input_grad = where(queue_id, ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::ltz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config),-t_inf, input_grad.value(),output_mem_config);
    input_grad = where(queue_id, ttnn::logical_and(queue_id, ttnn::eqz(queue_id, input, output_mem_config), ttnn::gtz(queue_id, grad, output_mem_config), std::nullopt, output_mem_config),t_inf, input_grad.value(), output_mem_config);
    }
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}
std::vector<std::optional<Tensor>> sqrt_bw(uint8_t queue_id,  const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    return operation::decorate_as_composite(__func__, _sqrt_bw)(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}
std::vector<std::optional<Tensor>> sqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _sqrt_bw)(default_queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}

std::vector<Tensor> _unary_div_bw(
    const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float inv_scalar = 1.0f / scalar;
    if (round_mode == "None") {
        Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
        if (scalar == 0.0) {
            float t_nan = std::nanf("");
            grad_tensor.emplace_back(where(
                ttnn::eqz(grad, output_mem_config),
                t_nan,
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                output_mem_config));
        } else {
            grad_tensor.emplace_back(ttnn::multiply(grad, inv_scalar, std::nullopt, output_mem_config));
        }
    } else {
        Tensor result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}
std::vector<Tensor> unary_div_bw(
    const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_div_bw)(
        grad, input, scalar, round_mode, output_mem_config);
}

std::vector<Tensor> _rdiv_bw(
    const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    if (round_mode == "None") {
        Tensor result = where(
            ttnn::nez(input),
            ttnn::multiply(ttnn::neg(grad, output_mem_config),
                (ttnn::multiply(ttnn::reciprocal(ttnn::square(input, output_mem_config)), scalar, std::nullopt, output_mem_config)),
                std::nullopt,
                output_mem_config),
            t_nan,
            output_mem_config);
        if (scalar > 0) {
            result = where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                result,
                output_mem_config);
            result = where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
                -t_inf,
                result,
                output_mem_config);
        } else if (scalar < 0) {
            result = where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
                -t_inf,
                result,
                output_mem_config);
            result = where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                result,
                output_mem_config);
        }
        grad_tensor.emplace_back(result);
    } else {
        Tensor result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}
std::vector<Tensor> rdiv_bw(
    const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rdiv_bw)(grad, input, scalar, round_mode, output_mem_config);
}

std::vector<std::optional<Tensor>> _tanh_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    TT_FATAL(are_required_outputs.at(0), "input_grad derivative is required output");

    Tensor tanh_res = ttnn::tanh(queue_id, input, output_mem_config);
    tanh_res = ttnn::square(queue_id, tanh_res, output_mem_config);
    tanh_res = ttnn::rsub(queue_id, tanh_res, 1.0f, output_mem_config);
    if(input_grad.has_value()){
        ttnn::multiply(queue_id, grad, tanh_res, std::nullopt, output_mem_config, input_grad);
    } else {
    input_grad = ttnn::multiply(queue_id, grad, tanh_res, std::nullopt, output_mem_config);
    }
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}
std::vector<std::optional<Tensor>> tanh_bw(uint8_t queue_id,  const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    return operation::decorate_as_composite(__func__, _tanh_bw)(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}
std::vector<std::optional<Tensor>> tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _tanh_bw)(default_queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
}

std::vector<Tensor> _fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor val = grad;
    val = global_sum(val, output_mem_config);
    Tensor result = zeros_like(grad, output_mem_config);
    result = bcast(result, val, BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _fill_bw)(grad, output_mem_config);
}

std::vector<Tensor> _lt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> lt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lt_bw)(grad, output_mem_config);
}

std::vector<Tensor> _gt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> gt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _gt_bw)(grad, output_mem_config);
}

std::vector<Tensor> _ne_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> ne_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _ne_bw)(grad, output_mem_config);
}

std::vector<Tensor> _log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(grad, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        grad_a,
        output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _log_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _abs_bw)(grad, input, output_mem_config);
}

// bw(expm1) = grad * expm1(input) + 1
std::vector<Tensor> _expm1_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor eresult = ttnn::expm1(input, output_mem_config);
    Tensor rp1 = ttnn::add(eresult, 1.0f, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, rp1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> expm1_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _expm1_bw)(grad, input, output_mem_config);
}

// #  bw (exp2) = grad * exp2(input) * M_LN2
// # M_LN2 = 0.693147180559945309417
std::vector<Tensor> _exp2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = ttnn::exp2(input, output_mem_config);
    exp_result = ttnn::multiply(exp_result, M_LN2, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> exp2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _exp2_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _gelu_bw(
    const Tensor& grad, const Tensor& input, string approximate, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;

    if (approximate == "tanh") {
        float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
        float kKappa = 0.044715;
        Tensor x_sq = ttnn::multiply(input, input, std::nullopt, output_mem_config);
        Tensor x_cube = ttnn::multiply(x_sq, input, std::nullopt, output_mem_config);
        Tensor inner = ttnn::multiply(ttnn::add(input, ttnn::multiply(x_cube, kKappa, std::nullopt, output_mem_config)), kBeta, std::nullopt, output_mem_config);
        Tensor tanh_inner = ttnn::tanh(inner, output_mem_config);

        Tensor left = ttnn::multiply(input, 0.5, std::nullopt, output_mem_config);
        Tensor right = ttnn::add(tanh_inner, 1, std::nullopt, output_mem_config);

        Tensor left_derivative = ttnn::multiply(right, 0.5, std::nullopt, output_mem_config);

        Tensor tanh_derivative =
            ttnn::neg(ttnn::subtract(ttnn::multiply(tanh_inner, tanh_inner, std::nullopt, output_mem_config), 1, std::nullopt, output_mem_config),
                output_mem_config);
        Tensor inner_derivative = ttnn::multiply(
            (ttnn::add(
                ttnn::multiply(ttnn::multiply(x_sq, kKappa, std::nullopt, output_mem_config), 3, std::nullopt, output_mem_config), 1, std::nullopt, output_mem_config)), kBeta);
        Tensor right_derivative =
            ttnn::multiply(ttnn::multiply(tanh_derivative, left, std::nullopt, output_mem_config),
                inner_derivative,
                std::nullopt,
                output_mem_config);

        Tensor grad_a = ttnn::multiply(grad, (ttnn::add(left_derivative, right_derivative)), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(grad_a);
    } else {
        float kAlpha = M_SQRT1_2;
        float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
        Tensor cdf =
            ttnn::multiply((ttnn::add(ttnn::erf(ttnn::multiply(input, kAlpha, std::nullopt, output_mem_config)), 1, std::nullopt, output_mem_config)), 0.5);
        Tensor pdf = ttnn::multiply(ttnn::exp(ttnn::multiply(ttnn::multiply(input, input), -0.5), false, output_mem_config), kBeta, std::nullopt, output_mem_config);
        Tensor grad_a = ttnn::multiply(grad, (ttnn::add(cdf, ttnn::multiply(input, pdf))));
        grad_tensor.emplace_back(grad_a);
    }

    return grad_tensor;
}
std::vector<Tensor> gelu_bw(
    const Tensor& grad, const Tensor& input, string approximate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _gelu_bw)(grad, input, approximate, output_mem_config);
}

std::vector<Tensor> _bias_gelu_unary_bw(
    const Tensor& grad,
    const Tensor& input_tensor,
    float bias,
    string approximate,
    const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor input = ttnn::add(input_tensor, bias);

    grad_tensor = gelu_bw(grad, input, approximate = approximate);

    return grad_tensor;
}
std::vector<Tensor> bias_gelu_unary_bw(
    const Tensor& grad, const Tensor& input, float bias, string approximate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _bias_gelu_unary_bw)(
        grad, input, bias, approximate, output_mem_config);
}

std::vector<Tensor> _hardsigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = where(
        ttnn::logical_or(
            ttnn::le(input, -3, std::nullopt, output_mem_config),
            ttnn::ge(input, 3, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        zeros_like(input, output_mem_config),
        ttnn::multiply(grad, 1.0 / 6),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> hardsigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardsigmoid_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _hardshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor hardshrink_result = hardshrink(input_tensor, lambd, output_mem_config);
    Tensor result = where(ttnn::eqz(hardshrink_result, output_mem_config), 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> hardshrink_bw(
    const Tensor& grad, const Tensor& input, float lambd, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardshrink_bw)(grad, input, lambd, output_mem_config);
}

// softshrink
//  result: torch.where(self < -lambd, grad, torch.where(self > lambd, grad, torch.tensor(0.0)))
std::vector<Tensor> _softshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = where(
        ttnn::logical_or(
            ttnn::lt(input_tensor, full_like(input_tensor, -lambd, output_mem_config), std::nullopt, output_mem_config),
            ttnn::gt(input_tensor, full_like(input_tensor, lambd, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        zeros_like(grad, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> softshrink_bw(
    const Tensor& grad, const Tensor& input, float lambd, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softshrink_bw)(grad, input, lambd, output_mem_config);
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> _hardswish_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::lt(input, full_like(input, -3.0f), std::nullopt, output_mem_config),
        0.0,
        where(
            ttnn::le(input, full_like(input, 3.0f), std::nullopt, output_mem_config),
            ttnn::multiply(grad,
                ttnn::add(ttnn::multiply(input, 0.3333f, std::nullopt, output_mem_config), 0.5f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> hardswish_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardswish_bw)(grad, input, output_mem_config);
}

// Softplus
std::vector<Tensor> _softplus_bw(
    const Tensor& grad, const Tensor& input, float beta, float threshold, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor mul_input_beta = ttnn::multiply(input, beta, std::nullopt, output_mem_config);
    Tensor exp_beta_self = ttnn::exp(mul_input_beta, false, output_mem_config);
    Tensor sub_result = ttnn::add(mul_input_beta, -threshold, std::nullopt, output_mem_config);
    Tensor temp =
        ttnn::multiply(ttnn::multiply(grad, exp_beta_self, std::nullopt, output_mem_config),
            ttnn::reciprocal(ttnn::add(exp_beta_self, 1.0f, std::nullopt, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config);
    Tensor grad_result = where(ttnn::gtz(sub_result, output_mem_config), grad, temp, output_mem_config);
    mul_input_beta.deallocate();
    exp_beta_self.deallocate();
    sub_result.deallocate();
    temp.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> softplus_bw(
    const Tensor& grad, const Tensor& input, float beta, float threshold, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softplus_bw)(grad, input, beta, threshold, output_mem_config);
}

std::vector<Tensor> _polygamma_bw(
    const Tensor& grad, const Tensor& input, int n, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float pos_neg = 1.0f;
    if (n == 2 || n == 4 || n == 6 || n == 8 || n == 10) {
        pos_neg = -1.0f;
    }
    Tensor grad_a = ttnn::multiply(grad, polygamma(input, (n + 1), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::le(input, 0.0, std::nullopt, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        ttnn::multiply(
            full_like(input, -std::numeric_limits<float>::infinity(), output_mem_config), pos_neg, std::nullopt, output_mem_config),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        ttnn::multiply(
            full_like(input, std::numeric_limits<float>::infinity(), output_mem_config), pos_neg, std::nullopt, output_mem_config),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> polygamma_bw(
    const Tensor& grad, const Tensor& input, int n, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polygamma_bw)(grad, input, n, output_mem_config);
}

std::vector<Tensor> _atan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::SQUARE},
    UnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam{UnaryOpType::RECIP}};
    Tensor grad_a = ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> atan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _atan_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _atanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::SUB_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::RECIP}};
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);
    grad_a =
        where(ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)), 0, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::logical_or(
                ttnn::eq(input, 1, std::nullopt, output_mem_config),
                ttnn::eq(input, -1, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::nez(grad, output_mem_config)),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eq(grad_a, t_inf, std::nullopt, output_mem_config), ttnn::ltz(grad, output_mem_config)),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> atanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _atanh_bw)(grad, input, output_mem_config);
}

// Asin
// result: grad * (-self * self + 1).rsqrt()
std::vector<Tensor> _asin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam {UnaryOpType::SQUARE},
    UnaryWithParam {UnaryOpType::NEG},
    UnaryWithParam {UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam {UnaryOpType::RSQRT, true}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
    Tensor sub_one = ttnn::add(input, -1, std::nullopt, output_mem_config);
    Tensor sub_minus_one = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::ltz(sub_minus_one, output_mem_config),
        t_nan,
        where(
            ttnn::gtz(sub_one, output_mem_config),
            t_nan,
            where(
                ttnn::eqz(sub_minus_one, output_mem_config),
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                where(
                    ttnn::eqz(sub_one, output_mem_config),
                    ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    grad_result,
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> asin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _asin_bw)(grad, input, output_mem_config);
}

// Asinh
// result: grad * (self * self + 1).rsqrt()
std::vector<Tensor> _asinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::SQUARE},
    UnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam{UnaryOpType::RSQRT, true}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> asinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _asinh_bw)(grad, input, output_mem_config);
}

// name: cosh(Tensor self) -> Tensor
// self: grad * self.sinh()
std::vector<Tensor> _cosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor t_neg_inf =
        ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, full_like(input, 88.50, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, full_like(input, -88.50, output_mem_config), std::nullopt, output_mem_config),
            t_neg_inf,
            ttnn::multiply(grad, sinh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_neg_inf.deallocate();
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> cosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _cosh_bw)(grad, input, output_mem_config);
}

// name: cos(Tensor self) -> Tensor
// self: grad * -self.sin()
std::vector<Tensor> _cos_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result =
        ttnn::multiply(grad, (ttnn::neg(ttnn::sin(input_tensor, output_mem_config), output_mem_config)), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> cos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _cos_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _acosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_rsqrt = ttnn::square(input, output_mem_config);
    in_rsqrt = ttnn::rsqrt(ttnn::subtract(in_rsqrt, 1.0, std::nullopt, output_mem_config), true, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor cond_result = ttnn::logical_or(
        ttnn::lt(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config),
        ttnn::gt(input, full_like(input, 1.0, output_mem_config), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(ttnn::eqz(cond_result, output_mem_config), t_nan, grad_a, output_mem_config);
    cond_result = ttnn::logical_or(
        ttnn::eq(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config),
        ttnn::eq(input, full_like(input, 1.0, output_mem_config), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::eq(cond_result, ones_like(input, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> acosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _acosh_bw)(grad, input, output_mem_config);
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> _acos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = ttnn::neg(input, output_mem_config);
    Tensor in_rsqrt =
        ttnn::rsqrt(ttnn::add(ttnn::multiply(neg_in, input, std::nullopt, output_mem_config), 1.0f, std::nullopt, output_mem_config), true, output_mem_config);
    in_rsqrt = ttnn::neg(in_rsqrt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor neg_one = full_like(input, -1.0, output_mem_config);
    Tensor pos_one = full_like(input, 1.0, output_mem_config);
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_or(
            ttnn::lt(input, neg_one, std::nullopt, output_mem_config),
            ttnn::gt(input, pos_one, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::eq(input, neg_one, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, pos_one, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> acos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _acos_bw)(grad, input, output_mem_config);
}

// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> _leaky_relu_bw(
    const Tensor& grad, const Tensor& input, float negative_slope, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config), grad, ttnn::multiply(grad, negative_slope, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> leaky_relu_bw(
    const Tensor& grad, const Tensor& input, float negative_slope, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _leaky_relu_bw)(grad, input, negative_slope, output_mem_config);
}

// ELU
// result : grad * (torch.where(input >= 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> _elu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gez(input, output_mem_config),
        grad,
        ttnn::multiply(grad, ttnn::multiply(ttnn::exp(input, false, output_mem_config), alpha, std::nullopt, output_mem_config), std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> elu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _elu_bw)(grad, input, alpha, output_mem_config);
}

// Hardtanh
// result: torch.where((input <= min) | (input >= max), 0.0, grad)
std::vector<Tensor> _hardtanh_bw(
    const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::le(input, full_like(input, min), std::nullopt, output_mem_config),
        0.0,
        where(ttnn::ge(input, full_like(input, max), std::nullopt, output_mem_config), 0.0, grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> hardtanh_bw(
    const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh_bw)(grad, input, min, max, output_mem_config);
}

// name: sin(Tensor self) -> Tensor
// self: grad * self.cos()
std::vector<Tensor> _sin_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input = ttnn::multiply(grad, ttnn::cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}
std::vector<Tensor> sin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sin_bw)(grad, input, output_mem_config);
}

// name: sinh(Tensor self) -> Tensor
// self: grad * self.cosh()
std::vector<Tensor> _sinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, full_like(input, 88.5, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, full_like(input, -88.5, output_mem_config), std::nullopt, output_mem_config),
            t_inf,
            ttnn::multiply(grad, cosh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> sinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sinh_bw)(grad, input, output_mem_config);
}

// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> _celu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor div_result = ttnn::multiply(
        input, ttnn::reciprocal(full_like(input, alpha, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    Tensor exp_result = ttnn::exp(div_result, false, output_mem_config);
    Tensor grad_result = where(
        ttnn::gt(input, zeros_like(input, output_mem_config), std::nullopt, output_mem_config),
        grad,
        ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> celu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _celu_bw)(grad, input, alpha, output_mem_config);
}


// erfinv
// self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
// for input -1 and 1: grad.sign() * inf, for input > 1 or < -1 : nan
std::vector<Tensor> _erfinv_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::sqrt(full_like(input, M_PI, output_mem_config), output_mem_config),
            ttnn::multiply(ttnn::exp(ttnn::square(ttnn::erfinv(input, output_mem_config), output_mem_config), false, output_mem_config),
                grad,
                std::nullopt,
                output_mem_config),
            std::nullopt,
            output_mem_config),
        0.5,
        std::nullopt,
        output_mem_config);
    Tensor neg_one = full_like(input, -1.0, output_mem_config);
    Tensor pos_one = full_like(input, 1.0, output_mem_config);
    Tensor t_inf = ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    result = where(
        ttnn::logical_or(
            ttnn::lt(input, neg_one, std::nullopt, output_mem_config),
            ttnn::gt(input, pos_one, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        result,
        output_mem_config);
    result = where(
        ttnn::eq(input, neg_one, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, pos_one, std::nullopt, output_mem_config), t_inf, result, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erfinv_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _erfinv_bw)(grad, input, output_mem_config);
}

// bw(log10(in)) = grad/(in * 2.30258509299404568402)
std::vector<Tensor> _log10_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN10, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log10_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _log10_bw)(grad, input, output_mem_config);
}

// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> _log1p_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor t_inp1 = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::eq(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(t_inp1, output_mem_config), ttnn::eqz(grad, output_mem_config)),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log1p_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _log1p_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _erf_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erf_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _erf_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _erfc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        -M_2_SQRTPI,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erfc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _erfc_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _digamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor grad_a = ttnn::multiply(grad, polygamma(input, 1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> digamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _digamma_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _deg2rad_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_PI_180 = M_PI / 180;
    Tensor grad_result = ttnn::multiply(grad, M_PI_180, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> deg2rad_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _deg2rad_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rad2deg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180 / M_PI;
    Tensor grad_result = ttnn::multiply(grad, M_180_PI, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> rad2deg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rad2deg_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _reciprocal_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::neg(ttnn::sign(grad, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        ttnn::multiply(ttnn::neg(grad, output_mem_config),
            ttnn::reciprocal(ttnn::square(input, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> reciprocal_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _reciprocal_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_tensor = zeros_like(input, output_mem_config);
    Tensor one_tensor = ones_like(input, output_mem_config);
    Tensor six_tensor = full_like(input, 6, output_mem_config);
    Tensor grad_result =
        where(ttnn::le(input, zero_tensor, std::nullopt, output_mem_config), zero_tensor, six_tensor, output_mem_config);
    grad_result = where(
        ttnn::logical_and(
            ttnn::gtz(input, output_mem_config),
            ttnn::lt(input, six_tensor, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        grad_result,
        output_mem_config);
    grad_result =
        where(ttnn::ge(input, six_tensor, std::nullopt, output_mem_config), zero_tensor, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> relu6_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _relu6_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rpow_bw(
    const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    Tensor grad_result = zeros_like(input, output_mem_config);
    if (exponent != 0.0) {
        grad_result =
            ttnn::multiply(grad,
                ttnn::multiply(pow(input, exponent - 1, output_mem_config), exponent, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config);
        grad_result = where(ttnn::ltz(input, output_mem_config), t_nan, grad_result, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> rpow_bw(
    const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rpow_bw)(grad, input, exponent, output_mem_config);
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<Tensor> _silu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_sigmoid = ttnn::multiply(grad, ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor add_sub = ttnn::add(
        ttnn::multiply(ttnn::subtract(ttnn::full_like(input, 1.0f) , ttnn::sigmoid(input, output_mem_config), std::nullopt, output_mem_config),
            input,
            std::nullopt,
            output_mem_config),
        1.0f,
        std::nullopt,
        output_mem_config);
    Tensor grad_result = ttnn::multiply(grad_sigmoid, add_sub, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> silu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _silu_bw)(grad, input, output_mem_config);
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = ttnn::multiply(grad, 1.0507f, std::nullopt, output_mem_config);
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad_lambd,
        ttnn::multiply(ttnn::multiply(grad_lambd, 1.673260f, std::nullopt, output_mem_config),
            ttnn::exp(input, false, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> selu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _selu_bw)(grad, input, output_mem_config);
}


// Autoformat support
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config) {
    auto formatted_input_tensor = temp;
    if(formatted_input_tensor.get_layout()==Layout::ROW_MAJOR){
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_legacy_shape(), false, false, true, true);
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return formatted_input_tensor;
}

// Prod
// along a single dimension --> result: grad_data * (y / input )
std::vector<Tensor> _prod_bw(
    const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor prod_result = prod(input, all_dimensions, dim, output_mem_config);
    if(prod_result.get_layout()==Layout::ROW_MAJOR && prod_result.storage_type() == StorageType::DEVICE){
        prod_result = tt::tt_metal::change_layout_to_tile(prod_result, output_mem_config);
        }
    if (all_dimensions == true) {
        Tensor temp =
            ttnn::multiply(prod_result, grad, std::nullopt, output_mem_config);  // result is stored in the first position
        Tensor fill_tensor = tt::numpy::fill_first_val_into_tensor<bfloat16>(
            temp, temp.get_dtype(), temp.get_layout(), temp.device(), output_mem_config);
        Tensor all_dimension_result =
            ttnn::multiply(ttnn::reciprocal(input, output_mem_config), fill_tensor, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(all_dimension_result);
        return grad_tensor;
    }
    // all_dimensions = False
    Tensor updated_grad = prod_result;
    if (prod_result.get_legacy_shape() != grad.get_legacy_shape()) {
        if (dim == 3 || dim == -1) {
            std::vector<int64_t> after_permute_dims = {0, 3, 1, 2};
            Tensor required = permute(grad, after_permute_dims, output_mem_config);
            const Shape start_index = {0, 0, 0, 0};
            const Shape end_index = {
                grad.get_legacy_shape()[0] - 1, 0, grad.get_legacy_shape()[1] - 1, grad.get_legacy_shape()[2] - 1};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, std::nullopt);
            after_permute_dims = {0, 2, 3, 1};
            updated_grad = permute(new_slice_tensor, after_permute_dims, output_mem_config);
            Tensor pad_updated_grad = updated_grad.pad_to_tile(1.0f);
            Tensor pad_prod_result = prod_result.pad_to_tile(1.0f);
            pad_updated_grad = pad_updated_grad.to(Layout::TILE);
            pad_prod_result = pad_prod_result.to(Layout::TILE);
            updated_grad = pad_updated_grad.to(input.device());
            prod_result = pad_prod_result.to(input.device());
            pad_updated_grad.deallocate();
            pad_prod_result.deallocate();
        } else if (dim == 2 || dim == -2) {
            std::vector<int64_t> after_permute_dims = {0, 2, 1, 3};
            Tensor required = permute(grad, after_permute_dims, output_mem_config);
            const Shape start_index = {0, 0, 0, 0};
            const Shape end_index = {
                grad.get_legacy_shape()[0] - 1, 0, grad.get_legacy_shape()[1] - 1, grad.get_legacy_shape()[3] - 1};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, std::nullopt);
            updated_grad = permute(new_slice_tensor, after_permute_dims, output_mem_config);
            if(updated_grad.get_layout()==Layout::ROW_MAJOR){
                updated_grad = tt::tt_metal::change_layout_to_tile(updated_grad, output_mem_config);
            }
        }
    }
    Tensor reciprocal_input = ttnn::reciprocal(input, output_mem_config);
    Tensor temp = ttnn::multiply(prod_result, (dim == 1 || dim == 0 || dim == -4 || dim == -3) ? grad : updated_grad, std::nullopt, output_mem_config);
    if(temp.get_layout()==Layout::ROW_MAJOR){
        temp = tt::tt_metal::change_layout_to_tile(temp, output_mem_config);
    }
    if (dim == 3 || dim == -1) {
        Tensor grad_result = bcast(reciprocal_input, temp, BcastOpMath::MUL, BcastOpDim::W, output_mem_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 2 || dim == -2) {
        Tensor grad_result = bcast(reciprocal_input, temp, BcastOpMath::MUL, BcastOpDim::H, output_mem_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 1 || dim == -3) {
        Tensor tensor_1_temp = reciprocal_input;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, 0},
                          {0, 32 - (reciprocal_input.get_legacy_shape()[1] % 32)},
                          {0, 0},
                          {0, 0}};
            tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, true, std::nullopt);
        }
        std::vector<int64_t> after_permute_dims = {0, 2, 3, 1};
        Tensor tensor_1 = permute(tensor_1_temp, after_permute_dims, output_mem_config);
        Tensor tensor_2 = permute(temp, after_permute_dims, output_mem_config);

        // put the tensor back on device because permute throws it off device
        // See: Remove auto format within permute_op.cpp #9404
        tensor_2 = AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

        after_permute_dims = {0, 3, 1, 2};
        Tensor result = permute(
            bcast(tensor_1, tensor_2, BcastOpMath::MUL, BcastOpDim::W, output_mem_config),
            after_permute_dims,
            output_mem_config);
        Tensor grad_result = result;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            const Shape start_index = {0, 0, 0, 0};
            const Shape end_index = {
                input.get_legacy_shape()[0] - 1,
                input.get_legacy_shape()[1] - 1,
                input.get_legacy_shape()[2] - 1,
                input.get_legacy_shape()[3] - 1};
            grad_result = ttnn::slice(result, start_index, end_index, std::nullopt);
        }
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    // dim 0
    Tensor tensor_1_temp = reciprocal_input;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, (32 - (reciprocal_input.get_legacy_shape()[0] % 32))},
                      {0, 0},
                      {0, 0},
                      {0, 0}};
        tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, false, std::nullopt);
    }
    std::vector<int64_t> after_permute_dims = {3, 1, 2, 0};
    Tensor tensor_1 = permute(tensor_1_temp, after_permute_dims, output_mem_config);
    Tensor tensor_2 = permute(temp, after_permute_dims, output_mem_config);

    // put the tensor back on device because permute throws it off device
    // See: Remove auto format within permute_op.cpp #9404
    tensor_2 = AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

    Tensor result = permute(
        bcast(tensor_1, tensor_2, BcastOpMath::MUL, BcastOpDim::W, output_mem_config),
        after_permute_dims,
        output_mem_config);
    Tensor grad_result = result;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        const Shape start_index = {0, 0, 0, 0};
        const Shape end_index = {
            input.get_legacy_shape()[0] - 1,
            input.get_legacy_shape()[1] - 1,
            input.get_legacy_shape()[2] - 1,
            input.get_legacy_shape()[3] - 1};
        grad_result = ttnn::slice(result, start_index, end_index, std::nullopt);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> prod_bw(
    const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _prod_bw)(grad, input, all_dimensions, dim, output_mem_config);
}

// square
// result:  2 * input * grad_data
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(ttnn::multiply(grad, 2.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> square_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _square_bw)(grad, input, output_mem_config);
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> _tanhshrink_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> tanhshrink_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _tanhshrink_bw)(grad, input, output_mem_config);
}

// threshold
// if input <= threshold = 0 else grad
std::vector<Tensor> _threshold_bw(
    const Tensor& grad, const Tensor& input, float threshold, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = where(
        ttnn::gtz(ttnn::add(input, -threshold, std::nullopt, output_mem_config), output_mem_config),
        grad,
        zeros_like(input, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> threshold_bw(
    const Tensor& grad, const Tensor& input, float threshold, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _threshold_bw)(grad, input, threshold, value, output_mem_config);
}

// Torch reference
// # if eps is not None:
// #         lo = eps
// #         hi = 1.0 - lo
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= lo, self <= hi),
// #             grad_output / (self * (1.0 - self)),
// #             0.0,
// #         )
// #     else:
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= 0.0, self <= 1.0),
// #             grad_output / (self * (1.0 - self)),
// #             self.new_full((), float("nan")),
// #         )
std::vector<Tensor> _logiteps_bw(
    const Tensor& grad, const Tensor& input, float eps, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float low, high;
    low = eps;
    high = 1.0 - low;
    Tensor grad_result =
        ttnn::multiply(grad,
            ttnn::reciprocal(ttnn::multiply(input, ttnn::rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)),
            std::nullopt,
            output_mem_config);
    Tensor t_eps = full_like(input, eps, output_mem_config);
    Tensor t_low = full_like(input, low, output_mem_config);
    Tensor t_high = full_like(input, high, output_mem_config);
    Tensor ltl_gth = ttnn::logical_or(
        ttnn::lt(input, t_low, std::nullopt, output_mem_config),
        ttnn::gt(input, t_high, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(ltl_gth, ones_like(input, output_mem_config), std::nullopt, output_mem_config),
        where(ttnn::ltz(t_eps, output_mem_config), std::nanf(" "), 0.0, output_mem_config),
        where(
            ttnn::logical_or(
                ttnn::eq(input, 0.0, std::nullopt, output_mem_config),
                ttnn::eq(input, 1.0, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config),
            grad_result,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> logiteps_bw(
    const Tensor& grad, const Tensor& input, float eps, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logiteps_bw)(grad, input, eps, output_mem_config);
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> _softsign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::ABS},
    UnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
    UnaryWithParam{UnaryOpType::SQUARE},
    UnaryWithParam{UnaryOpType::RECIP}};
    grad_tensor.emplace_back(
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> softsign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softsign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}
std::vector<Tensor> sign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _ceil_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}
std::vector<Tensor> ceil_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _ceil_bw)(grad, input, output_mem_config);
}

// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> _log2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad, ttnn::reciprocal(ttnn::multiply(input, M_LN2, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _log2_bw)(grad, input, output_mem_config);
}
std::vector<Tensor> _ge_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> ge_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _ge_bw)(grad, output_mem_config);
}

std::vector<Tensor> _le_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> le_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _le_bw)(grad, output_mem_config);
}

std::vector<Tensor> _unary_fmod_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_fmod_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_fmod_bw)(grad, input, scalar, output_mem_config);
}

std::vector<Tensor> _unary_remainder_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_remainder_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_remainder_bw)(grad, input, scalar, output_mem_config);
}

#define CHECK_FOR_COMPLEX(input)                                                     \
    do {                                                                             \
        TT_ASSERT(utility::is_complex_shape(input), "works for complex shape only"); \
        /* TT_ASSERT( input.shape()[0] == 1, "tensor should have batch size 1"); */  \
    } while (0);

// complex conj
// self: grad.conj()
std::vector<Tensor> _conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _conj_bw)(grad, input, output_mem_config);
}

// complex reciprocal
// self: -grad * (result * result).conj()
std::vector<Tensor> _complex_recip_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor input_r = real(input, output_mem_config);
    Tensor input_i = imag(input, output_mem_config);
    Tensor condition_nan =
        ttnn::logical_and(ttnn::eqz(input_r, output_mem_config), ttnn::eqz(input_i, output_mem_config), std::nullopt, output_mem_config);
    input_r.deallocate();
    input_i.deallocate();
    Tensor nan_flag = mk_complex(condition_nan, condition_nan, output_mem_config);
    condition_nan.deallocate();
    Tensor grad_result = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_mul(
            ttnn::neg(grad, output_mem_config),
            conj(
                complex_mul(
                    complex_recip(input, output_mem_config),
                    complex_recip(input, output_mem_config),
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    nan_flag.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> complex_recip_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_recip_bw)(grad, input, output_mem_config);
}

// complex imag
// imag: at::imag(grad)
std::vector<Tensor> _imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        mk_complex(zeros_like(real(input, output_mem_config), output_mem_config), grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _imag_bw)(grad, input, output_mem_config);
}

// complex real
// real: at::real(grad)
std::vector<Tensor> _real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        mk_complex(grad, zeros_like(imag(input, output_mem_config), output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _real_bw)(grad, input, output_mem_config);
}

// angle at::where(self == 0.0, at::zeros({}, self.options()), grad * self / self.abs().pow(2)
std::vector<Tensor> _angle_bw(
    const Tensor& grad, const Tensor& input, bool is_complextensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    if (is_complextensor) {
        CHECK_FOR_COMPLEX(input);
        Tensor inp_r = real(input, output_mem_config);
        Tensor inp_i = imag(input, output_mem_config);
        Tensor condition_zero =
            ttnn::logical_and(ttnn::eqz(inp_r, output_mem_config), ttnn::eqz(inp_i, output_mem_config), std::nullopt, output_mem_config);
        Tensor abs_squared = ttnn::reciprocal(
            ttnn::add(ttnn::square(inp_r, output_mem_config), ttnn::square(inp_i, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);
        Tensor real = where(
            condition_zero,
            zeros_like(inp_r, output_mem_config),
            ttnn::multiply(grad,
                ttnn::multiply(ttnn::neg(inp_i, output_mem_config), abs_squared, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            output_mem_config);
        Tensor imag = where(
            condition_zero,
            zeros_like(inp_i, output_mem_config),
            ttnn::multiply(grad, ttnn::multiply(inp_r, abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);
        condition_zero.deallocate();
        abs_squared.deallocate();
        inp_r.deallocate();
        inp_i.deallocate();
        Tensor grad_result = mk_complex(real, imag, output_mem_config);
        real.deallocate();
        imag.deallocate();
        grad_tensor.emplace_back(grad_result);
    } else {
        Tensor grad_result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(grad_result);
    }
    return grad_tensor;
}
std::vector<Tensor> angle_bw(
    const Tensor& grad, const Tensor& input, bool is_complextensor, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _angle_bw)(grad, input, is_complextensor, output_mem_config);
}

// complex abs
// self: grad * self.sgn()
std::vector<Tensor> _complex_abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor result = complex_abs(input, output_mem_config);
    result = mk_complex(result, result, output_mem_config);
    Tensor grad_c = mk_complex(grad, grad, output_mem_config);
    Tensor grad_result = where(
        ttnn::eqz(result, output_mem_config),
        zeros_like(result, output_mem_config),
        ttnn::multiply(grad_c,
            ttnn::multiply(input, ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> complex_abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_abs_bw)(grad, input, output_mem_config);
}
// polar
// grad_abs = torch.real(grad_conj * torch.sgn(result))
// result_mul_1_j = result * torch.tensor(0.0 + 1.0j)
// grad_angle = torch.real(grad_conj * result_mul_1_j)
// polar fwd op uses sin and cos hence input_b range is (0, 2*pi)
std::vector<Tensor> _polar_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor result = polar(input_a, input_b, output_mem_config);
    Tensor abs_result = complex_abs(result, output_mem_config);
    abs_result = mk_complex(abs_result, abs_result, output_mem_config);
    Tensor sgn_result = where(
        ttnn::eqz(abs_result, output_mem_config),
        zeros_like(result, output_mem_config),
        ttnn::multiply(result, ttnn::reciprocal(abs_result, output_mem_config), std::nullopt, output_mem_config),
        output_mem_config);
    abs_result.deallocate();
    Tensor grad_abs =
        real(complex_mul(conj(grad, output_mem_config), sgn_result, output_mem_config), output_mem_config);
    sgn_result.deallocate();
    Tensor flip_tensor = mk_complex(
        zeros_like(input_a, output_mem_config), full_like(input_b, 1.0, output_mem_config), output_mem_config);
    Tensor grad_angle = real(
        complex_mul(
            conj(grad, output_mem_config), complex_mul(result, flip_tensor, output_mem_config), output_mem_config),
        output_mem_config);
    result.deallocate();
    flip_tensor.deallocate();
    Tensor grad_result = mk_complex(grad_abs, grad_angle, output_mem_config);
    grad_abs.deallocate();
    grad_angle.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> polar_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polar_bw)(grad, input_a, input_b, output_mem_config);
}

// complex div
//  self: grad / other.conj();
//  other: -grad * ((self / other) / other).conj();
std::vector<Tensor> _complex_div_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor other_r = real(other, output_mem_config);
    Tensor other_i = imag(other, output_mem_config);
    Tensor condition_nan =
        ttnn::logical_and(ttnn::eqz(other_r, output_mem_config), ttnn::eqz(other_i, output_mem_config), std::nullopt, output_mem_config);
    other_r.deallocate();
    other_i.deallocate();
    Tensor nan_flag = mk_complex(condition_nan, condition_nan, output_mem_config);
    condition_nan.deallocate();
    Tensor grad_a = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_div(grad, conj(other, output_mem_config), output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor result = complex_div(input, other, output_mem_config);
    Tensor grad_b = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_mul(
            ttnn::neg(grad, output_mem_config),
            conj(complex_div(result, other, output_mem_config), output_mem_config),
            output_mem_config),
        output_mem_config);
    result.deallocate();
    nan_flag.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_div_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_div_bw)(grad, input, other, output_mem_config);
}

// complex mul
// grad_input = grad * other.conj()
// grad_other = grad * input.conj()
std::vector<Tensor> _complex_mul_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = complex_mul(grad, conj(other, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = complex_mul(grad, conj(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_mul_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_mul_bw)(grad, input, other, output_mem_config);
}

// complex add
// self: grad, other: grad * alpha
std::vector<Tensor> _complex_add_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(grad, alpha, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_add_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_add_bw)(grad, input, other, alpha, output_mem_config);
}

// complex sub
// self: grad, other: -grad * alpha
std::vector<Tensor> _complex_sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::NEG},
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha}};
    Tensor grad_b = ttnn::unary_chain(grad, ops_chain, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_sub_bw)(grad, input, other, alpha, output_mem_config);
}
#undef CHECK_FOR_COMPLEX

// Repeat Backward
std::vector<Tensor> _repeat_bw(
    const Tensor& grad, const Tensor& input, const Shape& shape, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto shape_wh = input.get_legacy_shape();
    TT_FATAL(shape_wh[0] == 1 && "input shape[0] should be 1");
    // input.get_legacy_shape()[0]
    // If repeat shape has 0's, it returns zeros of given input
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
        Tensor zero_tensor = zeros_like(input, output_mem_config);
        grad_tensor.emplace_back(zero_tensor);
        return grad_tensor;
    } else if (shape[0] > 1) {
        std::vector<int64_t> dim = {0};
        TT_FATAL(shape[1] == 1 && shape[2] == 1 && shape[3] == 1 && "repeat[1], [2], [3] should be 1");
        Shape required = {1, shape_wh[1], shape_wh[2], shape_wh[3]};
        Tensor result = tt::operations::primary::moreh_sum(
            grad,
            dim,
            true,
            zeros(required, input.get_dtype(), input.get_layout(), input.device(), output_mem_config),
            output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    } else if (shape[1] > 1) {
        std::vector<int64_t> dim = {1};
        TT_FATAL(shape[0] == 1 && shape[2] == 1 && shape[3] == 1 && "repeat[0], [2], [3] should be 1");
        Shape required = {shape_wh[0], 1, shape_wh[2], shape_wh[3]};
        Tensor result = tt::operations::primary::moreh_sum(
            grad,
            dim,
            true,
            zeros(required, input.get_dtype(), input.get_layout(), input.device(), output_mem_config),
            output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    return grad_tensor;
}
std::vector<Tensor> repeat_bw(
    const Tensor& grad, const Tensor& input, const Shape& shape, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _repeat_bw)(grad, input, shape, output_mem_config);
}

std::vector<Tensor> _floor_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> floor_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _floor_bw)(grad, output_mem_config);
}

std::vector<Tensor> _round_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> round_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _round_bw)(grad, output_mem_config);
}

std::vector<Tensor> _unary_div_no_nan_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zeros = zeros_like(grad, output_mem_config);
    Tensor val = full_like(input, scalar, output_mem_config);
    Tensor result = where(
        ttnn::eq(val, 0, std::nullopt, output_mem_config), zeros, ttnn::multiply(grad, 1 / scalar, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_div_no_nan_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_div_no_nan_bw)(grad, input, scalar, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt
