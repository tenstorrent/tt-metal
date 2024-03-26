// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/complex/complex_ops.hpp"

namespace tt {

namespace tt_metal {


std::vector<Tensor> _addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = mul_unary(grad, alpha, output_mem_config);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addalpha_bw)(grad, input, other, alpha, output_mem_config);
}

std::vector<Tensor> add_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addalpha_bw)(grad, input, other, 1, output_mem_config);
}

std::vector<Tensor> _unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(grad, scalar, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_mul_bw)(grad, input, scalar, output_mem_config);
}

std::vector<Tensor> _unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    const float ZERO_THRESHOLD = std::numeric_limits<float>::epsilon()*10.0f;
    TT_FATAL(exponent >= 0.0, "negative exponents are not supported; use recip(pow(input,abs(exponent)))");
    if ( std::abs(exponent) < ZERO_THRESHOLD ) {
        grad_tensor.emplace_back( zeros_like( input, output_mem_config) );
        return grad_tensor;
    }

    Tensor power_input = power(input, fabs(exponent - 1.0f), output_mem_config);
    if ( exponent < 1.0f ) {
        power_input = recip(power_input,output_mem_config);
    }

    Tensor result = mul_unary(power_input, exponent, output_mem_config);
    Tensor final_result = mul(result, grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(final_result);
    return grad_tensor;
}
std::vector<Tensor> unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_pow_bw)(grad, input, exponent, output_mem_config);
}

std::vector<Tensor> _unary_add_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_add_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_add_bw)(grad, input, alpha, output_mem_config);
}


std::vector<Tensor> _mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, input_b, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul(grad, input_a, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _mul_bw)(grad, input_a, input_b, output_mem_config);
}


std::vector<Tensor> _exp_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = exp(input, output_mem_config);
    Tensor result = mul(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> exp_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _exp_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_a = mul_unary(mul(grad, tensor2, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul_unary(mul(grad, tensor1, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcmul_bw)(grad, input, tensor1, tensor2, value, output_mem_config);
}


std::vector<Tensor> _unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_assign_bw)(grad, input, output_mem_config);
}
std::vector<Tensor> binary_assign_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_assign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor sqrt_result = sqrt(input, output_mem_config);
    Tensor result = mul(grad, recip(mul_unary(sqrt_result, 2.0, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    float t_nan  = std::nanf("");
    result = where(ltz(input, output_mem_config), t_nan, result, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> sqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sqrt_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float inv_scalar = 1.0f/scalar;
    if (round_mode=="None"){
        Tensor result = mul_unary(grad, inv_scalar, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    else{
        Tensor result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}
std::vector<Tensor> unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_div_bw)(grad, input, scalar, round_mode, output_mem_config);
}


std::vector<Tensor> _div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, string round_mode, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    if (round_mode=="None"){
        Tensor grad_a = mul(grad, recip(other, output_mem_config), std::nullopt, output_mem_config);
        Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
        Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
        grad_tensor.emplace_back( where(eqz(other, output_mem_config),
                                        where(eqz(grad, output_mem_config),
                                            t_nan,
                                            mul(t_inf, sign(grad, output_mem_config), std::nullopt, output_mem_config), output_mem_config),
                                        grad_a, output_mem_config));
        Tensor grad_b = mul(neg(grad, output_mem_config) , (mul(input, recip(square(other, output_mem_config), output_mem_config), std::nullopt, output_mem_config)), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(where(eqz(other, output_mem_config),
                                    where(eqz(grad, output_mem_config),
                                        t_nan,
                                        where(eqz(input, output_mem_config),
                                            t_nan,
                                            mul( mul( neg(t_inf, output_mem_config), sign(input, output_mem_config), std::nullopt, output_mem_config), sign(grad, output_mem_config), std::nullopt, output_mem_config), output_mem_config), output_mem_config),
                                    grad_b, output_mem_config));
    } else{
        Tensor grad_a = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(grad_a);
        Tensor grad_b = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(grad_b);
    }

    return grad_tensor;
}
std::vector<Tensor> div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, string round_mode, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _div_bw)(grad, input, other, round_mode, output_mem_config);
}

std::vector<Tensor> _rdiv_bw(const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    if (round_mode=="None"){
        Tensor result = mul(neg(grad, output_mem_config) , (mul_unary(recip(square(input, output_mem_config)), scalar, output_mem_config)), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    else{
        Tensor result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}
std::vector<Tensor> rdiv_bw(const Tensor& grad, const Tensor& input, float scalar, string round_mode, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _rdiv_bw)(grad, input, scalar, round_mode, output_mem_config);
}

std::vector<Tensor> _tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = tanh(input, output_mem_config);
    tanh_res = square(tanh_res, output_mem_config);
    tanh_res = rsub(tanh_res, 1.0, output_mem_config);
    Tensor result = mul(grad, tanh_res, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _tanh_bw)(grad, input, output_mem_config);
}

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> _sigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    std::vector<Tensor> grad_tensor;
    Tensor sig_result = sigmoid(input, output_mem_config);
    Tensor rsub_term = rsub(sig_result, 1.0f, output_mem_config);
    Tensor prod_term_1 = mul(sig_result, rsub_term,{},output_mem_config);
    Tensor prod_term_2 = mul(prod_term_1, grad,{},output_mem_config);
    grad_tensor.emplace_back(prod_term_2);
    return grad_tensor;
}

std::vector<Tensor> sigmoid_bw(const Tensor& grad, const Tensor& input,
                                const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sigmoid_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _tan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tan_result = tan(input, output_mem_config);
    Tensor result = mul(grad, add1(square(tan_result, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> tan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _tan_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _addcdiv_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
    Tensor grad_a = mul(mul_unary(grad, value, output_mem_config), recip(tensor2, output_mem_config));
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, t_inf, output_mem_config),
        grad_a,
        output_mem_config));
    Tensor tmp = mul(
        mul_unary(neg(grad, output_mem_config), value, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b = mul(tmp, recip(square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, neg(t_inf, output_mem_config), output_mem_config),
        grad_b,
        output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> addcdiv_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcdiv_bw)(grad, input, tensor1, tensor2, value, output_mem_config);
}

std::vector<Tensor> _where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = where(condition, grad, 0.0f, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = where(condition, 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _where_bw)(grad, condition, input, other, output_mem_config);
}

//template parameter min_or_max = TRUE for MAX, FALSE for MIN
template<bool min_or_max>
std::vector<Tensor> _min_or_max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    Tensor zeros_t = zeros_like(input, output_mem_config);
    std::vector<Tensor> grad_tensor;
    Tensor t_scale_grad = mul_unary(grad, 0.5, output_mem_config);
    Tensor t_sub = sub(other, input, std::nullopt, output_mem_config);
    Tensor t_sub_gtz = gtz(t_sub,output_mem_config);
    Tensor t_sub_eqz = eqz(t_sub,output_mem_config);
    Tensor t_sub_ltz = ltz(t_sub,output_mem_config);
    Tensor grad_other = add(mul(t_sub_ltz, grad,{},output_mem_config),mul(t_sub_eqz, t_scale_grad,{},output_mem_config), std::nullopt, output_mem_config);
    Tensor grad_input = add(mul(t_sub_gtz, grad,{},output_mem_config),mul(t_sub_eqz, t_scale_grad,{},output_mem_config), std::nullopt, output_mem_config);

    if (min_or_max) {
        //MAX
        grad_tensor.emplace_back(grad_other);
        grad_tensor.emplace_back(grad_input);
    } else {
        //MIN
        grad_tensor.emplace_back(grad_input);
        grad_tensor.emplace_back(grad_other);
    }
    return grad_tensor;
}
auto _max_bw = _min_or_max_bw<true>;
auto _min_bw = _min_or_max_bw<false>;

std::vector<Tensor> max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _max_bw)(grad, input, other, output_mem_config);
}

std::vector<Tensor> min_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _min_bw)(grad, input, other, output_mem_config);
}


std::vector<Tensor> _fill_zero_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> fill_zero_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _fill_zero_bw)(grad, output_mem_config);
}

std::vector<Tensor> _fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor val = grad;
    val = global_sum(val, output_mem_config);
    Tensor result = zeros_like(grad, output_mem_config);
    result = bcast(result, val,  BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _fill_bw)(grad, output_mem_config);
}

std::vector<Tensor> _embedding_bw(const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config) {
    TT_FATAL(input.get_dtype() == DataType::UINT32, "Input must be UINT32");
    TT_FATAL(grad.get_legacy_shape()[0] == 1 && grad.get_legacy_shape()[1] == 1, "First two dimensions for the grad must be 1");
    TT_FATAL(input.get_legacy_shape()[1] == 1 && input.get_legacy_shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = embeddings(input, grad, false);
    grad_tensor.emplace_back(grad_a);

    return grad_tensor;
}
std::vector<Tensor> embedding_bw(const Tensor& grad, const Tensor& input, const Tensor& weight,  const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _embedding_bw)(grad, input, weight, output_mem_config);
}

// - name: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
//   self: grad
//   other: -grad * alpha

std::vector<Tensor> _subalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = mul_unary(neg(grad, output_mem_config), alpha, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> subalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _subalpha_bw)(grad, input, other, alpha, output_mem_config);
}

std::vector<Tensor> sub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _subalpha_bw)(grad, input, other, 1.0, output_mem_config);
}

// - name: sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
//   self: grad
std::vector<Tensor> _unary_sub_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_sub_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_sub_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _neg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = neg(grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> neg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _neg_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rsub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor = _subalpha_bw(grad,input,other, 1.0f, output_mem_config);
    std::swap(grad_tensor[0],grad_tensor[1]);
    return grad_tensor;
}
std::vector<Tensor> rsub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _rsub_bw)(grad, input, other, output_mem_config);
}

std::vector<Tensor> _lt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> lt_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lt_bw)(grad, output_mem_config);
}

std::vector<Tensor> _gt_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> gt_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _gt_bw)(grad, output_mem_config);
}

std::vector<Tensor> _ne_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> ne_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ne_bw)(grad, output_mem_config);
}

std::vector<Tensor> _log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return {mul(grad,recip(input,output_mem_config),std::nullopt,output_mem_config)};
}
std::vector<Tensor> log_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _log_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _abs_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _binary_le_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_le_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_le_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rsqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor rsqrt_result = power(rsqrt(input, true, output_mem_config), 3, output_mem_config);
    Tensor result = mul_unary(mul(grad, rsqrt_result, std::nullopt, output_mem_config) , -0.5, output_mem_config);
    float t_inf = std::numeric_limits<float>::infinity();
    result = where(eqz(input, output_mem_config), t_inf, result, output_mem_config);
    float t_nan  = std::nanf("");
    result = where(ltz(input, output_mem_config), t_nan, result, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> rsqrt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _rsqrt_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _clamp_bw(const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config)
{
    std::vector<Tensor> grad_tensor;
    Tensor minT = gte_unary(input, min, output_mem_config);
    Tensor maxT = lte_unary(input, max, output_mem_config);
    Tensor result = logical_and(minT, maxT, std::nullopt, output_mem_config);
    result = mul(grad, result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> clamp_bw(const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _clamp_bw)(grad, input, min, max, output_mem_config);
}


std::vector<Tensor> _clamp_min_bw(const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config)
{
    std::vector<Tensor> grad_tensor;
    Tensor minT = gte_unary(input, min, output_mem_config);
    Tensor result = mul(grad, minT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> clamp_min_bw(const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _clamp_min_bw)(grad, input, min, output_mem_config);
}


std::vector<Tensor> _clamp_max_bw(const Tensor& grad, const Tensor& input, float max, const MemoryConfig& output_mem_config)
{
    std::vector<Tensor> grad_tensor;
    Tensor maxT = lte_unary(input, max, output_mem_config);
    Tensor result = mul(grad, maxT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> clamp_max_bw(const Tensor& grad, const Tensor& input, float max, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _clamp_max_bw)(grad, input, max, output_mem_config);
}
std::vector<Tensor> _relu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(gtz(input,output_mem_config), grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> relu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _relu_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _atan2_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor recip_mul = mul(grad, recip(square(hypot(input,other))),std::nullopt, output_mem_config);
    Tensor grad_a = mul(other, recip_mul, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul(neg(input), recip_mul, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> atan2_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _atan2_bw)(grad, input, other, output_mem_config);
}

std::vector<Tensor> _hypot_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result_recip = recip(hypot(input, other, output_mem_config), output_mem_config);
    Tensor grad_a = mul(grad, mul(input, result_recip, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul(grad, mul(other, result_recip, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> hypot_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hypot_bw)(grad, input, other, output_mem_config);
}

//bw(expm1) = grad * expm1(input) + 1
std::vector<Tensor> _expm1_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor eresult = expm1(input, output_mem_config);
    Tensor rp1 = add1(eresult , output_mem_config);
    Tensor result = mul(grad, rp1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> expm1_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _expm1_bw)(grad, input, output_mem_config);
}


// #  bw (exp2) = grad * exp2(input) * M_LN2
// # M_LN2 = 0.693147180559945309417
std::vector<Tensor> _exp2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = exp2(input, output_mem_config);
    exp_result = mul_unary(exp_result, M_LN2, output_mem_config);
    Tensor result = mul(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> exp2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _exp2_bw)(grad, input, output_mem_config);
}

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight
std::vector<Tensor> _lerp(const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float sub_scalar = 1.0f - weight;
    Tensor result_1 = mul_unary(grad, sub_scalar, output_mem_config);
    grad_tensor.emplace_back(result_1);
    Tensor result_2 = mul_unary(grad, weight, output_mem_config);
    grad_tensor.emplace_back(result_2);
    return grad_tensor;
}
std::vector<Tensor> lerp_bw(const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lerp)(grad, input, end, weight, output_mem_config);
}

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight
std::vector<Tensor> _lerp_overload(const Tensor& grad, const Tensor& input, const Tensor& end, const Tensor& weight, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result_1 = mul(grad, sub_unary(1.0, weight, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_1);
    Tensor result_2 = mul(grad, weight, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_2);
    return grad_tensor;
}
std::vector<Tensor> lerp_bw(const Tensor& grad, const Tensor& input, const Tensor& end, const Tensor& weight, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lerp_overload)(grad, input, end, weight, output_mem_config);
}

std::vector<Tensor> _gelu_bw(const Tensor& grad, const Tensor& input, string approximate, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;

    if (approximate == "tanh"){
        float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
        float kKappa = 0.044715;
        Tensor x_sq = mul(input , input, std::nullopt, output_mem_config);
        Tensor x_cube = mul(x_sq , input, std::nullopt, output_mem_config);
        Tensor inner = mul_unary(kBeta , add(input , mul_unary(kKappa , x_cube, output_mem_config)), output_mem_config);
        Tensor tanh_inner = tanh(inner, output_mem_config);

        Tensor left = mul_unary(0.5 , input, output_mem_config);
        Tensor right = add_unary(1 , tanh_inner, output_mem_config);

        Tensor left_derivative = mul_unary(0.5 , right, output_mem_config);

        Tensor tanh_derivative = neg(sub_unary(mul(tanh_inner , tanh_inner, std::nullopt, output_mem_config),1, output_mem_config), output_mem_config);
        Tensor inner_derivative = mul_unary(kBeta , (add_unary(1 , mul_unary(3 , mul_unary(kKappa , x_sq, output_mem_config), output_mem_config), output_mem_config)));
        Tensor right_derivative = mul(mul(left , tanh_derivative, std::nullopt, output_mem_config) , inner_derivative, std::nullopt, output_mem_config);

        Tensor grad_a = mul(grad , (add(left_derivative , right_derivative)), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(grad_a);
    }
    else{
        float kAlpha = M_SQRT1_2;
        float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
        Tensor cdf = mul_unary(0.5 , (add_unary(1 , erf(mul_unary(input , kAlpha, output_mem_config)), output_mem_config)));
        Tensor pdf = mul_unary(kBeta , exp(mul_unary(mul(input , input) , -0.5), output_mem_config), output_mem_config);
        Tensor grad_a = mul(grad , (add(cdf , mul(input , pdf))));
        grad_tensor.emplace_back(grad_a);
    }

    return grad_tensor;
}
std::vector<Tensor> gelu_bw(const Tensor& grad, const Tensor& input, string approximate, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _gelu_bw)(grad, input, approximate, output_mem_config);
}

std::vector<Tensor> _bias_gelu_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, string approximate, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor input = add(input_a, input_b);

    grad_tensor = gelu_bw(grad, input, approximate=approximate);

    return grad_tensor;
}
std::vector<Tensor> bias_gelu_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, string approximate, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _bias_gelu_bw)(grad, input_a, input_b, approximate, output_mem_config);
}

std::vector<Tensor> _bias_gelu_unary_bw(const Tensor& grad, const Tensor& input_tensor, float bias, string approximate, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor input = add_unary(input_tensor, bias);

    grad_tensor = gelu_bw(grad, input, approximate=approximate);

    return grad_tensor;
}
std::vector<Tensor> bias_gelu_unary_bw(const Tensor& grad, const Tensor& input, float bias, string approximate, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _bias_gelu_unary_bw)(grad, input, bias, approximate, output_mem_config);
}

std::vector<Tensor> _squared_difference_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor difference = sub(input, other);
    Tensor grad_a = mul_unary(2, mul(grad, difference, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul_unary(-1, grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> squared_difference_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _squared_difference_bw)(grad, input, other, output_mem_config);
}


// torch reference
// - name: ldexp(Tensor self, Tensor other) -> Tensor
//   self: grad * 2^other
//   other: grad * self * ln(2) * (2^other)
// # M_LN2 = ln(2)= 0.693147180559945309417
std::vector<Tensor> _ldexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tpow_o = mul(grad, rpow(other, 2.0, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(tpow_o);
    Tensor result = mul(input, mul_unary(tpow_o, M_LN2, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> ldexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ldexp_bw)(grad, input, other, output_mem_config);
}


std::vector<Tensor> _xlogy_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad1_result = mul(grad, log(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor zero_tensor = full_like(other, 0.0, output_mem_config);
    grad1_result = where(logical_and(eqz(input, output_mem_config), lte(other, zero_tensor, std::nullopt, output_mem_config), std::nullopt, output_mem_config) , zero_tensor,
                   where(ltz(other, output_mem_config), std::nanf(" "), grad1_result, output_mem_config), output_mem_config);
    grad1_result = where(eq_unary(input, std::nanf(" "), output_mem_config), std::nanf(" "), grad1_result, output_mem_config);
    grad_tensor.emplace_back(grad1_result);
    Tensor div_result = mul(input, recip(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = mul(grad, div_result , std::nullopt, output_mem_config);
    grad2_result = where(eqz(other, output_mem_config), mul_unary(sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), output_mem_config), grad2_result, output_mem_config);
    grad2_result = where(eq_unary(other, std::nanf(" "), output_mem_config), std::nanf(" "), grad2_result, output_mem_config);
    grad_tensor.emplace_back(grad2_result);
    return grad_tensor;
}
std::vector<Tensor> xlogy_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _xlogy_bw)(grad, input, other, output_mem_config);
}

/*
Torch Reference:
name: logaddexp(Tensor self, Tensor other) -> Tensor
self: grad / (1 + exp(other - self)).conj()
other: grad / (1 + exp(self - other)).conj()
*/
std::vector<Tensor> _logaddexp_bw(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor opexp = add1(exp(sub(other, input_a, std::nullopt, output_mem_config), output_mem_config), output_mem_config);
    Tensor grad_a = mul(grad, recip(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    opexp = add1(exp(sub(input_a, other, std::nullopt, output_mem_config), output_mem_config), output_mem_config);
    Tensor grad_b = mul(grad, recip(opexp, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> logaddexp_bw(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _logaddexp_bw)(grad, input_a, other, output_mem_config);
}

/*
Torch reference
name: logaddexp2(Tensor self, Tensor other) -> Tensor
self: grad / (1 + pow(2, other - self))
other: grad / (1 + pow(2, self - other))
*/

std::vector<Tensor> _logaddexp2_bw(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor oppow = add1(rpow(sub(other, input_a, std::nullopt, output_mem_config), 2,  output_mem_config), output_mem_config);
    Tensor grad_a = mul(grad, recip(oppow, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    oppow = add1(rpow(sub(input_a, other, std::nullopt, output_mem_config), 2, output_mem_config), output_mem_config);
    Tensor grad_b = mul(grad, recip(oppow, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> logaddexp2_bw(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _logaddexp2_bw)(grad, input_a, other, output_mem_config);
}
std::vector<Tensor> _concat_bw(const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    const Shape start_index = {0, 0, 0, 0};
    const Shape end_index = {input.get_legacy_shape()[0] - 1, input.get_legacy_shape()[1] - 1, input.get_legacy_shape()[2] - 1, input.get_legacy_shape()[3] - 1};

    Tensor grad_a = unpad(grad, start_index, end_index);
    grad_tensor.emplace_back(grad_a);

    Shape start_index_2 = {0, 0, 0, 0};
    if(dim == 0)
    {
      start_index_2 = {input.get_legacy_shape()[0], 0, 0, 0};
    }
    else if(dim == 1)
    {
        start_index_2 = {input.get_legacy_shape()[0] - 1, input.get_legacy_shape()[1], 0, 0};
    }
    else if(dim == 2)
    {
        start_index_2 = {input.get_legacy_shape()[0] - 1, input.get_legacy_shape()[1] - 1, input.get_legacy_shape()[2], 0};
    }
    else if(dim == 3)
    {
        start_index_2 = {0, 0, 0, input.get_legacy_shape()[3]};
    }
    const Shape end_index_2 = {grad.get_legacy_shape()[0] - 1, grad.get_legacy_shape()[1] - 1, grad.get_legacy_shape()[2] - 1, grad.get_legacy_shape()[3] - 1};
    Tensor grad_b = unpad(grad, start_index_2, end_index_2);
    grad_tensor.emplace_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> concat_bw(const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _concat_bw)(grad, input, other, dim, output_mem_config);
}



std::vector<Tensor> _hardsigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = where(logical_or(lte_unary(input, -3, output_mem_config), gte_unary(input, 3, output_mem_config), std::nullopt, output_mem_config), zeros_like(input, output_mem_config), mul_unary(grad, 1.0/6), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> hardsigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hardsigmoid_bw)(grad, input, output_mem_config);
}

float factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

std::vector<Tensor> _i0_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;

    Tensor result=zeros_like(input);
    Tensor term=zeros_like(input);
    Tensor final_res=zeros_like(input);

    float fact;
    for (int i=0; i<100; i++){
        fact=factorial(i);
        term = mul_unary(power(div_unary(input, 2.0, output_mem_config), 2*i-1, output_mem_config), i / (fact*fact), output_mem_config);
        result = add(result,term);
    }
    final_res= mul(result, grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(final_res);
    return grad_tensor;
}
std::vector<Tensor> i0_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _i0_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _hardshrink_bw(const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor hardshrink_result = hardshrink(input_tensor, lambd, output_mem_config);
    Tensor result = where(eqz(hardshrink_result, output_mem_config), 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> hardshrink_bw(const Tensor& grad, const Tensor& input, float lambd, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hardshrink_bw)(grad, input, lambd, output_mem_config);
}

//softshrink
// result: torch.where(self < -lambd, grad, torch.where(self > lambd, grad, torch.tensor(0.0)))
std::vector<Tensor> _softshrink_bw(const Tensor& grad, const Tensor& input_tensor, float lambd, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = where(logical_or(lt(input_tensor, full_like(input_tensor, -lambd, output_mem_config), std::nullopt, output_mem_config), gt(input_tensor, full_like(input_tensor, lambd, output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config), grad, zeros_like(grad, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> softshrink_bw(const Tensor& grad, const Tensor& input, float lambd, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _softshrink_bw)(grad, input, lambd, output_mem_config);
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> _hardswish_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(lt(input, full_like(input, -3.0f), std::nullopt, output_mem_config),
                        0.0, where(lte(input, full_like(input, 3.0f), std::nullopt, output_mem_config),
                        mul(grad, add_unary(mul_unary(input, 0.3333f, output_mem_config), 0.5f, output_mem_config), std::nullopt, output_mem_config), grad), output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> hardswish_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hardswish_bw)(grad, input, output_mem_config);
}

// Softplus
// (threshold >= 0) grad_self = grad * torch.exp(beta * self) / (1 + torch.exp(beta * self))
// (threshold < 0) grad_self = grad * torch.exp(beta * self) / (torch.exp(beta * self) + torch.exp(threshold))
std::vector<Tensor> _softplus_bw(const Tensor& grad, const Tensor& input, float beta, float threshold, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor mul_input_beta = mul_unary(input, beta, output_mem_config);
    Tensor exp_beta_self = exp(mul_input_beta, output_mem_config);
    Tensor sub_result = add_unary(-threshold , mul_input_beta, output_mem_config);
    Tensor temp = mul(mul(grad, exp_beta_self, std::nullopt, output_mem_config), recip(add1(exp_beta_self, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    Tensor grad_result = where(gtz(sub_result, output_mem_config), grad, temp, output_mem_config);
    mul_input_beta.deallocate();
    exp_beta_self.deallocate();
    sub_result.deallocate();
    temp.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> softplus_bw(const Tensor& grad, const Tensor& input, float beta, float threshold, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _softplus_bw)(grad, input, beta, threshold, output_mem_config);
}

std::vector<Tensor> _polygamma_bw(const Tensor& grad, const Tensor& input, int n, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, polygamma(input, (n+1), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> polygamma_bw(const Tensor& grad, const Tensor& input, int n, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _polygamma_bw)(grad, input, n, output_mem_config);
}

std::vector<Tensor> _atan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, recip(add_unary(square(input), 1), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> atan_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _atan_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _atanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, recip(sub_unary(1, square(input)), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> atanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _atanh_bw)(grad, input, output_mem_config);
}

// Asin
// result: grad * (-self * self + 1).rsqrt()
std::vector<Tensor> _asin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    UnaryWithParam op1 {UnaryOpType::SQUARE};
    UnaryWithParam op2 {UnaryOpType::NEG};
    UnaryWithParam op3 {UnaryOpType::ADD_UNARY_SFPU, 1.0f};
    UnaryWithParam op4 {UnaryOpType::RSQRT, true};
    Tensor grad_result = mul(grad, unary_chain( input, {op1, op2, op3, op4}, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
    Tensor sub_one = add_unary(-1, input, output_mem_config);
    Tensor sub_minus_one = add1(input, output_mem_config);
    Tensor result = where(
        ltz(sub_minus_one, output_mem_config),
        t_nan,
        where(
            gtz(sub_one, output_mem_config),
            t_nan,
            where(
                eqz(sub_minus_one, output_mem_config),
                mul(sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                where(
                    eqz(sub_one, output_mem_config),
                    mul(sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    grad_result,
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> asin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _asin_bw)(grad, input, output_mem_config);
}

// Asinh
// result: grad * (self * self + 1).rsqrt()
std::vector<Tensor> _asinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    UnaryWithParam op1 {UnaryOpType::SQUARE};
    UnaryWithParam op2 {UnaryOpType::ADD_UNARY_SFPU, 1.0f};
    UnaryWithParam op3 {UnaryOpType::RSQRT, true};
    Tensor grad_result = mul(grad, unary_chain( input, {op1, op2, op3}, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> asinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _asinh_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _cosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, sinh(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> cosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _cosh_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _cos_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, (neg(sin(input_tensor, output_mem_config), output_mem_config)), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> cos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _cos_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _acosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_rsqrt = square(input, output_mem_config);
    in_rsqrt = rsqrt(sub_unary(in_rsqrt, 1.0, output_mem_config), true, output_mem_config);
    Tensor grad_a = mul(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan  = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor cond_result = logical_or(lt(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config),
                        gt(input, full_like(input, 1.0, output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(eqz(cond_result, output_mem_config), t_nan, grad_a, output_mem_config);
    cond_result = logical_or(eq(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config),
                        eq(input, full_like(input, 1.0, output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(eq(cond_result, ones_like(input, output_mem_config), std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> acosh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _acosh_bw)(grad, input, output_mem_config);
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> _acos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = neg(input, output_mem_config);
    Tensor in_rsqrt = rsqrt(add1(mul(neg_in, input, std::nullopt, output_mem_config), output_mem_config), true, output_mem_config);
    in_rsqrt = neg(in_rsqrt, output_mem_config);
    Tensor grad_a = mul(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor neg_one = full_like(input, -1.0, output_mem_config);
    Tensor pos_one = full_like(input, 1.0, output_mem_config);
    Tensor t_inf = mul_unary(sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), output_mem_config);
    grad_a = where(logical_or(lt(input, neg_one, std::nullopt, output_mem_config),
             gt(input, pos_one, std::nullopt, output_mem_config), std::nullopt, output_mem_config), std::nanf(" "), grad_a, output_mem_config);
    grad_a = where(eq(input, neg_one, std::nullopt, output_mem_config), t_inf,
                   where(eq(input, pos_one, std::nullopt, output_mem_config), t_inf,
                   grad_a, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> acos_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _acos_bw)(grad, input, output_mem_config);
}

// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> _leaky_relu_bw(const Tensor& grad, const Tensor& input, float negative_slope, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(gtz(input, output_mem_config), grad, mul_unary(grad, negative_slope, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> leaky_relu_bw(const Tensor& grad, const Tensor& input, float negative_slope, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _leaky_relu_bw)(grad, input, negative_slope, output_mem_config);
}

// ELU
// result : grad * (torch.where(input >= 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> _elu_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(gez(input, output_mem_config), grad, mul(grad, mul_unary(exp(input, output_mem_config), alpha, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> elu_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _elu_bw)(grad, input, alpha, output_mem_config);
}

// Hardtanh
// result: torch.where((input <= min) | (input >= max), 0.0, grad)
std::vector<Tensor> _hardtanh_bw(const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(lte(input, full_like(input, min), std::nullopt, output_mem_config),
                        0.0, where(gte(input, full_like(input, max), std::nullopt, output_mem_config),
                        0.0, grad), output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> hardtanh_bw(const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _hardtanh_bw)(grad, input, min, max, output_mem_config);
}

std::vector<Tensor> _angle_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> angle_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _angle_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sin_bw(const Tensor& grad, const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input = mul(grad, cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}
std::vector<Tensor> sin_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sin_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, cosh(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> sinh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sinh_bw)(grad, input, output_mem_config);
}

// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> _celu_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor div_result = mul(input, recip(full_like(input, alpha, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    Tensor exp_result = exp(div_result, output_mem_config);
    Tensor grad_result = where(gt(input, zeros_like( input, output_mem_config), std::nullopt, output_mem_config),
                        grad, mul(grad, exp_result, std::nullopt, output_mem_config), output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> celu_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _celu_bw)(grad, input, alpha, output_mem_config);
}

std::vector<Tensor> _binary_lt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_lt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_lt_bw)(grad, input, output_mem_config);
}

//erfinv
//self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
std::vector<Tensor> _erfinv_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(0.5, mul(sqrt(full_like(input, M_PI , output_mem_config), output_mem_config), mul( exp( square(erfinv(input, output_mem_config), output_mem_config), output_mem_config), grad, std::nullopt, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erfinv_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _erfinv_bw)(grad, input, output_mem_config);
}


// bw(log10(in)) = grad/(in * 2.30258509299404568402)
std::vector<Tensor> _log10_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(ltz(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor grad_a = mul(grad, recip(mul_unary(input, M_LN10, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(logical_and(eqz(input, output_mem_config), eqz(grad, output_mem_config), std::nullopt, output_mem_config), std::nanf(" "),
             where(eqz(input, output_mem_config), t_inf, grad_a, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log10_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _log10_bw)(grad, input, output_mem_config);
}


// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> _log1p_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(ltz(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_inp1 = add1(input, output_mem_config);
    Tensor grad_a = mul(grad, recip(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(eq(input, full_like(input, -1.0, output_mem_config), std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config);
    grad_a = where(logical_and(eqz(t_inp1, output_mem_config), eqz(grad, output_mem_config)), std::nanf(" "), grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log1p_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _log1p_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _binary_ne_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_ne_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_ne_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _erf_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(M_2_SQRTPI, mul(exp(neg(square(input, output_mem_config), output_mem_config), output_mem_config), grad, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erf_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _erf_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _erfc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(-M_2_SQRTPI, mul(exp(neg(square(input, output_mem_config), output_mem_config), output_mem_config), grad, std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> erfc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _erfc_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _digamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, polygamma(input, 1, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> digamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _digamma_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _deg2rad_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_PI_180 = M_PI/180;
    Tensor grad_result = mul_unary(grad, M_PI_180, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> deg2rad_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _deg2rad_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rad2deg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180/M_PI;
    Tensor grad_result = mul_unary(grad, M_180_PI, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> rad2deg_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _rad2deg_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _reciprocal_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mul(neg(grad, output_mem_config), recip(square(input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> reciprocal_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _reciprocal_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _relu6_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_tensor = zeros_like(input, output_mem_config);
    Tensor one_tensor = ones_like(input, output_mem_config);
    Tensor six_tensor = full_like(input, 6, output_mem_config);
    Tensor grad_result = where(lte(input, zero_tensor, std::nullopt, output_mem_config), zero_tensor, six_tensor, output_mem_config);
    grad_result = where(logical_and(gtz(input, output_mem_config), lt(input , six_tensor, std::nullopt, output_mem_config), std::nullopt, output_mem_config), grad, grad_result, output_mem_config);
    grad_result = where(gte(input, six_tensor, std::nullopt, output_mem_config), zero_tensor, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> relu6_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _relu6_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _rpow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = zeros_like(input, output_mem_config);
    if (exponent != 0.0){
        grad_result = mul(grad, mul_unary(pow(input, exponent - 1, output_mem_config), exponent, output_mem_config), std::nullopt, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> rpow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _rpow_bw)(grad, input, exponent, output_mem_config);
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<Tensor> _silu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_sigmoid = mul(grad, sigmoid(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor add_sub = add1(mul(sub_unary(1.0f, sigmoid(input, output_mem_config), output_mem_config), input, std::nullopt, output_mem_config), output_mem_config);
    Tensor grad_result = mul(grad_sigmoid, add_sub, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> silu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _silu_bw)(grad, input, output_mem_config);
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> _selu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = mul_unary(grad, 1.0507f, output_mem_config);
    Tensor grad_result = where(gtz(input, output_mem_config), grad_lambd, mul(mul_unary(grad_lambd, 1.673260f, output_mem_config), exp(input, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> selu_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _selu_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _binary_ge_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_ge_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_ge_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _binary_eq_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_eq_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_eq_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _binary_gt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    Tensor zero_input = zeros_like(input, output_mem_config);
    grad_tensor.emplace_back(zero_input);
    return grad_tensor;
}
std::vector<Tensor> binary_gt_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _binary_gt_bw)(grad, input, output_mem_config);
}

// square
// result:  2 * input * grad_data
std::vector<Tensor> _square_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mul(mul_unary(grad, 2.0f, output_mem_config), input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> square_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _square_bw)(grad, input, output_mem_config);
}


std::vector<Tensor> _lgamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mul(grad, digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> lgamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _lgamma_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _frac_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> frac_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _frac_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _trunc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> trunc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _trunc_bw)(grad, input, output_mem_config);
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> _log_sigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = where(ltz(input, output_mem_config), 1, 0, output_mem_config);
    Tensor in_sign = where(ltz(input, output_mem_config), 1, -1, output_mem_config);
    Tensor in_abs = abs(input, output_mem_config);
    Tensor z = exp(neg(in_abs, output_mem_config), output_mem_config);

    Tensor mul_z = mul(z, recip((add1(z , output_mem_config)), output_mem_config), std::nullopt, output_mem_config);

    Tensor mul_sign = mul(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = sub(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result =  mul(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> log_sigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _log_sigmoid_bw)(grad, input, output_mem_config);
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> _tanhshrink_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = square(tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(mul(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> tanhshrink_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _tanhshrink_bw)(grad, input, output_mem_config);
}

//threshold
//if input <= threshold = 0 else grad
std::vector<Tensor> _threshold_bw(const Tensor& grad, const Tensor& input, float threshold, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = where(gtz(add_unary(-threshold , input, output_mem_config), output_mem_config), grad, zeros_like( input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> threshold_bw(const Tensor& grad, const Tensor& input, float threshold, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _threshold_bw)(grad, input, threshold, value, output_mem_config);
}

std::vector<Tensor> _unary_eq_bw(const Tensor& grad, const Tensor& input, float other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}
std::vector<Tensor> unary_eq_bw(const Tensor& grad, const Tensor& input, float other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_eq_bw)(grad, input, other, output_mem_config);
}


// Torch reference
// # if eps is not None:
// #         lo = eps
// #         hi = 1.0 - lo
// #         return torch.where(
// #             torch.logical_and(self >= lo, self <= hi),
// #             grad_output / (self * (1.0 - self)),
// #             0.0,
// #         )
// #     else:
// #         return torch.where(
// #             torch.logical_and(self >= 0.0, self <= 1.0),
// #             grad_output / (self * (1.0 - self)),
// #             self.new_full((), float("nan")),
// #         )
std::vector<Tensor> _logiteps_bw(const Tensor& grad, const Tensor& input, float eps, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float low, high;
    low = eps;
    high = 1.0 - low ;
    Tensor grad_result = mul(grad, recip(mul(input, rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)), std::nullopt, output_mem_config);
    Tensor t_eps = full_like(input, eps, output_mem_config);
    Tensor t_low = full_like(input, low, output_mem_config);
    Tensor t_high = full_like(input, high, output_mem_config);
    Tensor ltl_gth = logical_or(lt(input, t_low, std::nullopt, output_mem_config),
                    gt(input, t_high, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_result = where(eq(ltl_gth, ones_like(input, output_mem_config), std::nullopt, output_mem_config),
                  where(ltz(t_eps, output_mem_config), std::nanf(" "), 0.0, output_mem_config),
                  where(logical_or(eq_unary(input, 0.0, output_mem_config),
                        eq_unary(input, 1.0, output_mem_config), std::nullopt, output_mem_config),
                        mul_unary(sign(grad, output_mem_config),
                        std::numeric_limits<float>::infinity(), output_mem_config), grad_result, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> logiteps_bw(const Tensor& grad, const Tensor& input, float eps, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _logiteps_bw)(grad, input, eps, output_mem_config);
}


std::vector<Tensor> _logit_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mul(grad, recip(mul(input, rsub(input, 1.0f, output_mem_config), std::nullopt, output_mem_config)), std::nullopt, output_mem_config);
    Tensor status = logical_and(gte_unary(input, 0.0f, output_mem_config),
                    lte_unary(input, 1.0f, output_mem_config), std::nullopt, output_mem_config);
    grad_result = where(eq(status, ones_like(input, output_mem_config), std::nullopt, output_mem_config), grad_result, std::nanf(""));
    grad_result = where(logical_or(eq_unary(input, 0.0, output_mem_config),
                        eq_unary(input, 1.0, output_mem_config), std::nullopt, output_mem_config),
                        mul_unary(sign(grad, output_mem_config),
                        std::numeric_limits<float>::infinity(), output_mem_config), grad_result, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> logit_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _logit_bw)(grad, input, output_mem_config);
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> _softsign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    UnaryWithParam op1 {UnaryOpType::ABS};
    UnaryWithParam op2 {UnaryOpType::ADD_UNARY_SFPU, 1.0f};
    UnaryWithParam op3 {UnaryOpType::SQUARE};
    UnaryWithParam op4 {UnaryOpType::RECIP};
    grad_tensor.emplace_back( mul(grad, unary_chain( input, {op1, op2, op3, op4}, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}
std::vector<Tensor> softsign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _softsign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}
std::vector<Tensor> sign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _ceil_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}
std::vector<Tensor> ceil_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ceil_bw)(grad, input, output_mem_config);
}

// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> _log2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(ltz(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor grad_a = mul(grad, recip(mul_unary(input, M_LN2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(logical_and(eqz(input, output_mem_config), eqz(grad, output_mem_config), std::nullopt, output_mem_config), std::nanf(" "),
             where(eqz(input, output_mem_config), t_inf, grad_a, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
std::vector<Tensor> log2_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _log2_bw)(grad, input, output_mem_config);
}
std::vector<Tensor> _ge_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> ge_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ge_bw)(grad, output_mem_config);
}


std::vector<Tensor> _le_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}
std::vector<Tensor> le_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _le_bw)(grad, output_mem_config);
}


std::vector<Tensor> _unary_fmod_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_fmod_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_fmod_bw)(grad, input, scalar, output_mem_config);
}

std::vector<Tensor> _unary_remainder_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_remainder_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_remainder_bw)(grad, input, scalar, output_mem_config);
}

#define CHECK_FOR_COMPLEX(input) do {\
  TT_ASSERT( utility::is_complex_shape(input), "works for complex shape only"); \
  /* TT_ASSERT( input.shape()[0] == 1, "tensor should have batch size 1"); */ \
  } while(0);

//complex conj
std::vector<Tensor> _conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _conj_bw)(grad, input, output_mem_config);
}

// complex imag
// imag: at::imag(grad)
std::vector<Tensor> _imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mk_complex(zeros_like(real(input, output_mem_config), output_mem_config), grad, output_mem_config) ;
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _imag_bw)(grad, input, output_mem_config);
}

// complex real
// real: at::real(grad)
std::vector<Tensor> _real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = mk_complex(grad, zeros_like(imag(input, output_mem_config), output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _real_bw)(grad, input, output_mem_config);
}

#undef CHECK_FOR_COMPLEX

std::vector<Tensor> _multigammaln_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result = mul(grad, digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = mul(grad, digamma(add_unary(-0.5 , input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);

    Tensor grad_result = add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = mul(grad, digamma(add_unary(-1.0 , input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = mul(grad, digamma(add_unary(-1.5 , input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> multigammaln_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _multigammaln_bw)(grad, input, output_mem_config);
}

}//namespace tt_metal

}//namespace tt
