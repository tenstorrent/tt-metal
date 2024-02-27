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


std::vector<Tensor> _unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float inv_scalar = 1.0f/scalar;
    Tensor result = mul_unary(grad, inv_scalar, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_div_bw)(grad, input, scalar, output_mem_config);
}


std::vector<Tensor> _div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, recip(other, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul(mul(neg(grad, output_mem_config), input, std::nullopt, output_mem_config), recip(square(other, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _div_bw)(grad, input, other, output_mem_config);
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
    Tensor grad_a = mul(mul_unary(grad, value, output_mem_config), recip(tensor2, output_mem_config));
    grad_tensor.emplace_back(grad_a);
    Tensor tmp = mul(mul_unary(neg(grad, output_mem_config), value, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b = mul(tmp, recip(square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);

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
    TT_FATAL(input.dtype() == DataType::UINT32, "Input must be UINT32");
    TT_FATAL(grad.shape()[0] == 1 && grad.shape()[1] == 1, "First two dimensions for the grad must be 1");
    TT_FATAL(input.shape()[1] == 1 && input.shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
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
std::vector<Tensor> _sub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = neg(grad);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> sub_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sub_bw)(grad, input, other, output_mem_config);
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
    std::vector<Tensor> grad_tensor = _sub_bw(grad,input,other,output_mem_config);
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
//   self: 2^other
//   other: self * ln(2) * (2^other)
// # M_LN2 = ln(2)= 0.693147180s559945309417
std::vector<Tensor> _ldexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tpow_o = mul_unary(exp(other, output_mem_config), M_LN2, output_mem_config);
    grad_tensor.emplace_back(tpow_o);
    Tensor result = mul(input, mul_unary(tpow_o, M_LN2, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> ldexp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _ldexp_bw)(grad, input, other, output_mem_config);
}

// torch reference
// # - name: xlogy.Tensor(Tensor self, Tensor other) -> Tensor
// #   self: at::xlogy(grad, other).masked_fill((self == 0.) & (other <= 0.), 0.)
// #   other: grad * self / other
// #   result: at::xlogy(self_t, other_p).masked_fill((self_p == 0.) & (other_p <= 0.), 0.) + other_t * self_p / other_p
std::vector<Tensor> _xlogy_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad1_result = mul(grad, log(input, output_mem_config), std::nullopt, output_mem_config);
    grad1_result = where(ltz(other, output_mem_config), 0.0, grad1_result);
    grad_tensor.emplace_back(grad1_result);
    Tensor div_result = mul(input, recip(other, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad2_result = mul(grad, div_result , std::nullopt, output_mem_config);
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
    const Shape end_index = {input.shape()[0] - 1, input.shape()[1] - 1, input.shape()[2] - 1, input.shape()[3] - 1};

    Tensor grad_a = unpad(grad, start_index, end_index);
    grad_tensor.emplace_back(grad_a);

    Shape start_index_2 = {0, 0, 0, 0};
    if(dim == 0)
    {
      start_index_2 = {input.shape()[0], 0, 0, 0};
    }
    else if(dim == 1)
    {
        start_index_2 = {input.shape()[0] - 1, input.shape()[1], 0, 0};
    }
    else if(dim == 2)
    {
        start_index_2 = {input.shape()[0] - 1, input.shape()[1] - 1, input.shape()[2], 0};
    }
    else if(dim == 3)
    {
        start_index_2 = {input.shape()[0] - 1, input.shape()[1] - 1, 0, input.shape()[3]};
    }
    const Shape end_index_2 = {grad.shape()[0] - 1, grad.shape()[1] - 1, grad.shape()[2] - 1, grad.shape()[3] - 1};
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
    Tensor grad_a = mul_unary(grad, 1.0/6);
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

}//namespace tt_metal

}//namespace tt
