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

    Tensor power_input = power(input, exponent - 1, output_mem_config);

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

}//namespace tt_metal

}//namespace tt
