// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/math.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> _addalpha_bw(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    Tensor grad_b = mul_unary(grad, alpha, output_mem_config);
    grad_tensor.push_back(grad_b);

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
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_mul_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_mul_bw)(grad, input, scalar, output_mem_config);
}

std::vector<Tensor> _unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor power_input = power(input, exponent - 1, output_mem_config);

    Tensor result = mul_unary(power_input, exponent, output_mem_config);
    Tensor final_result = mul(result, grad, std::nullopt, output_mem_config);
    grad_tensor.push_back(final_result);
    return grad_tensor;
}
std::vector<Tensor> unary_pow_bw(const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_pow_bw)(grad, input, exponent, output_mem_config);
}

std::vector<Tensor> _unary_add_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_add_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_add_bw)(grad, input, alpha, output_mem_config);
}


std::vector<Tensor> _mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, input_b, std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = mul(grad, input_a, std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> mul_bw(const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _mul_bw)(grad, input_a, input_b, output_mem_config);
}


std::vector<Tensor> _exp_bw(const Tensor& grad, const Tensor& exp_result, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> exp_bw(const Tensor& grad, const Tensor& exp_result, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _exp_bw)(grad, exp_result, output_mem_config);
}


std::vector<Tensor> _addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    Tensor grad_a = mul_unary(mul(grad, tensor2, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = mul_unary(mul(grad, tensor1, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.push_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addcmul_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcmul_bw)(grad, input, tensor1, tensor2, value, output_mem_config);
}


std::vector<Tensor> _unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    return grad_tensor;
}
std::vector<Tensor> unary_assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_assign_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _sqrt_bw(const Tensor& grad, const Tensor& sqrt_result, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, recip(mul_unary(sqrt_result, 2.0, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> sqrt_bw(const Tensor& grad, const Tensor& sqrt_result, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _sqrt_bw)(grad, sqrt_result, output_mem_config);
}


std::vector<Tensor> _unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor scalar_t = full_like(input, scalar, output_mem_config);
    Tensor result = mul(grad, recip(scalar_t, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> unary_div_bw(const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _unary_div_bw)(grad, input, scalar, output_mem_config);
}


std::vector<Tensor> _div_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = mul(grad, recip(other, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = mul(mul(neg(grad, output_mem_config), input, std::nullopt, output_mem_config), recip(square(other, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_b);
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
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> tanh_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _tanh_bw)(grad, input, output_mem_config);
}

std::vector<Tensor> _tan_bw(const Tensor& grad, const Tensor& tan_result, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul(grad, add1(square(tan_result, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(result);
    return grad_tensor;
}
std::vector<Tensor> tan_bw(const Tensor& grad, const Tensor& tan_result, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _tan_bw)(grad, tan_result, output_mem_config);
}

std::vector<Tensor> _addcdiv_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.push_back(grad);
    Tensor grad_a = mul(mul_unary(grad, value, output_mem_config), recip(tensor2, output_mem_config));
    grad_tensor.push_back(grad_a);
    Tensor tmp = mul(mul_unary(neg(grad, output_mem_config), value, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b = mul(tmp, recip(square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> addcdiv_bw(const Tensor& grad, const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _addcdiv_bw)(grad, input, tensor1, tensor2, value, output_mem_config);
}

std::vector<Tensor> _where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = zeros_like(grad, output_mem_config);
    Tensor grad_a = where(condition, grad, t_zero, output_mem_config);
    grad_tensor.push_back(grad_a);
    Tensor grad_b = where(condition, t_zero, grad, output_mem_config);
    grad_tensor.push_back(grad_b);

    return grad_tensor;
}
std::vector<Tensor> where_bw(const Tensor& grad, const Tensor& condition, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _where_bw)(grad, condition, input, other, output_mem_config);
}

std::vector<Tensor> _max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    Tensor zeros_t = zeros_like(input, output_mem_config);
    std::vector<Tensor> grad_tensor;

    Tensor t_scale_grad = mul_unary(grad, 0.5, output_mem_config);
    //op_type=unary_op_type, .param=static_cast<float>(param)}}, output_mem_config
    Tensor t_sub_oi = sub(other, input, std::nullopt, output_mem_config);
    Tensor t_sub_gtz = gtz(t_sub_oi,output_mem_config);
    Tensor t_sub_eqz = eqz(t_sub_oi,output_mem_config);
    Tensor t_sub_gtz_io = ltz(t_sub_oi,output_mem_config);
    Tensor t_alternate = where(t_sub_eqz, t_scale_grad, zeros_t, output_mem_config);
    Tensor grad_other = add(where(t_sub_gtz, grad, zeros_t, output_mem_config),
                            t_alternate, std::nullopt, output_mem_config);
    Tensor grad_input = add(where(t_sub_gtz_io, grad, zeros_t, output_mem_config),
                            t_alternate, std::nullopt, output_mem_config);
    grad_tensor.push_back(grad_input);
    grad_tensor.push_back(grad_other);
    return grad_tensor;
}
std::vector<Tensor> max_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _max_bw)(grad, input, other, output_mem_config);
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
    for(int rank = val.shape().rank()-1; rank >=0; rank--)
        val = sum(val, rank, output_mem_config);
    Tensor result = zeros_like(grad, output_mem_config);
    result = bcast(result, val,  BcastOpMath::ADD, BcastOpDim::HW, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}
std::vector<Tensor> fill_bw(const Tensor& grad, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _fill_bw)(grad, output_mem_config);
}

}//namespace tt_metal

}//namespace tt
