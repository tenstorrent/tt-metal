// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "common/constants.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

namespace tt {

namespace tt_metal {

using unary_tensor_op_t = Tensor(const Tensor& a);
using binary_tensor_op_t = Tensor(const Tensor& a, const Tensor& b);

// Note: inline doesn't allow pybind to work well so we keep few function not inlined.

template <typename T>
Tensor mk_scalar(T value) {
    assert(std::is_scalar<T>::value && "T should be scalar");
    std::array<unsigned int, 4> shape = {1, 1, 1, 1};
    auto buffer = owned_buffer::create(std::vector{bfloat16(value)});
    Tensor scalar = Tensor(OwnedStorage{buffer}, shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
    return scalar;
}

template <typename T>
Tensor mk_tiled_scalar(T value) {
    assert(std::is_scalar<T>::value && "T should be scalar");
    std::array<unsigned int, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    std::vector<bfloat16> buffer_vec(TILE_HW, bfloat16(0));
    buffer_vec[0] = bfloat16(value);
    auto buffer = owned_buffer::create(std::move(buffer_vec));
    Tensor scalar = Tensor(OwnedStorage{buffer}, shape, DataType::BFLOAT16, Layout::TILE);
    return scalar;
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor softshrink(
    const Tensor& a, float param, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor hardshrink(
    const Tensor& a, float param, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function: bias gelu
// Ref: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx_commicrosoft_BiasGelu.html
Tensor bias_gelu_unary(
    const Tensor& a, float bias, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor softsign(const Tensor& a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor mac(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mac(
    const Tensor& a, float b, float c, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function sign
// compute sgn(x) = (x>=0) - (x<=0);
// inline
// Tensor sign(const Tensor& x);

// log1p 1
// use transformation y = log(1.0 + x) by broadcast
Tensor log1p(const Tensor& x, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// softplus[x] = log[1 + exp[x]]
// use transformation y = log[1+exp[x]] by broadcast
Tensor softplus(const Tensor& x, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// mish[x] = x*tanh[softplus[x]]
// use transformation y = x*tanh[softplus[x]] by broadcast
// Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor mish(const Tensor& x, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function Selu - scaled exponential linear
// use transformation y = scale * alpha * (exp(X)-1) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor selu(
    const Tensor& x,
    const float scale = 1.0507009873554804934193349852946,
    const float alpha = 1.6732632423543772848170429916717,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function Swish = same as SILU
// use transformation y = x * sigmoid( x ) by broadcast
Tensor swish(const Tensor& a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// compute polyval by Horner's rule
Tensor polyval(
    const Tensor& input_tensor,
    std::vector<float> coeffs,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// min(a,b)
Tensor min(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// max(a,b)
Tensor max(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// tanhshrink = x - tanh(x)
Tensor tanhshrink(
    const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor lgamma(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// multivariate log-gamma function
Tensor multigammaln(const Tensor& a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor logical_andi(
    const Tensor& input_a,
    float immediate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// ∣input−other∣≤ atol+rtol×∣other∣
Tensor isclose(
    const Tensor& input_a,
    const Tensor& input_b,
    float rtol,
    float atol,
    bool equal_nan = false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// xlogy(x,y))=x*log(y)
Tensor xlogy(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// logical_xor
Tensor logical_xor(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// logical_noti
Tensor logical_noti(
    const Tensor& input_a,
    float immediate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

/*
Returns a new tensor with the signed angles in radians between vectors

                x > 0 and y>= 0 atan(y/x)
                x > 0 and y < 0 -atan(y/x)
                x < 0 and y > 0 pi - atan(y/x)
atan2(y, x) =   x < 0 and y < 0 atan(y/x) - pi
                x < 0 and y = 0 pi
                x = 0 and y > 0 pi/2
                x = 0 and y < 0 -pi/2
                x = 0 and y = 0 0.0
*/
Tensor atan2(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// subalpha(input,other,alpha)=input-alpha*other
Tensor subalpha(
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// addalpha(input, other, alpha) = input + (alpha * other)
Tensor addalpha(
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// repeat interleave
Tensor repeat_interleave(
    const Tensor& input_a,
    uint32_t repeat,
    int32_t dim,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// lerp(input, end, weight) = start + weight * (end - start), weight is float
Tensor lerp(
    const Tensor& input_a,
    const Tensor& input_b,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// lerp(input, end, weight) = start + weight * (end - start), weight is tensor
Tensor lerp(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// hypot(a,b) = sqrt[ a^2 + b^2 ]
Tensor hypot(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor scatter(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// threshold(a,t,v) = (a < t)*v + (a > t)*a
Tensor threshold(
    const Tensor& input_a,
    float threshold,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
Tensor cbrt(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// PyTorch version:
// hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
//
// for Theano version use scale = 1.0/5.0f = 0.2f with shift = 0.5f.
Tensor hardsigmoid(
    const Tensor& tensor_a,
    float scale = 1.0f / 6.0f,
    float shift = 0.5f,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor hardswish(
    const Tensor& a,
    float scale = 1.0f / 6.0f,
    float shift = 0.5f,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
Tensor where(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor where(
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor where(
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor where(
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 0s with shape
Tensor empty(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor arange(
    int32_t start,
    int32_t end,
    int32_t step = 1,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// rpow: y = k**(a)
Tensor rpow(const Tensor& a, float k, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// clip
Tensor clip(
    const Tensor& a,
    float low,
    float high,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// hardtanh
Tensor hardtanh(
    const Tensor& a,
    float low = -1.0f,
    float high = +1.0f,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// clamp
Tensor clamp(
    const Tensor& a,
    float low,
    float high,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// machine epsilon
Tensor eps(
    const Shape shape,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// logit(input, eps)=log(input / 1 - input)
Tensor logit(
    const Tensor& input_a, float eps, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// polygamma
Tensor polygamma(
    const Tensor& input_a, uint32_t k, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor logical_xori(
    const Tensor& input_a,
    float immediate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

/** hyperbolic operations **/
// sinh(x) = (exp(x) - exp(-x))/2
Tensor sinh(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// digamma
Tensor digamma(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// cosh(x) = (exp(x) + exp(-x))/2
Tensor cosh(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

/** Inverse hyperbolic operations **/
// asinh(x) = log(x + sqrt(x^2 + 1))
Tensor asinh(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// acosh(x) = log(x + sqrt(x^2 - 1))
Tensor acosh(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// nextafter
Tensor nextafter(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor atanh(const Tensor& input_a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function variance of whole tensor.
// Tensor variance(const Tensor& y,const Tensor& mean_y);
Tensor var_hw(const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
// Ref: torch.std
Tensor std_hw(const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// Tensor std(const Tensor& y,const Tensor& mean_y);

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize_hw(const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor normalize_global(
    const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor logical_ori(
    const Tensor& input_a,
    float immediate,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// Gated Linear Unit activation
Tensor glu(
    const Tensor& input_a,
    int32_t dim = -1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// ReLU based GLU
Tensor reglu(
    const Tensor& input_a,
    int32_t dim = -1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// Gelu based GLU
Tensor geglu(
    const Tensor& input_a,
    int32_t dim = -1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
// Swish based GLU
Tensor swiglu(
    const Tensor& input_a,
    int32_t dim = -1,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with shape and filled with value
Tensor sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config);

// tril : select lower triangular region of input matrix
Tensor tril(
    const Tensor& input_a,
    int32_t diag = 0,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// triu : select upper triangular region of input matrix
Tensor triu(
    const Tensor& input_a,
    int32_t diag = 0,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// power_fp : power with floating point exponent
Tensor power_fp(
    const Tensor& input_a,
    float exponent,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// repeat a input tensor @input_a as specified by the number of dimensions
Tensor repeat(
    const Tensor& input_a,
    const Shape& shape,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor pow(
    const Tensor& input_a,
    float exponent,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor pow(
    const Tensor& input_a,
    int exponent,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor argmax(
    const Tensor& input_a,
    int64_t dim = 0,
    bool all = false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor argmin(
    const Tensor& input_a,
    int64_t dim = 0,
    bool all = false,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
