#pragma once
#include <type_traits>
#include "common/constants.hpp"

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"


namespace tt {

namespace tt_metal {

#define DECLARE_COMPOSITE_OPERATION(ReturnType, Name, ...) \
    ReturnType Name(__VA_ARGS__); \
    namespace composite_operations { \
        constexpr auto decorated ## Name = DECORATE_AS_COMPOSITE_OPERATION(Name); \
        constexpr inline ReturnType (*Name)(__VA_ARGS__) = []<typename... Args>(Args... args) { \
            return decorated ## Name(args...); \
        }; \
    }

using unary_tensor_op_t = Tensor (const Tensor& a);
using binary_tensor_op_t = Tensor (const Tensor& a, const Tensor& b);

//Note: inline doesn't allow pybind to work well so we keep few function not inlined.

template<typename T>
Tensor mk_scalar(T value) {
    assert(std::is_scalar<T>::value && "T should be scalar");
    std::array<unsigned int,4> shape = {1,1,1,1};
    auto buffer = owned_buffer::create(std::vector{bfloat16(value)});
    Tensor scalar = Tensor(OwnedStorage{buffer}, shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
    return scalar;
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor softshrink(const Tensor& a,float param);

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor hardshrink(const Tensor& a,float param);

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor softsign(const Tensor& a);

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor mac(const Tensor& a, const Tensor& b, const Tensor & c);
Tensor mac_scalar(const Tensor& a, float b, float c);

//Function sign
//compute sgn(x) = (x>=0) - (x<=0);
//inline
//Tensor sign(const Tensor& x);

// Function SILU
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
DECLARE_COMPOSITE_OPERATION(Tensor, silu, const Tensor&);

//log1p 1
//use transformation y = log(1.0 + x) by broadcast
Tensor log1p(const Tensor& x);

//softplus[x] = log[1 + exp[x]]
//use transformation y = log[1+exp[x]] by broadcast
Tensor softplus(const Tensor& x);

//mish[x] = x*tanh[softplus[x]]
//use transformation y = x*tanh[softplus[x]] by broadcast
//Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor mish(const Tensor& x);

// Function Selu - scaled exponential linear
//use transformation y = scale * alpha * (exp(X)-1) by broadcast
//Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor selu(const Tensor& x,const float scale = 1.0507009873554804934193349852946, const float alpha = 1.6732632423543772848170429916717);

// Function Swish = same as SILU
//use transformation y = x * sigmoid( x ) by broadcast
namespace composite_operations {
inline auto swish = silu;
}

//compute polyval by Horner's rule
Tensor polyval(const Tensor &input_tensor,std::vector<float> coeffs);

//min(a,b)
Tensor min(const Tensor &input_a,const Tensor &input_b);

//max(a,b)
Tensor max(const Tensor &input_a,const Tensor &input_b);

//logsigmoid(x) = log(sigmoid(x))
Tensor log_sigmoid(const Tensor &input_a);
//tanhshrink = x - tanh(x)
Tensor tanhshrink(const Tensor &input_a);

//square difference(x, y) = (x - y) * (x - y)
Tensor squared_difference(const Tensor &input_a,const Tensor &input_b);

//addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor addcmul(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value);

//addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor addcdiv(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, float value);

//hypot(a,b) = sqrt[ a^2 + b^2 ]
Tensor hypot(const Tensor &input_a, const Tensor &input_b);

//threshold(a,t,v) = (a < t)*v + (a > t)*a
Tensor threshold(const Tensor &input_a,float threshold, float value);

//cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
Tensor cbrt(const Tensor &input_a);

//PyTorch version:
//hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
//
//for Theano version use scale = 1.0/5.0f = 0.2f with shift = 0.5f.
Tensor hardsigmoid(const Tensor& tensor_a,float scale = 1.0f/6.0f,float shift = 0.5f);

//hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor hardswish(const Tensor& a,float scale = 1.0f/6.0f,float shift = 0.5f);

//where - ternary operator y = (predicate) ? value_true : value_false; elementwise
Tensor where(const Tensor& predicate, const Tensor& value_true, const Tensor& value_false);

//on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(const Tensor& reference_tensor);

//on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor);

//on-device tensor creation with value like @reference_tensor
Tensor full_like(const Tensor& reference_tensor,float value);

//on-device tensor creation 0s with shape
Tensor zeros(const Shape shape, Layout layout = Layout::ROW_MAJOR, Device * device = nullptr);

//on-device tensor creation 1s with shape
Tensor ones(const Shape shape, Layout layout = Layout::ROW_MAJOR, Device * device = nullptr);

Tensor arange(int32_t start, int32_t end, int32_t step = 1, Device * device = nullptr);

//on-device tensor creation with shape and filled with value
Tensor full(const Shape shape, float value, Layout layout = Layout::ROW_MAJOR, Device * device = nullptr);


//clip
Tensor clip(const Tensor& a,float low, float high);

//hardtanh
Tensor hardtanh(const Tensor& a,float low = -1.0f, float high = +1.0f);

//clamp
extern std::function<Tensor(const Tensor& a,float low, float high)> clamp;

/** hyperbolic operations **/
//sinh(x) = (exp(x) - exp(-x))/2
Tensor sinh(const Tensor& input_a);

//cosh(x) = (exp(x) + exp(-x))/2
Tensor cosh(const Tensor& input_a);

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor outer(Tensor& a, Tensor& b);

} //namespace tt_metal

} //namespace tt
