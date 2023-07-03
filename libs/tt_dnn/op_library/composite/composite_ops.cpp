#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"

#include "tt_numpy/functions.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(const Tensor& reference_tensor);

//TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template<typename T>
Tensor mk_filled_tensor_like(const Tensor& reference_tensor, T val) {
  Tensor k = mk_scalar(val);
  Tensor zero_like = mk_zero_tensor_like(reference_tensor);
  Tensor result = bcast(zero_like,k,BcastOpMath::ADD,BcastOpDim::HW);
  return result;
}

// Function SILU (same as Swish)
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
Tensor silu(const Tensor& a) {
    //using sigmoid known to be a bit off
    //  Tensor sigmoid_a = sigmoid(a);
    //  Tensor silu_a = mul(a,sigmoid_a);
    //  return silu_a;
    //x / (1.0f + exp(-x))
    Tensor sigmoid_a = sigmoid(a);
    Tensor silu_a = mul(a,sigmoid_a);
    return silu_a;
}


// Function neg
//use transformation y = -1 * x by broadcast
Tensor neg(const Tensor& a) {
    Tensor minus_one = mk_scalar(-1.0f);
    Tensor result_neg = bcast(a,minus_one,BcastOpMath::MUL, BcastOpDim::HW);
    return result_neg;
}


//add 1
//use transformation y = 1.0 + x by broadcast
Tensor add1(const Tensor& a) {
    Tensor one = mk_scalar(1.0f);
    Tensor result_addone = bcast(a,one,BcastOpMath::ADD, BcastOpDim::HW);
    return result_addone;
}

//log1p 1
//use transformation y = log(1.0 + x) by broadcast
Tensor log1p(const Tensor& x) {
    Tensor x_1 = add1(x);
    Tensor result_log1p = log(x_1);
    return result_log1p;
}

//softplus[x] = log[1 + exp[x]]
//use transformation y = log[1+exp[x]] by broadcast
Tensor softplus(const Tensor& x) {
    Tensor exp_x = exp(x);
    Tensor result_log1p = log1p(exp_x);
    return result_log1p;
}

//mish[x] = x*tanh[softplus[x]]
//use transformation y = x*tanh[softplus[x]] by broadcast
//Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor mish(const Tensor& x) {
    Tensor sp_x = softplus(x);
    Tensor tanh_x = tanh(sp_x);
    Tensor mish_x = mul(x,tanh_x);
    return mish_x;
}


// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
*    scale = 1.0507009873554804934193349852946
*    return scale * elu(x, alpha)
*
*/
// Function Selu - scaled exponential linear
//use transformation y = scale *(max(0,x)) + min(0,alpha * (exp(X)-1)) by broadcast
//Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor selu(const Tensor& x,const float scale, const float alpha) {
    // term 2
    Tensor t_alpha = mk_scalar(alpha);
    Tensor x_Exp = exp(x);
    Tensor minus_one = mk_scalar(-1.0f);
    Tensor x_Exp_minus_1 = bcast(x_Exp,minus_one,BcastOpMath::ADD, BcastOpDim::HW);
    Tensor result_t2_ = bcast(x_Exp_minus_1,t_alpha,BcastOpMath::MUL, BcastOpDim::HW);
    Tensor result_term2 = mul(gtz(result_t2_),result_t2_);

    // term 2
    Tensor t_scale = mk_scalar(scale);
    Tensor x_relu = relu(x);
    Tensor result_term1 = bcast(x_relu,t_scale,BcastOpMath::MUL, BcastOpDim::HW);

    Tensor result_selu = add(result_term1,result_term2);

    return result_selu;
}

//ELU :
// Theano defins it as,
// return tensor.switch(x > 0, x, alpha * tensor.expm1(x))

// Function Swish
//use transformation y = x * sigmoid( x ) by broadcast
std::function<unary_tensor_op_t> swish = silu;

// Function Clip
//use clip y = min( max( x, min_value), max_value) by broadcast
//Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor clip(const Tensor& a,float low, float high) {
  const Tensor h_const = full_like(a,high);
  const Tensor l_const = full_like(a,low);
  Tensor a_max = tt::tt_metal::min(a,h_const);
  Tensor a_clip = ( low == 0.0f ) ? relu(a_max) : tt::tt_metal::max(a_max,l_const);
  return a_clip;
}

// Function Hard Sigmoid
//     Ref: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
//
//     slope = tensor.constant(0.2, dtype=out_dtype)
//     shift = tensor.constant(0.5, dtype=out_dtype)
//
//     x1 = (x * slope) + shift
//     y = tensor.clip(x1, 0, 1)
//
// PyTorch version:
// hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
Tensor hardsigmoid(const Tensor& a,float scale,float shift) {
    Tensor a_mac = mac_scalar(a,scale,shift);//multiply and add.
    Tensor a_clip = relu_max(a_mac,1.0f);
    return a_clip;
}

// Function @hard_swish
//use transformation y = x * hardsigmoid( x ) by broadcast
//Ref: PyTorch
//hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor hardswish(const Tensor& a,float scale,float shift) {
    Tensor a_sigmoid = hardsigmoid(a,scale,shift);
    Tensor result_sq = mul(a_sigmoid,a);
    return result_sq;
}


//use transformation min = - max( -a, -b)
//Tensor min(const Tensor &a, const Tensor &b) {
//    Tensor aneg( neg(a) );
//    Tensor bneg( neg(b) );
//    Tensor maxneg = tt::tt_metal::max(aneg,bneg);
//    return  neg(maxneg) );
//}


//compute polyval by Horner's rule
Tensor polyval(const Tensor &input_tensor,std::vector<float> coeffs) {
  TT_ASSERT( coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
  if ( coeffs.size() == 1 ) {
    return  mk_filled_tensor_like( input_tensor, coeffs[0] );
  }

  std::vector<Tensor> results(coeffs.size(),input_tensor); //pipeline
  results[0] =
		       bcast(input_tensor,mk_scalar(coeffs[0]),BcastOpMath::MUL,BcastOpDim::HW);
  for(int idx=1; idx < coeffs.size(); idx++) {
    Tensor& last = results[idx-1];
    results[idx] =  bcast( mul(last,input_tensor) , mk_scalar(coeffs[idx]), BcastOpMath::ADD,BcastOpDim::HW);
  }
  return results[coeffs.size()-1];
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor mac(const Tensor& a, const Tensor& b, const Tensor & c) {
  bool a_is_scalar = a.volume() == 1;
  bool b_is_scalar = b.volume() == 1;
  bool c_is_scalar = c.volume() == 1;

  const auto dim = BcastOpDim::HW;
  if ( !a_is_scalar && !b_is_scalar && !c_is_scalar ) {
    //all tensors
    return add(mul(a,b),c);
  } else if ( !a_is_scalar && !b_is_scalar && c_is_scalar ) {
    //a - tensor, b - tensor, c - is scalar
    return bcast(mul(a,b),c,BcastOpMath::ADD,dim);
  } else if ( !a_is_scalar && b_is_scalar && !c_is_scalar ) {
    //a - tensor, b - scalar, c - is tensor
    return add(bcast(a,b,BcastOpMath::MUL,dim),c);
  } else if ( !a_is_scalar && b_is_scalar && c_is_scalar ) {
    //a - tensor, b - scalar, c - is scalar
    return bcast(bcast(a,b,BcastOpMath::MUL,dim),c,BcastOpMath::ADD,dim);
  } else if ( a_is_scalar && !b_is_scalar && !c_is_scalar ) {
    //a - scalar, b - tensor, c - tensor
    return add(bcast(b,a,BcastOpMath::MUL,dim),c);
  } else if ( a_is_scalar && !b_is_scalar && c_is_scalar ) {
    //a - scalar, b - tensor, c - is scalar
    return bcast(bcast(b,a,BcastOpMath::MUL,dim),c,BcastOpMath::ADD,dim);
  } else if ( a_is_scalar && b_is_scalar && !c_is_scalar ) {
    //a - scalar, b - scalar, c - is tensor
    return  bcast(c,mul(a,b),BcastOpMath::ADD,dim);
  }

  // all scalars
  //a - scalar, b - scalar, c - is scalar
  TT_ASSERT( a_is_scalar && b_is_scalar && c_is_scalar);
  return add(mul(a,b),c);
}


Tensor mac_scalar(const Tensor& a, float b, float c) {
  Tensor t_b = mk_scalar(b);
  Tensor t_c = mk_scalar(c);
  return  mac(a,t_b,t_c);
}

Tensor mk_zero_tensor_like(const Tensor& reference_tensor) {
  static const Tensor zero = mk_scalar(0.0f);
  Tensor zero_like = bcast(reference_tensor,zero,BcastOpMath::MUL,BcastOpDim::HW);
  return zero_like;
}

//Function sign
//compute sgn(x) = (x>=0) - (x=<0);
//Tensor sign(const Tensor& x) {
//  return  sub(gez(x),lez(x)) );
//}

//min(a,b) = a - (a - b > 0 )*(a-b)
Tensor min(const Tensor &input_a,const Tensor &input_b)
{
  Tensor t_diff = sub(input_a,input_b);
  Tensor t_flag  = gtz(t_diff);
  Tensor result = sub(input_a, mul(t_flag,t_diff) );
  return result;
}

//max(a,b) = a + (b - a > 0 )*(b-a)
Tensor max(const Tensor &input_a,const Tensor &input_b)
{
  Tensor t_diff = sub(input_b,input_a);
  Tensor t_flag  = gtz(t_diff);
  Tensor result = add(input_a, mul(t_flag,t_diff) );
  return result;
}

//these ops need more polish - TBD
#if 0
/**
  Austin Ho:
  This is just reduce sum so I don't think it should be in composite_ops?
  Reduce sum also returns a 32x32 tensor where the result is the first value. So this assert would fail.
*/
Tensor sum(const Tensor& y) {
  Tensor sum_y = reduce(y, ReduceOpMath::SUM, ReduceOpDim::HW);
  TT_ASSERT( sum_y.volume()%32 == 0, "reduce sum should return a scalar sized tensor");
  return sum_y;
}


//Function std
//compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
// Ref: torch.std
Tensor std(const Tensor& y);

// Function mean
//use transformation y = (y - mean(y))/std(y) by broadcast
// Ref: torch.mean
Tensor mean(const Tensor& y);

// Function normalize
//use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize(const Tensor& a);


Tensor mean(const Tensor& y) {
  Tensor sum_y = sum(y);
  const float val = 1.0f/(float)y.volume();
  Tensor recip_size = mk_scalar(val);
  Tensor mean_y = bcast(sum_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
  return mean_y;
}


// Function normalize
//use transformation y = (y - mean(y))/std(y) by broadcast
Tensor normalize(const Tensor& y) {
  Tensor mean_y = mean(y);
  Tensor y_minus_mean_y = bcast(y,mean_y,BcastOpMath::SUB, BcastOpDim::HW);
  Tensor sqr_y_minus_mean_y = square(y_minus_mean_y);
  float scale = 1.0f/(float)y.volume();
  Tensor recip_size = mk_scalar(scale);
  Tensor var_y = bcast(sqr_y_minus_mean_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
  Tensor std_y = sqrt(var_y);
  Tensor recip_std_y = recip(std_y);
  Tensor z = bcast(y_minus_mean_y,recip_std_y,BcastOpMath::MUL, BcastOpDim::HW);
  return z;
}

Tensor std(const Tensor& y) {
  Tensor mean_y = mean(y);
  Tensor y_minus_mean_y = bcast(y,mean_y,BcastOpMath::SUB, BcastOpDim::HW);
  Tensor sqr_y_minus_mean_y = square(y_minus_mean_y);
  float scale = 1.0f/(float)y.volume();
  Tensor recip_size = mk_scalar(scale);
  Tensor var_y = bcast(sqr_y_minus_mean_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
  Tensor std_y = sqrt(var_y);
  return std_y;
}
#endif

//deg2rad(a) using scale pi/180.
Tensor deg2rad(const Tensor &input_a) {
  constexpr float scale = (float)(M_PI/180.0);
  Tensor t_scale = mk_scalar(scale);
  return  bcast(input_a,t_scale,BcastOpMath::MUL, BcastOpDim::HW);
}

//rad2deg(a) using scale 180/pi.
Tensor rad2deg(const Tensor &input_a) {
  constexpr float scale = (float)(180.0/M_PI);
  Tensor t_scale = mk_scalar(scale);
  return  bcast(input_a,t_scale,BcastOpMath::MUL, BcastOpDim::HW);
}

//hypot(a,b) = sqrt[ a^2 + b^2 ]
Tensor hypot(const Tensor &input_a, const Tensor &input_b) {
  Tensor a_sq = square(input_a);
  Tensor b_sq = square(input_b);
  Tensor c_sq = add(a_sq,b_sq);
  return  sqrt( c_sq );
}

//relu6(a) = min(relu(a),6);
Tensor relu6(const Tensor &input_a) {
  return relu_max(input_a,6.0f);
}



//threshold(a,t,v) = (a < t)*v + (a > t)*a
Tensor threshold(const Tensor &input_a,float threshold, float value) {
  Tensor t_value = mk_scalar(value);
  Tensor t_threshold = mk_scalar(threshold);
  auto bcast_sub = [](const Tensor& a, const Tensor& b) -> Tensor {
		     return bcast(a,b,BcastOpMath::SUB, BcastOpDim::HW);
		   };
  Tensor t0 = bcast_sub(input_a,t_threshold);
  Tensor t1 = bcast(ltz(t0),t_value,BcastOpMath::MUL, BcastOpDim::HW);
  Tensor t2 = mul(gtz(t0),input_a);
  return add(t1,t2);
}

//cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//        = exp[ (1/3)*log[a] ]
Tensor cbrt(const Tensor &input_a) {
  constexpr float scale = (float)(1.0/3.0);
  Tensor t_scale = mk_scalar(scale);
  Tensor t_ln_input = log(abs(input_a)); //negative log is not useful here
  Tensor t1 = bcast(t_ln_input,t_scale,BcastOpMath::MUL, BcastOpDim::HW);
  Tensor t2 = exp(t1);
  Tensor t3 = mul(t2,sign(input_a));
  return t3;
}

//where - ternary operator y = (predicate) ? value_true : value_false; elementwise
//           y = (predicate >= 0)*value_true + (predicate < 0)*value_false
Tensor where(const Tensor& predicate, const Tensor& value_true, const Tensor& value_false) {
  Tensor t2 = mul(gtz(predicate),value_true);
  Tensor t1 = mul(lez(predicate),value_false);
  return add(t2,t1);
}

//on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(const Tensor& reference_tensor) {
    return full_like(reference_tensor,0.0f);
}

//on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor) {
    return full_like(reference_tensor,1.0f);
}

//on-device tensor creation with value like @reference_tensor
Tensor full_like(const Tensor& reference_tensor,float value) {
    return mac_scalar(reference_tensor,0.0f,value);
}

//hardtanh
Tensor hardtanh(const Tensor& a,float low /* = -1.0f */, float high /* = +1.0f */) {
  return  clip(a, low, high);
}

//clamp
std::function<Tensor(const Tensor& a,float low, float high)> clamp = clip;

//on-device tensor creation 0s with shape
Tensor zeros(const Shape shape) {
  return tt::numpy::zeros(shape, DataType::BFLOAT16);
}

//on-device tensor creation 1s with shape
Tensor ones(const Shape shape) {
  return tt::numpy::ones(shape, DataType::BFLOAT16);
}

//on-device tensor creation with shape and filled with value
Tensor full(const Shape shape, float value) {
  return tt::numpy::full(shape, value, DataType::BFLOAT16);
}

//on-device with increment
Tensor arange(int32_t start, int32_t end, int32_t step /*= 1*/) {
  return tt::numpy::arange<bfloat16>(start, end, step);
}


/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor outer(Tensor& a, Tensor& b) {
    const Shape s_a = a.shape();
    const Shape s_b = b.shape();

    auto num_ones = [](const Shape& s) -> uint32_t {
      uint32_t num1s = 0;
      for(uint32_t idx = 0 ; idx < 4; idx++)
          num1s += (uint32_t)(s[idx] == 1);
      return num1s;
    };

    //check if 3 dimensions are 1
    TT_ASSERT( !(num_ones(s_a) < 3) , "3 dimensions are required to be 1 for use with outer product");
    TT_ASSERT( !(num_ones(s_b) < 3) , "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1 );
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1 );


    if ( skip_reshape_a && skip_reshape_b ) {
        return std::move(matmul(a,b));
    } else if ( !skip_reshape_a && skip_reshape_b ) {
        Tensor a_slim = reshape (a, 1, 1, a.volume(), 1);
        return std::move(matmul(a_slim,b));
    } else if ( skip_reshape_a && !skip_reshape_b ) {
        Tensor b_slim = reshape (b, 1, 1, 1, b.volume());
        return std::move(matmul(a,b_slim));
    } else {
      //TT_ASSERT( !skip_reshape_a && !skip_reshape_b,
      //"both operands should require reshape at this point");
      Tensor a_slim = reshape (a, 1, 1, a.volume(), 1);
      Tensor b_slim = reshape (b, 1, 1, 1, b.volume());
      return std::move(matmul(a_slim,b_slim));
    }
}

}//namespace tt_metal

} //namespace tt
