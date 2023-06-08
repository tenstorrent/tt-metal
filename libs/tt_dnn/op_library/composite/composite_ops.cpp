#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"

namespace tt {

namespace tt_metal {

// Function SILU (same as Swish)
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
Tensor silu(const Tensor& a) {
    //using sigmoid known to be a bit off
    //  Tensor sigmoid_a = sigmoid(a);
    //  Tensor silu_a = mul(a,sigmoid_a);
    //  return std::move(silu_a);
    //x / (1.0f + exp(-x))
    Tensor sigmoid_a = sigmoid(a);
    Tensor silu_a = mul(a,sigmoid_a);
    return std::move(silu_a);
}


// Function neg
//use transformation y = -1 * x by broadcast
Tensor neg(const Tensor& a) {
    Tensor minus_one = mk_scalar(-1.0f);
    Tensor result_neg = bcast(a,minus_one,BcastOpMath::MUL, BcastOpDim::HW);
    return std::move(result_neg);
}


//add 1
//use transformation y = 1.0 + x by broadcast
Tensor add1(const Tensor& a) {
    Tensor one = mk_scalar(1.0f);
    Tensor result_addone = bcast(a,one,BcastOpMath::ADD, BcastOpDim::HW);
    return std::move(result_addone);
}

//log1p 1
//use transformation y = log(1.0 + x) by broadcast
Tensor log1p(const Tensor& x) {
    Tensor x_1 = add1(x);
    Tensor result_log1p = log(x_1);
    return std::move(result_log1p);
}

//softplus[x] = log[1 + exp[x]]
//use transformation y = log[1+exp[x]] by broadcast
Tensor softplus(const Tensor& x) {
    Tensor exp_x = exp(x);
    Tensor result_log1p = log1p(exp_x);
    return std::move(result_log1p);
}

//mish[x] = x*tanh[softplus[x]]
//use transformation y = x*tanh[softplus[x]] by broadcast
//Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor mish(const Tensor& x) {
    Tensor sp_x = softplus(x);
    Tensor tanh_x = tanh(sp_x);
    Tensor mish_x = mul(x,tanh_x);
    return std::move(mish_x);
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

    return std::move(result_selu);
}

//ELU :
// Theano defins it as,
// return tensor.switch(x > 0, x, alpha * tensor.expm1(x))

// Function Swish
//use transformation y = x * sigmoid( x ) by broadcast
std::function<unary_tensor_op_t> swish = silu;

//FIXME: required MAX operator in BINARY OPS
//       required relu_max in UNARY OPS

// Function Clip
//use transformation y = x * sigmoid( x ) by broadcast
//Ref: https://www.tensorflow.org/api_docs/python/tf/keras/activations/swish
Tensor clip(const Tensor& a,float low, float high) {
  const Tensor h_const = mk_scalar(high);
  const Tensor l_const = mk_scalar(low);
  Tensor a_max = tt::tt_metal::max(a,h_const);
  Tensor a_clip = ( low == 0.0f ) ? relu(a_max) : tt::tt_metal::min(a_max,l_const);
    return std::move(a_clip);
}

// Function Hard Sigmoid
//     Ref: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
//
//     slope = tensor.constant(0.2, dtype=out_dtype)
//     shift = tensor.constant(0.5, dtype=out_dtype)
//
//     x1 = (x * slope) + shift
//     y = tensor.clip(x1, 0, 1)
Tensor hard_sigmoid(const Tensor& a) {
    Tensor a_mac = mac_scalar(a,0.2f,0.5f);//multiply and add.
    Tensor a_clip = relu_max(a_mac,1.0f);
    return std::move(a_clip);
}

// Function @hard_swish
//use transformation y = x * hard_sigmoid( x ) by broadcast
//Ref:
Tensor hard_swish(const Tensor& a) {
    Tensor a_sigmoid = hard_sigmoid(a);
    Tensor result_sq = mul(a_sigmoid,a);
    return std::move(result_sq);
}


//use transformation min = - max( -a, -b)
//Tensor min(const Tensor &a, const Tensor &b) {
//    Tensor aneg( neg(a) );
//    Tensor bneg( neg(b) );
//    Tensor maxneg = tt::tt_metal::max(aneg,bneg);
//    return std::move( neg(maxneg) );
//}


//compute polyval by Horner's rule
Tensor polyval(const Tensor &input_tensor,std::vector<float> coeffs) {
  TT_ASSERT( coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
  if ( coeffs.size() == 1 ) {
    return std::move( mk_filled_tensor_like( input_tensor, coeffs[0] ) );
  }

  std::vector<Tensor> results(coeffs.size(),input_tensor); //pipeline
  results[0] = std::move(
		       bcast(input_tensor,mk_scalar(coeffs[0]),BcastOpMath::MUL,BcastOpDim::HW) );
  for(int idx=1; idx < coeffs.size(); idx++) {
    Tensor& last = results[idx-1];
    results[idx] = std::move( bcast( mul(last,input_tensor) , mk_scalar(coeffs[idx]), BcastOpMath::ADD,BcastOpDim::HW) );
  }
  return std::move(results[coeffs.size()-1]);
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
    return std::move( bcast(c,mul(a,b),BcastOpMath::ADD,dim) );
  }

  // all scalars
  //a - scalar, b - scalar, c - is scalar
  TT_ASSERT( a_is_scalar && b_is_scalar && c_is_scalar);
  return add(mul(a,b),c);
}


Tensor mac_scalar(const Tensor& a, float b, float c) {
  Tensor t_b = mk_scalar(b);
  Tensor t_c = mk_scalar(c);
  return std::move( mac(a,t_b,t_c) );
}

//Function sign
//compute sgn(x) = (x>=0) - (x=<0);
//Tensor sign(const Tensor& x) {
//  return std::move( sub(gez(x),lez(x)) );
//}

//min(a,b) = a - (a - b > 0 )*(a-b)
Tensor min(const Tensor &input_a,const Tensor &input_b)
{
  Tensor t_diff = sub(input_a,input_b);
  Tensor t_flag  = gtz(t_diff);
  Tensor result = sub(input_a, mul(t_flag,t_diff) );
  return std::move(result);
}

//max(a,b) = a + (b - a > 0 )*(b-a)
Tensor max(const Tensor &input_a,const Tensor &input_b)
{
  Tensor t_diff = sub(input_b,input_a);
  Tensor t_flag  = gtz(t_diff);
  Tensor result = add(input_a, mul(t_flag,t_diff) );
  return std::move(result);
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
  return std::move(sum_y);
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
  return std::move(mean_y);
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
  return std::move(z);
}

Tensor std(const Tensor& y) {
  Tensor mean_y = mean(y);
  Tensor y_minus_mean_y = bcast(y,mean_y,BcastOpMath::SUB, BcastOpDim::HW);
  Tensor sqr_y_minus_mean_y = square(y_minus_mean_y);
  float scale = 1.0f/(float)y.volume();
  Tensor recip_size = mk_scalar(scale);
  Tensor var_y = bcast(sqr_y_minus_mean_y,recip_size,BcastOpMath::MUL, BcastOpDim::HW);
  Tensor std_y = sqrt(var_y);
  return std::move(std_y);
}
#endif

}//namespace tt_metal

} //namespace tt
