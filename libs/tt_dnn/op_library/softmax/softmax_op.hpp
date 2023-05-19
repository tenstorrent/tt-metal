#pragma once

#include "libs/tensor/tensor.hpp"

namespace tt { namespace tt_metal {

// const ref prevents in-place
Tensor softmax_in_place(Tensor &x);

// computes
// tmp1 = bcast_hw_mul(scale, x)  ; shape of scale is [1,1,32,32]
// tmp2 = bcast_add_w->h(tmp1, mask) ; shape of attn mask is [1,N,32,W]
// y = softmax(tmp2)              ; r=result
// If scale == 0.0f then just y = softmax(x) is computed
Tensor scale_mask_softmax_in_place(float scale, const Tensor& mask, Tensor &x);

} }  // namespace tt::tt_metal
