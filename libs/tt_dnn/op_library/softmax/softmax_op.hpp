#pragma once

#include "libs/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt { namespace tt_metal {

// const ref prevents in-place
Tensor softmax_in_place(Tensor& input_tensor);

// computes
// tmp1 = bcast_hw_mul(scale, x)  ; shape of scale is [1,1,32,32]
// tmp2 = bcast_add_w->h(tmp1, mask) ; shape of attn mask is [1,N,32,W]
// y = softmax(tmp2)              ; r=result
// If scale == 0.0f then just y = softmax(x) is computed
Tensor scale_mask_softmax_in_place(float scale, std::optional<std::reference_wrapper<const Tensor>> mask, Tensor& input_tensor);

struct AttentionSoftmaxInPlace {
    float scale;

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    Program create_program(
        const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

} }  // namespace tt::tt_metal
