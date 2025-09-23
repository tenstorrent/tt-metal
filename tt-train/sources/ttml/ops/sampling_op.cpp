#include "sampling_op.hpp"

namespace ttml::ops {

autograd::TensorPtr ttml::ops::sample_op(
    autograd::TensorPtr& t, float temperature, uint32_t seed, std::optional<tt::tt_metal::Tensor> logits_padding_mask) {
    auto out_tensor = ttnn_fixed::sample(t->get_value(), temperature, seed, logits_padding_mask);
}

}  // namespace ttml::ops
