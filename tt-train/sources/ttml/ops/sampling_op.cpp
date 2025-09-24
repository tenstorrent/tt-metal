#include "sampling_op.hpp"

namespace ttml::ops {

autograd::TensorPtr ttml::ops::sample_op(
    autograd::TensorPtr& t, float temperature, uint32_t seed, std::optional<tt::tt_metal::Tensor> logits_padding_mask) {
    auto sampled_tensor = ttnn_fixed::sample(t->get_value(), temperature, seed, logits_padding_mask);
    auto out = autograd::create_tensor();
    out->set_value(sampled_tensor);

    autograd::GradFunction grad = [t, out]() {
        // Argmax in sampling is non-differentiable; no gradient to propagate.
        // TODO: replace argmax with softmax?
        out->set_grad(ttml::core::zeros_like(out->get_value()));
    };

    return out;
}

}  // namespace ttml::ops
