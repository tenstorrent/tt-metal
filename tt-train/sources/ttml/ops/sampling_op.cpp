#include "sampling_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr ttml::ops::sample_op(
    autograd::TensorPtr& t, float temperature, uint32_t seed, std::optional<autograd::TensorPtr&> logits_padding_mask) {
    tt::tt_metal::Tensor sampled_tensor;

    if (logits_padding_mask.has_value()) {
        sampled_tensor =
            ttnn_fixed::sample(t->get_value(), temperature, seed, logits_padding_mask.value()->get_value());
    } else {
        sampled_tensor = ttnn_fixed::sample(t->get_value(), temperature, seed);
    }

    auto out = autograd::create_tensor();
    out->set_value(sampled_tensor);

    autograd::GradFunction grad = [t, out]() {
        // Argmax in sampling is non-differentiable; no gradient to propagate.
        // TODO: replace argmax with softmax?
        throw std::runtime_error("Sampling operation backward pass is not implemented.");
    };

    auto links = autograd::get_links(t);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
