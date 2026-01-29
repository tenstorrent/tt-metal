#include "view_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr view(const autograd::TensorPtr& tensor, const ttnn::Shape& new_shape) {
    auto original_shape = tensor->get_value().logical_shape();
    auto result = ttnn::view(tensor->get_value(), new_shape);
    auto out = autograd::create_tensor(result);

    autograd::GradFunction grad = [tensor, original_shape, out]() {
        auto grad_output = out->get_grad();
        auto grad_input = ttnn::view(grad_output, original_shape);
        tensor->add_grad(grad_input);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
