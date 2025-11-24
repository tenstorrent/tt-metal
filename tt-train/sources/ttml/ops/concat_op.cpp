#include "concat_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr concat(const std::vector<autograd::TensorPtr>& tensors, int32_t dim) {
    std::vector<ttnn::Tensor> ttnn_tensors;
    ttnn_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        ttnn_tensors.push_back(tensor->get_value());
    }

    auto result_tensor = ttnn::concat(ttnn_tensors, dim);
    auto out = autograd::create_tensor(result_tensor);

    autograd::GradFunction grad = [tensors, out]() { throw std::runtime_error("concat backward not implemented yet"); };

    std::vector<autograd::NodeId> links;
    for (const auto& tensor : tensors) {
        if (tensor) {
            auto node = tensor->get_node();
            if (node.has_value()) {
                links.push_back(node.value());
            }
        }
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
