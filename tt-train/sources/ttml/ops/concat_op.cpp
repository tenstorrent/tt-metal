#include "concat_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr concat(const std::vector<autograd::TensorPtr>& tensors, int32_t dim) {
    std::vector<ttnn::Tensor> ttnn_tensors;
    ttnn_tensors.reserve(tensors.size());

    std::vector<uint32_t> sizes_at_dim;
    sizes_at_dim.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        ttnn_tensors.push_back(tensor->get_value());
        sizes_at_dim.push_back(tensor->get_value().logical_shape()[dim]);
    }

    auto result_tensor = ttnn::concat(ttnn_tensors, dim);
    auto out = autograd::create_tensor(result_tensor);

    autograd::GradFunction grad = [tensors, dim, sizes_at_dim, out]() {
        auto grad_output = out->get_grad();
        auto grad_shape = grad_output.logical_shape();

        uint32_t offset = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto size_at_dim = sizes_at_dim[i];

            ttnn::SmallVector<uint32_t> start(4, 0);
            ttnn::SmallVector<uint32_t> end(4);
            ttnn::SmallVector<uint32_t> step(4, 1);

            for (size_t d = 0; d < 4; ++d) {
                end[d] = grad_shape[d];
            }

            start[dim] = offset;
            end[dim] = offset + size_at_dim;

            auto grad_slice = ttnn::slice(grad_output, start, end, step);
            tensors[i]->add_grad(grad_slice);

            offset += size_at_dim;
        }
    };

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
