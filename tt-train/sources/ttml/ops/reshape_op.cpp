// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, std::span<uint32_t> shape) {
    auto out = autograd::create_tensor();

    // Convert span to SmallVector for ttnn::Shape construction
    // ttnn::Shape expects SmallVector<uint32_t> (which is the Container type)
    ttnn::SmallVector<uint32_t> shape_vec(shape.begin(), shape.end());

    // Construct ttnn::Shape from SmallVector
    ttnn::Shape ttnn_shape(shape_vec);

    // Capture original shape at forward time for backward pass
    auto original_shape = tensor->get_value().logical_shape();

    // Perform reshape operation
    out->set_value(ttnn::reshape(tensor->get_value(), ttnn_shape));

    // Backward pass: reshape gradient back to original shape
    autograd::GradFunction grad = [tensor, out, original_shape]() {
        auto grad_reshaped = ttnn::reshape(out->get_grad(), original_shape);
        tensor->add_grad(grad_reshaped);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
