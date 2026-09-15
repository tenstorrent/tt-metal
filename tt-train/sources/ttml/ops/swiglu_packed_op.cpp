// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/ops/swiglu_packed_bw/swiglu_packed_bw.hpp"
#include "metal/ops/swiglu_packed_fw/swiglu_packed_fw.hpp"

namespace ttml::ops {

autograd::TensorPtr swiglu_packed(const autograd::TensorPtr& packed) {
    auto h = ttml::metal::swiglu_packed_fw(packed->get_value());
    auto out = autograd::create_tensor(h);

    autograd::GradFunction grad = [packed, out]() {
        auto dL_dpacked = ttml::metal::swiglu_packed_bw(packed->get_value(), out->get_grad());
        packed->add_grad(dL_dpacked);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, packed));
    return out;
}

}  // namespace ttml::ops
