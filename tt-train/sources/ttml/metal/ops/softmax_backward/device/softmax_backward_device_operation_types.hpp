// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal::ops::softmax_backward::device {

struct SoftmaxBackwardParams {
    uint32_t dim;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct SoftmaxBackwardInputs {
    ttnn::Tensor softmax_output;
    ttnn::Tensor upstream_grad;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    tt::tt_metal::KernelHandle unary_writer_kernel_id;
};

}  // namespace ttml::metal::ops::softmax_backward::device
