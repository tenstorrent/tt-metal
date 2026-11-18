// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::log_softmax::device {

struct LogSoftmaxParams {
    bool backward = false;
};

struct LogSoftmaxInputs {
    ttnn::Tensor input;
    // Backward only: the upstream gradient and the forward's row state.
    std::optional<ttnn::Tensor> upstream;
    std::optional<ttnn::Tensor> state;
};

using operation_attributes_t = LogSoftmaxParams;
using tensor_args_t = LogSoftmaxInputs;

// Forward: {output, row state}. Backward: {input gradient}.
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::log_softmax::device
