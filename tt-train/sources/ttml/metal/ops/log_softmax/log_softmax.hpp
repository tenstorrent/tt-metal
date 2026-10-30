// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {
// Row state consists of three FP32 column-vector tiles per input tile row:
// maximum, selected maximum index, and log1p(exponential tail).
std::vector<ttnn::Tensor> log_softmax_fw(const ttnn::Tensor& input);
ttnn::Tensor log_softmax_bw(const ttnn::Tensor& input, const ttnn::Tensor& upstream, const ttnn::Tensor& state);
}  // namespace ttml::metal
