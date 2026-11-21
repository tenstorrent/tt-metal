// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal {
// Row state consists of three FP32 column-vector tiles per input tile row:
// maximum, selected maximum index, and log1p(exponential tail).
// Whether the fused kernels take `input` (along its last axis): a non-empty rank-4 BF16 or FP32 tensor of supported
// tiles in interleaved DRAM.
bool log_softmax_supported(const ttnn::Tensor& input);
std::vector<ttnn::Tensor> log_softmax_fw(const ttnn::Tensor& input);
// `upstream` may be in any memory configuration, layout or dtype; it is converted to the input's tiles in DRAM.
ttnn::Tensor log_softmax_bw(const ttnn::Tensor& input, const ttnn::Tensor& upstream, const ttnn::Tensor& state);
}  // namespace ttml::metal
