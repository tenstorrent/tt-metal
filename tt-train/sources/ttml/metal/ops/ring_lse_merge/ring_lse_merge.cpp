// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_lse_merge.hpp"

#include "device/ring_lse_merge_device_operation.hpp"

namespace ttml::metal {

void ring_lse_merge(
    const ttnn::Tensor& output_accum,
    const ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates) {
    ttnn::prim::ttml_ring_lse_merge(output_accum, running_lse, step_output, step_intermediates);
}

}  // namespace ttml::metal
