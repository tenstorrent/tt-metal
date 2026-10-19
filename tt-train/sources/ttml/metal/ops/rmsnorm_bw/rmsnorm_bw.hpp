// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,  // intermediate from fw
    const ttnn::Tensor& dL_dout_tensor);

// First half of the backward for a row whose last dim is sharded across devices: sum(x * gamma * dL_dout) over
// the local shard, [B,1,S,32] fp32 with the sum in column 0 and zeros elsewhere. All-gather it along the last dim
// and pass the result to rmsnorm_bw_external_scale.
ttnn::Tensor rmsnorm_bw_local_sum(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& gamma_tensor, const ttnn::Tensor& dL_dout_tensor);

// Second half: dL/dx and dL/dgamma of the local shard from the gathered local sums of all shards of the row.
// full_width is the width of the unsharded row.
std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw_external_scale(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& gathered_local_sums,
    uint32_t full_width);

}  // namespace ttml::metal
