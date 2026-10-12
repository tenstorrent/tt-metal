// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

using ttnn::DeviceComputeKernelConfig;
using ttnn::Tensor;

struct QkvCausalConv1dSiluBackwardParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    uint32_t channel_chunk_size;
    uint32_t sequence_splits;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct QkvCausalConv1dSiluBackwardInputs {
    Tensor grad_q;
    Tensor grad_k;
    Tensor grad_v;
    Tensor input;
    Tensor history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device
