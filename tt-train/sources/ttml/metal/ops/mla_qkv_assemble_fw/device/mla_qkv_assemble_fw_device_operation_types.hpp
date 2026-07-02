// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_fw::device {

struct MLAQKVAssembleFwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};
};

struct MLAQKVAssembleFwInputs {
    const ttnn::Tensor& q_pre;
    const ttnn::Tensor& kv_up;
    const ttnn::Tensor& k_pe;
};

using operation_attributes_t = MLAQKVAssembleFwParams;
using tensor_args_t = MLAQKVAssembleFwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::mla_qkv_assemble_fw::device
