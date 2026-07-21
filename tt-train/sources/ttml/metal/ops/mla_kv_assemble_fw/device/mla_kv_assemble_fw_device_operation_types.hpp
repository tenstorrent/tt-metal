// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_kv_assemble_fw::device {

struct MLAKVAssembleFwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};
};

struct MLAKVAssembleFwInputs {
    const ttnn::Tensor& kv_up;
    const ttnn::Tensor& k_pe;
};

using operation_attributes_t = MLAKVAssembleFwParams;
using tensor_args_t = MLAKVAssembleFwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::mla_kv_assemble_fw::device
