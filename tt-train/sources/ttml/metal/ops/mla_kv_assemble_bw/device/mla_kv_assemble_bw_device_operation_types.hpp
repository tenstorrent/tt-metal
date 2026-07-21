// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_kv_assemble_bw::device {

struct MLAKVAssembleBwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};
};

struct MLAKVAssembleBwInputs {
    const ttnn::Tensor& dK;
    const ttnn::Tensor& dV;
};

using operation_attributes_t = MLAKVAssembleBwParams;
using tensor_args_t = MLAKVAssembleBwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::mla_kv_assemble_bw::device
