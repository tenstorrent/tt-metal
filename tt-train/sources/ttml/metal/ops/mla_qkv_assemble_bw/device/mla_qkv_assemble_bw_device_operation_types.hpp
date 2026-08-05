// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_bw::device {

struct MLAQKVAssembleBwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};
};

struct MLAQKVAssembleBwInputs {
    const ttnn::Tensor& dQ;
    const ttnn::Tensor& dK;
    const ttnn::Tensor& dV;
};

using operation_attributes_t = MLAQKVAssembleBwParams;
using tensor_args_t = MLAQKVAssembleBwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::mla_qkv_assemble_bw::device
