// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_fw::device {

struct MLAQKVAssembleFwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};

    static constexpr auto attribute_names = std::forward_as_tuple("n_heads", "qk_nope_dim", "qk_rope_dim", "v_dim");
    auto attribute_values() const {
        return std::forward_as_tuple(n_heads, qk_nope_dim, qk_rope_dim, v_dim);
    }
};

struct MLAQKVAssembleFwInputs {
    const ttnn::Tensor& q_pre;
    const ttnn::Tensor& kv_up;
    const ttnn::Tensor& k_pe;

    static constexpr auto attribute_names =
        std::forward_as_tuple("kv_up_dtype", "q_pre_logical_shape", "kv_up_logical_shape", "k_pe_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(
            kv_up.dtype(),
            std::cref(q_pre.logical_shape()),
            std::cref(kv_up.logical_shape()),
            std::cref(k_pe.logical_shape()));
    }
};

using operation_attributes_t = MLAQKVAssembleFwParams;
using tensor_args_t = MLAQKVAssembleFwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::mla_qkv_assemble_fw::device
