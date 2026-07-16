// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_bw::device {

struct MLAQKVAssembleBwParams {
    uint32_t n_heads{};
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};
    uint32_t v_dim{};

    static constexpr auto attribute_names = std::forward_as_tuple("n_heads", "qk_nope_dim", "qk_rope_dim", "v_dim");
    auto attribute_values() const {
        return std::forward_as_tuple(n_heads, qk_nope_dim, qk_rope_dim, v_dim);
    }
};

struct MLAQKVAssembleBwInputs {
    const ttnn::Tensor& dQ;
    const ttnn::Tensor& dK;
    const ttnn::Tensor& dV;

    static constexpr auto attribute_names =
        std::forward_as_tuple("dQ_dtype", "dQ_logical_shape", "dK_logical_shape", "dV_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(
            dQ.dtype(), std::cref(dQ.logical_shape()), std::cref(dK.logical_shape()), std::cref(dV.logical_shape()));
    }
};

using operation_attributes_t = MLAQKVAssembleBwParams;
using tensor_args_t = MLAQKVAssembleBwInputs;
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::mla_qkv_assemble_bw::device
