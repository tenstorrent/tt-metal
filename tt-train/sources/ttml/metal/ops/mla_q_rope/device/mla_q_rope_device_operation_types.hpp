// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::mla_q_rope::device {

struct MlaQRopeParams {
    uint32_t qk_nope_dim{};
    uint32_t qk_rope_dim{};

    static constexpr auto attribute_names = std::forward_as_tuple("qk_nope_dim", "qk_rope_dim");
    auto attribute_values() const {
        return std::forward_as_tuple(qk_nope_dim, qk_rope_dim);
    }
};

struct MlaQRopeInputs {
    const ttnn::Tensor& q_in;
    const ttnn::Tensor& cos_cache;
    const ttnn::Tensor& sin_cache;
    const ttnn::Tensor& trans_mat;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "q_in_dtype", "q_in_logical_shape", "cos_cache_logical_shape", "q_in", "cos_cache", "sin_cache", "trans_mat");
    auto attribute_values() const {
        return std::make_tuple(
            q_in.dtype(),
            std::cref(q_in.logical_shape()),
            std::cref(cos_cache.logical_shape()),
            std::cref(q_in),
            std::cref(cos_cache),
            std::cref(sin_cache),
            std::cref(trans_mat));
    }
};

using operation_attributes_t = MlaQRopeParams;
using tensor_args_t = MlaQRopeInputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::mla_q_rope::device
