// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

struct SwigluElemwiseBwParams {};

struct SwigluElemwiseBwInputs {
    ttnn::Tensor linear1;
    ttnn::Tensor gate;
    ttnn::Tensor dL_dprod;
    std::optional<ttnn::Tensor> preallocated_dL_dlinear1 = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dL_dgate = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "linear1_dtype",
        "linear1_logical_shape",
        "linear1_padded_shape",
        "gate_dtype",
        "gate_logical_shape",
        "gate_padded_shape",
        "dL_dprod_dtype",
        "dL_dprod_logical_shape",
        "dL_dprod_padded_shape",
        "dL_dlinear1_memcfg",
        "dL_dgate_memcfg",
        "linear1",
        "gate",
        "dL_dprod",
        "preallocated_dL_dlinear1",
        "preallocated_dL_dgate");
    auto attribute_values() const {
        const auto& dL_dlinear1_memcfg =
            preallocated_dL_dlinear1.has_value() ? preallocated_dL_dlinear1->memory_config() : linear1.memory_config();
        const auto& dL_dgate_memcfg =
            preallocated_dL_dgate.has_value() ? preallocated_dL_dgate->memory_config() : linear1.memory_config();
        return std::make_tuple(
            linear1.dtype(),
            std::cref(linear1.logical_shape()),
            std::cref(linear1.padded_shape()),
            gate.dtype(),
            std::cref(gate.logical_shape()),
            std::cref(gate.padded_shape()),
            dL_dprod.dtype(),
            std::cref(dL_dprod.logical_shape()),
            std::cref(dL_dprod.padded_shape()),
            dL_dlinear1_memcfg,
            dL_dgate_memcfg,
            std::cref(linear1),
            std::cref(gate),
            std::cref(dL_dprod),
            preallocated_dL_dlinear1,
            preallocated_dL_dgate);
    }
};

struct SwigluElemwiseBwResult {
    ttnn::Tensor dL_dlinear1;
    ttnn::Tensor dL_dgate;
};

using SwigluElemwiseBwOutputSpecs = std::vector<ttnn::TensorSpec>;

// Backward-compat aliases for in-flight branches.
using operation_attributes_t = SwigluElemwiseBwParams;
using tensor_args_t = SwigluElemwiseBwInputs;
using tensor_return_value_t = SwigluElemwiseBwResult;
using spec_return_value_t = SwigluElemwiseBwOutputSpecs;

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
