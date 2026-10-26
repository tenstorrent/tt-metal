// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;
using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;

// Averages `tensor` over the devices of `cluster_axes`. `reduce_dtype` is the dtype the values are summed in: unset or
// BFLOAT16 sums in the tensor's own dtype (bf16 gradients in tt-train), FLOAT32 sums in fp32 and casts the mean back to
// the tensor's dtype, which keeps the low bits a bf16 sum loses at every hop.
ttnn::Tensor synchronize_tensor(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

void synchronize_gradients(
    const serialization::NamedParameters& parameters, std::optional<ttnn::DataType> reduce_dtype = std::nullopt);
void synchronize_gradients(
    const serialization::NamedParameters& parameters,
    const std::vector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

}  // namespace ttml::core::distributed
