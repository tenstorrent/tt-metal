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
// BFLOAT16 sums in the tensor's own dtype (bf16 gradients in tt-train); FLOAT32 sums a narrower tensor in fp32 and
// writes the mean back in the tensor's dtype, which keeps most of the low bits a bf16 sum rounds away at every hop. The
// fp32 sum is not exact fp32: the FPU aligns the smaller operand to the larger one with fewer fraction bits (#62595).
// A FLOAT32 sum over one axis of a narrower tensor with at least 2^19 elements per device, whose last or second-to-last
// dim splits into whole-tile slices no other mesh axis already splits, takes an fp32 reduce-scatter and an all-gather
// in the tensor's own dtype; every other sum takes an all-reduce per axis.
ttnn::Tensor synchronize_tensor(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

// Average each initialized gradient over the parallelism context's DDP and CP axes (first overload; an axis the
// parameter is sharded on is skipped, FSDP reduces those) or over `cluster_axes`, summing in `reduce_dtype` as
// synchronize_tensor does. Unset takes the training config's precision.reduce_dtype
// (autograd::AutoContext::get_gradient_reduce_dtype); a value passed here wins over it.
void synchronize_gradients(
    const serialization::NamedParameters& parameters, std::optional<ttnn::DataType> reduce_dtype = std::nullopt);
void synchronize_gradients(
    const serialization::NamedParameters& parameters,
    const std::vector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

}  // namespace ttml::core::distributed
