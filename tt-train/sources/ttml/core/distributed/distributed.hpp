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
// the tensor's dtype, which keeps the low bits a bf16 sum loses at every hop. A large tensor reduced over one axis
// takes an fp32 reduce-scatter and a bf16 all-gather instead of an fp32 all-reduce; both sum in fp32.
ttnn::Tensor synchronize_tensor(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

// True when `value` is Shard on mesh axis `axis` of the mesh it is on. A collapsed `{N}, [placement]` label counts as
// that placement on every axis of size > 1; a host tensor, or a label that is neither per-axis nor a collapsed label
// covering the mesh, is an error.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis);

// Averages every parameter's gradient along the mesh axes the parallelism context reduces (context parallel, then
// data parallel), skipping for each parameter any axis it is sharded on (see is_sharded_on_axis), summing in
// `reduce_dtype` as synchronize_tensor does. Unset takes the training config's precision.reduce_dtype
// (autograd::AutoContext::get_gradient_reduce_dtype); a value passed here wins over it. A no-op when no parallelism
// context is initialized.
void synchronize_gradients(
    const serialization::NamedParameters& parameters, std::optional<ttnn::DataType> reduce_dtype = std::nullopt);
// The same along `cluster_axes`: a parameter sharded on one of them is not averaged along it.
void synchronize_gradients(
    const serialization::NamedParameters& parameters,
    const std::vector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

}  // namespace ttml::core::distributed
