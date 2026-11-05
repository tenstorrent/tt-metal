// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <functional>

#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;
using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;

// Averages `tensor` over the devices of `cluster_axes`. `reduce_dtype` is the dtype the values are summed in: unset or
// BFLOAT16 sums in the tensor's own dtype (bf16 gradients in tt-train); FLOAT32 sums a narrower tensor in fp32 and
// writes the mean back in the tensor's dtype, which keeps most of the low bits a bf16 sum rounds away at every hop. The
// fp32 sum is not exact fp32: the FPU aligns the smaller operand to the larger one with fewer fraction bits (#62595).
// Large fp32 sums over one axis take a reduce-scatter and an all-gather instead of an all-reduce; the route rule is at
// kFp32ReduceScatterMinElements in distributed.cpp.
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

// The mean of `count` gradients, which `gradient(i)` hands over one at a time (the 3-tier aggregator receives them from
// its workers this way, so only one is held besides the sum). With `reduce_dtype` FLOAT32, a narrower gradient is cast
// to fp32 before it is added, the sum and the division run in fp32, and the result is written in the first gradient's
// dtype once; unset or BFLOAT16 runs every op in the gradients' own dtype. `sum_over_devices` then all-reduces the mean
// over every device of the mesh, in the same dtype as the sum: a sum, not a mean, as the 3-tier aggregator has always
// done under DDP.
ttnn::Tensor average_gradients(
    uint32_t count,
    const std::function<ttnn::Tensor(uint32_t)>& gradient,
    std::optional<ttnn::DataType> reduce_dtype,
    bool sum_over_devices = false);

}  // namespace ttml::core::distributed
