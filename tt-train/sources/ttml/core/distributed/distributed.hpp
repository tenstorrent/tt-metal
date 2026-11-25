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

// Throws std::invalid_argument (ValueError in Python) unless `reduce_dtype` is unset, BFLOAT16 or FLOAT32, the dtypes
// a gradient sum runs in.
void check_reduce_dtype(std::optional<ttnn::DataType> reduce_dtype);

// The dtype a gradient sum runs in: `reduce_dtype` once check_reduce_dtype accepts it, else the training config's
// precision.reduce_dtype that apply_precision_config stores (autograd::AutoContext::get_gradient_reduce_dtype), which
// may be unset too. synchronize_gradients and FSDP's backward hook resolve an unset value with it;
// synchronize_tensor takes reduce_dtype only as an argument.
std::optional<ttnn::DataType> resolve_reduce_dtype(std::optional<ttnn::DataType> reduce_dtype);

// The tensor dim synchronize_tensor's reduce-scatter route scatters `tensor` along for these arguments, or none when
// the sum takes the all-reduce. The route is taken only by a sum in fp32 of a narrower tensor over one mesh axis, of at
// least 2^19 elements per device, along a dim that splits into whole-tile slices for that axis's devices and that no
// other mesh axis already splits. A tensor that is not on the device mesh (a host or deallocated one) is a TT_FATAL.
std::optional<int> reduce_scatter_route_dim(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

// Averages `tensor` over the devices of `cluster_axes`. `reduce_dtype` is the dtype the values are summed in: unset or
// BFLOAT16 sums in the tensor's own dtype (bf16 gradients in tt-train); FLOAT32 sums a narrower tensor in fp32 and
// writes the mean back in the tensor's dtype, which keeps most of the low bits a bf16 sum rounds away at every hop. The
// fp32 sum is not exact fp32: the FPU aligns the smaller operand to the larger one with fewer fraction bits (#62595).
// Large fp32 sums over one axis take a reduce-scatter and an all-gather instead of an all-reduce:
// reduce_scatter_route_dim() says which.
ttnn::Tensor synchronize_tensor(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype = std::nullopt);

// `value`'s placement on each axis of its device mesh. A collapsed 1-D label, such as the default mappers'
// `{N}, [placement]`, means that placement on every axis of size > 1. Any other label that does not cover the mesh
// is an error.
ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_per_axis(
    const ttnn::Tensor& value);

// The same reading of the label `config` gives a tensor on a mesh of `mesh_shape`, for a tensor not yet mapped (a lazy
// parameter).
ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_per_axis(
    const tt::tt_metal::distributed::MeshMapperConfig& config, const tt::tt_metal::distributed::MeshShape& mesh_shape);

// Whether `value` is sharded on mesh axis `axis`, as placements_per_axis reads it.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis);

// The dim of a rank-`rank` tensor that `placement` shards along, a negative dim counted from the end, or nullopt where
// it replicates.
std::optional<int> shard_dim(const tt::tt_metal::distributed::MeshMapperConfig::Placement& placement, int rank);

// Relabels `value` with `placements`, one per mesh axis. Ops call it when ttnn's default label, which copies each
// input's shard dim, misdescribes their output.
void set_placements_per_axis(
    ttnn::Tensor& value, const ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>& placements);

// Relabels `grad`, a gradient of `parameter`, on the mesh axes `parameter` is sharded on.
void label_grad_like_parameter_shards(const ttnn::Tensor& parameter, ttnn::Tensor& grad);

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
