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

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor, const ttsl::SmallVector<uint32_t>& cluster_axes);

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

// Relabels `value` with `placements`, one per mesh axis. Ops call it when ttnn's default label, which copies each
// input's shard dim, misdescribes their output.
void set_placements_per_axis(
    ttnn::Tensor& value, const ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>& placements);

// Relabels `grad`, a gradient of `parameter`, on the mesh axes `parameter` is sharded on.
void label_grad_like_parameter_shards(const ttnn::Tensor& parameter, ttnn::Tensor& grad);

// Averages every parameter's gradient along the mesh axes the parallelism context reduces (context parallel, then
// data parallel), skipping for each parameter any axis it is sharded on (see is_sharded_on_axis). A no-op when no
// parallelism context is initialized.
void synchronize_gradients(const serialization::NamedParameters& parameters);
// The same along `cluster_axes`: a parameter sharded on one of them is not averaged along it.
void synchronize_gradients(const serialization::NamedParameters& parameters, const std::vector<uint32_t>& cluster_axes);

}  // namespace ttml::core::distributed
