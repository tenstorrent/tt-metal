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

// True when `value` is Shard on mesh axis `axis` of the mesh it is on. A collapsed `{N}, [placement]` label counts as
// that placement on every axis of size > 1; a host tensor, or a label that is neither per-axis nor a collapsed label
// covering the mesh, is an error.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis);

// Averages every parameter's gradient along the mesh axes the parallelism context reduces (context parallel, then
// data parallel), skipping for each parameter any axis it is sharded on (see is_sharded_on_axis). A no-op when no
// parallelism context is initialized.
void synchronize_gradients(const serialization::NamedParameters& parameters);
// The same along `cluster_axes`: a parameter sharded on one of them is not averaged along it.
void synchronize_gradients(const serialization::NamedParameters& parameters, const std::vector<uint32_t>& cluster_axes);

}  // namespace ttml::core::distributed
