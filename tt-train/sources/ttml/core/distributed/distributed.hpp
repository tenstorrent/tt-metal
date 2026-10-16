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

// True when `value` is Shard{...} rather than Replicate on mesh axis `axis` of the current device mesh. A collapsed
// 1-D label (the `{N}, [placement]` the default mappers produce) is expanded to one placement per mesh axis first:
// Replicate is Replicate on every axis, Shard{d} over the whole mesh is Shard{d} on every axis of size > 1. This
// decides which axes a gradient is all-reduced along, so a label that is neither one placement per mesh axis nor a
// collapsed label covering the mesh in row-major order (fewer shards than devices, a sub-mesh) is an error rather
// than a guess: guessing Replicate reduces a shard, guessing Shard skips a reduction.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis);

void synchronize_gradients(const serialization::NamedParameters& parameters);
void synchronize_gradients(const serialization::NamedParameters& parameters, const std::vector<uint32_t>& cluster_axes);

}  // namespace ttml::core::distributed
