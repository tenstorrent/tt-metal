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

// `value`'s placement on each axis of its device mesh. A collapsed 1-D label (the default mappers' `{N}, [placement]`)
// is read as that placement on every mesh axis of size > 1. A label that is neither one placement per axis nor a
// collapsed label covering the whole mesh is an error, not a guess: guessing Replicate would reduce a shard, guessing
// Shard would skip a reduction.
ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_per_axis(
    const ttnn::Tensor& value);

// True when `value` is Shard rather than Replicate on mesh axis `axis`, read through placements_per_axis.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis);

// What the ranks along a mesh axis share, which decides how a gradient is combined across them.
enum class AxisKind {
    // Each rank computes the loss of its own samples (dp, fsdp, cp): the global gradient is the mean of the ranks'.
    Data,
    // The ranks compute one loss together (tp): a gradient that differs across them holds partial sums.
    Model,
};

// One entry per mesh axis; std::nullopt for an axis with no declared kind.
using AxisKinds = std::vector<std::optional<AxisKind>>;

struct GradientReduction {
    uint32_t axis;
    AxisKind kind;

    bool operator==(const GradientReduction&) const = default;
};

// The mesh axes along which `parameter`'s gradient must be reduced, in mesh-axis order, read from the parameter's and
// the gradient's TensorTopology. A gradient is reduced only along axes where its parameter is replicated:
// - Data axis: always, averaging. A mean of equal gradients is a no-op, so the gradient's label is not consulted.
// - Model axis: only when the gradient is sharded there (it differs across ranks), summing its partials.
// A gradient that differs along an axis with no declared kind is an error: averaging and summing would both guess.
std::vector<GradientReduction> gradient_reductions(const autograd::TensorPtr& parameter, const AxisKinds& kinds);

// Reduces every gradient along its `gradient_reductions`, leaving each one replicated wherever its parameter is.
void synchronize_gradients(const serialization::NamedParameters& parameters, const AxisKinds& kinds);
// The ParallelismContext's axes: ddp and cp are data axes, tp a model axis. A no-op without a context.
void synchronize_gradients(const serialization::NamedParameters& parameters);

}  // namespace ttml::core::distributed
