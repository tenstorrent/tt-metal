// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttml::core::distributed {

// Merge shard topology from two sources onto dst: if either has Shard on an axis, dst gets it.
inline void propagate_topology(
    const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b, tt::tt_metal::Tensor& dst) {
    const auto& topo_a = a.tensor_topology();
    const auto& topo_b = b.tensor_topology();
    const auto& pa = topo_a.placements();
    const auto& pb = topo_b.placements();

    if (pa.size() <= 1 && pb.size() <= 1) {
        return;
    }

    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using ShardT = tt::tt_metal::distributed::MeshMapperConfig::Shard;

    size_t ndim = std::max(pa.size(), pb.size());
    ttsl::SmallVector<Placement> merged;
    for (size_t i = 0; i < ndim; ++i) {
        auto p_a = (i < pa.size()) ? pa[i] : Placement{tt::tt_metal::distributed::MeshMapperConfig::Replicate{}};
        auto p_b = (i < pb.size()) ? pb[i] : Placement{tt::tt_metal::distributed::MeshMapperConfig::Replicate{}};
        merged.push_back(std::holds_alternative<ShardT>(p_a) ? p_a : p_b);
    }

    tt::tt_metal::TensorTopology new_topo(
        topo_a.distribution_shape(), std::move(merged), {topo_a.mesh_coords().begin(), topo_a.mesh_coords().end()});

    dst.update_tensor_topology(new_topo);
}

// Single-source overload: copy topology from src to dst.
inline void propagate_topology(const tt::tt_metal::Tensor& src, tt::tt_metal::Tensor& dst) {
    const auto& placements = src.tensor_topology().placements();
    if (placements.size() <= 1) {
        return;
    }
    dst.update_tensor_topology(src.tensor_topology());
}

}  // namespace ttml::core::distributed
