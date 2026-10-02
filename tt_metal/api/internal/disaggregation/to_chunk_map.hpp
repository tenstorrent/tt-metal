// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include <internal/disaggregation/kv_chunk_address_table.hpp>
#include <internal/disaggregation/cache_tensor_layout_spec.hpp>

namespace tt::tt_metal::internal::disaggregation {

// Runtime extents the chunk map is enumerated over. Mirrors the device-free reference factory's
// `Geometry` (kv_manager/tests/kv_layout_spec_smoke/kv_layout_spec.py): the (layer, slot, position)
// grid. The DRAM base address is per-cache (CacheConfig::base_addr); the fabric mesh is a call arg.
struct MapGeometry {
    uint32_t num_layers = 0;
    uint32_t num_slots = 0;
    uint32_t max_seq_len = 0;         // in tokens
    uint32_t position_step = kTile;   // token stride between enumerated positions
};

// One co-resident cache to address: the residence-agnostic spec, the mesh distribution the tensor was
// allocated with (TensorTopology), the op/engine generation policy, and the allocated buffer address.
// The addresser reads shape/dtype/shard-spec off `spec.tensor`, the mesh geometry + device coords off
// `topology`, the DRAM bank count + bank order off `spec.arch` (num_dram_banks / optimal_bank_order),
// `mesh_id` from the call args, and everything else from `policy`.
struct CacheConfig {
    KvLayoutSpec spec;
    TensorTopology topology;
    GenerationPolicy policy;
    uint64_t base_addr = 0;
};

// The commonized factory. `configs` is a model expressed as an ordered LIST of co-resident caches
// (K/V + indexer, per layer-type). ONE dispatch over bank_scheme x temporal x distribution fills a
// KvChunkAddressTable: each (config, layer, slot, position[, head]) resolves to a KvCacheLocation whose
// noc_addr encodes (bank_id << 32) | per_bank_offset and whose device_group_index references the replica
// set of mesh coordinates the placement implies.
//
// One config per cache; a per-head cache (BLOCK/CYCLIC GQA) fans its heads into ADDITIONAL slots so
// heads and slots share one flat slot axis: config slot = head * num_slots + slot.
KvChunkAddressTable to_chunk_map(
    const std::vector<CacheConfig>& configs,
    tt::tt_fabric::MeshId mesh_id,
    const MapGeometry& geometry);

}  // namespace tt::tt_metal::internal::disaggregation
