// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <unordered_map>
#include "assert.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "umd/device/types/cluster_descriptor_types.h"

namespace tt::tt_metal::dispatch {

// Keeps track of the FD Topology for activated device pools
class FDTopologyManager {
private:
    // Similar to boost hash combine
    struct ChipIdHash {
        std::size_t operator()(const std::set<chip_id_t>& s) const {
            std::size_t hash = 0;
            for (const auto& id : s) {
                hash ^= std::hash<chip_id_t>{}(id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    ~FDTopologyManager() = default;
    FDTopologyManager() = default;

    std::unordered_map<std::set<chip_id_t>, FDTopologyGraph, ChipIdHash> topology;
    std::mutex lock;

public:
    FDTopologyManager& operator=(const FDTopologyManager&) = delete;
    FDTopologyManager& operator=(FDTopologyManager&& other) noexcept = delete;
    FDTopologyManager(const FDTopologyManager&) = delete;
    FDTopologyManager(FDTopologyManager&& other) noexcept = delete;

    static FDTopologyManager& instance() noexcept {
        static FDTopologyManager _inst;
        return _inst;
    }

    // Initialize topology graph for given device_ids
    void initialize_topology(const std::set<chip_id_t>& active_device_ids, uint32_t num_hw_cqs) {
        std::lock_guard<std::mutex> guard(this->lock);
        if (this->topology.contains(active_device_ids)) {
            // Reset the topology for these devices. num_hw_cqs may be different
            // need to track num_hw_cqs are well to re-use
            this->topology.erase(active_device_ids);
        }

        topology[active_device_ids] = std::move(populate_fd_kernels(active_device_ids, num_hw_cqs));
    }

    // Delete all topologies
    void reset() {
        std::lock_guard<std::mutex> guard(this->lock);
        this->topology.clear();
    }

    // Returns the topology for a pool of active devices configured by DevicePool
    FDTopologyGraph& get_topology(const std::set<chip_id_t>& active_device_ids) {
        if (this->topology.empty() || !this->topology.contains(active_device_ids)) {
            TT_THROW("Topology is not configured for given devices. Need to call initialize_topology");
        }
        return this->topology[active_device_ids];
    }
};

}  // namespace tt::tt_metal::dispatch
