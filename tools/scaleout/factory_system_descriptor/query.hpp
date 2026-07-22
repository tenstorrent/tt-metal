// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

namespace tt::scaleout_tools {

// Read-only query interface over a FactorySystemDescriptor.
// Builds a hostname -> host_id index once; the referenced proto must outlive this object.
class FsdQuery {
public:
    explicit FsdQuery(const fsd::proto::FactorySystemDescriptor& fsd);

    // Longest common prefix of the two hosts' instance_path segments.
    // host_id indexes hosts positionally (the i-th host has host_id i), matching connection endpoints.
    std::vector<std::string> longest_common_prefix(uint32_t host_id_a, uint32_t host_id_b) const;
    std::vector<std::string> longest_common_prefix(const std::string& hostname_a, const std::string& hostname_b) const;

    // Hierarchy tier of a link = length of the two hosts' instance_path common prefix.
    // Larger depth = closer in the hierarchy (same node); smaller = farther (crosses the top).
    uint32_t hierarchy_depth(uint32_t host_id_a, uint32_t host_id_b) const;
    uint32_t hierarchy_depth(const std::string& hostname_a, const std::string& hostname_b) const;

    // Deepest (most-connected / closest) tier across all eth_connections. Cached; O(1).
    uint32_t max_hierarchy_depth() const { return max_hierarchy_depth_; }

    // Distinct tiers present across all eth_connections, sorted deepest-first (the phasing order).
    // front() == max_hierarchy_depth() when non-empty.
    const std::vector<uint32_t>& hierarchy_tiers_deepest_first() const { return hierarchy_tiers_; }

private:
    uint32_t host_id_for(const std::string& hostname) const;
    // Common prefix length of two hosts' instance_paths (shared by longest_common_prefix / hierarchy_depth).
    uint32_t lcp_length(uint32_t host_id_a, uint32_t host_id_b) const;

    const fsd::proto::FactorySystemDescriptor& fsd_;
    std::unordered_map<std::string, uint32_t> hostname_to_host_id_;
    std::vector<uint32_t> hierarchy_tiers_;  // distinct LCP depths over eth_connections, deepest-first
    uint32_t max_hierarchy_depth_ = 0;
};

}  // namespace tt::scaleout_tools
