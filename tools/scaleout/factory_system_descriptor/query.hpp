// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

namespace tt::scaleout_tools {

class FsdQuery;

// Names one hierarchy tier of one subgroup: the links whose endpoints' instance_path LCP is exactly `depth`
// and which lie in the depth-`depth` subgroup containing `member_hostname`.
//
// Passed to validation so the expected link set comes from the FSD hierarchy rather than from whatever
// discovery happened to find. The difference matters for a partial GSD: scoping by discovered hosts silently
// drops an absent host's links, while scoping by the hierarchy keeps them in scope so they surface as missing.
struct HierarchyTierScope {
    const FsdQuery* query = nullptr;  // must outlive the call
    std::string member_hostname;      // any host of the subgroup; usually the local host
    uint32_t depth = 0;
};

// Read-only query interface over a FactorySystemDescriptor.
// Builds a hostname -> host_id index once; the referenced proto must outlive this object.
class FsdQuery {
public:
    explicit FsdQuery(const fsd::proto::FactorySystemDescriptor& fsd);
    FsdQuery(fsd::proto::FactorySystemDescriptor&&) = delete;

    // Hostname of a host by positional host_id (the inverse of the hostname index).
    const std::string& get_hostname(uint32_t host_id) const;

    // Full instance_path segments of a host (root -> host).
    std::vector<std::string> get_instance_path(uint32_t host_id) const;
    std::vector<std::string> get_instance_path(const std::string& hostname) const;

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

    // Partition hosts into subgroups by their depth-`depth` instance_path prefix: each returned group is
    // the host_ids sharing the same first `depth` segments (one hierarchy node at that level). Hosts with
    // fewer than `depth` segments group by their full path. Groups are ordered deterministically by prefix,
    // host_ids ascending within a group. This is the subgroup set for per-hierarchy-node phased discovery.
    std::vector<std::vector<uint32_t>> hierarchy_partition(uint32_t depth) const;

    // Index into hierarchy_partition(depth) of the subgroup containing `hostname`. Deterministic across ranks
    // (same FSD everywhere), which makes it usable directly as a collective split color.
    uint32_t subgroup_index(const std::string& hostname, uint32_t depth) const;

    // Hostnames of the depth-`depth` subgroup containing `hostname` (including it). Known a priori from the
    // FSD, so callers can tell "this host should be here" apart from "this host was discovered" — the
    // distinction that lets an absent host be reported rather than silently dropped from scope.
    std::set<std::string> subgroup_hosts(const std::string& hostname, uint32_t depth) const;

    // Number of eth_connections this tier holds `hostname`'s subgroup responsible for: those at exactly `depth`
    // with both endpoints inside the depth-`depth` subgroup containing `hostname`. Matches the scope that
    // validate_fsd_against_gsd applies, so it is the expected-connection count for one subgroup at one tier.
    // Counted from the FSD alone (no discovery), so it is the a-priori figure a tier should have covered.
    uint32_t count_subgroup_tier_connections(const std::string& hostname, uint32_t depth) const;

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
