// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "query.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

#include <fmt/format.h>

#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

FsdQuery::FsdQuery(const fsd::proto::FactorySystemDescriptor& fsd) : fsd_(fsd) {
    const auto& hosts = fsd_.hosts();
    hostname_to_host_id_.reserve(hosts.size());
    for (int i = 0; i < hosts.size(); ++i) {
        auto [it, inserted] = hostname_to_host_id_.emplace(hosts[i].hostname(), static_cast<uint32_t>(i));
        if (!inserted) {
            throw std::runtime_error(
                fmt::format("Duplicate hostname '{}' in factory system descriptor", hosts[i].hostname()));
        }
    }

    // Cache the distinct hierarchy tiers (deepest-first) across all eth connections.
    std::set<uint32_t, std::greater<>> tiers;
    for (const auto& connection : fsd_.eth_connections().connection()) {
        tiers.insert(lcp_length(connection.endpoint_a().host_id(), connection.endpoint_b().host_id()));
    }
    hierarchy_tiers_.assign(tiers.begin(), tiers.end());
    max_hierarchy_depth_ = hierarchy_tiers_.empty() ? 0 : hierarchy_tiers_.front();
}

const std::string& FsdQuery::get_hostname(uint32_t host_id) const {
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    if (host_id >= num_hosts) {
        throw std::out_of_range(fmt::format("host_id out of range (id={}, num_hosts={})", host_id, num_hosts));
    }
    return fsd_.hosts()[host_id].hostname();
}

std::vector<std::string> FsdQuery::get_instance_path(uint32_t host_id) const {
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    if (host_id >= num_hosts) {
        throw std::out_of_range(fmt::format("host_id out of range (id={}, num_hosts={})", host_id, num_hosts));
    }
    const auto& path = fsd_.hosts()[host_id].instance_path();
    return std::vector<std::string>(path.begin(), path.end());
}

std::vector<std::string> FsdQuery::get_instance_path(const std::string& hostname) const {
    return get_instance_path(host_id_for(hostname));
}

uint32_t FsdQuery::lcp_length(uint32_t host_id_a, uint32_t host_id_b) const {
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    if (host_id_a >= num_hosts || host_id_b >= num_hosts) {
        throw std::out_of_range(
            fmt::format("host_id out of range (a={}, b={}, num_hosts={})", host_id_a, host_id_b, num_hosts));
    }
    const auto& path_a = fsd_.hosts()[host_id_a].instance_path();
    const auto& path_b = fsd_.hosts()[host_id_b].instance_path();

    const int limit = std::min(path_a.size(), path_b.size());
    int n = 0;
    while (n < limit && path_a[n] == path_b[n]) {
        ++n;
    }
    return static_cast<uint32_t>(n);
}

std::vector<std::string> FsdQuery::longest_common_prefix(uint32_t host_id_a, uint32_t host_id_b) const {
    const uint32_t n = lcp_length(host_id_a, host_id_b);
    const auto& path_a = fsd_.hosts()[host_id_a].instance_path();
    std::vector<std::string> prefix;
    prefix.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        prefix.push_back(path_a[i]);
    }
    return prefix;
}

std::vector<std::string> FsdQuery::longest_common_prefix(
    const std::string& hostname_a, const std::string& hostname_b) const {
    return longest_common_prefix(host_id_for(hostname_a), host_id_for(hostname_b));
}

uint32_t FsdQuery::hierarchy_depth(uint32_t host_id_a, uint32_t host_id_b) const {
    return lcp_length(host_id_a, host_id_b);
}

uint32_t FsdQuery::hierarchy_depth(const std::string& hostname_a, const std::string& hostname_b) const {
    return lcp_length(host_id_for(hostname_a), host_id_for(hostname_b));
}

std::vector<std::vector<uint32_t>> FsdQuery::hierarchy_partition(uint32_t depth) const {
    // Group host_ids by their first `depth` instance_path segments (fewer if the path is shorter).
    std::map<std::vector<std::string>, std::vector<uint32_t>> groups;
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    for (uint32_t id = 0; id < num_hosts; ++id) {
        const auto& path = fsd_.hosts()[id].instance_path();
        const int take = std::min(static_cast<int>(depth), path.size());
        groups[std::vector<std::string>(path.begin(), path.begin() + take)].push_back(id);
    }
    std::vector<std::vector<uint32_t>> partition;
    partition.reserve(groups.size());
    for (auto& [prefix, ids] : groups) {
        partition.push_back(std::move(ids));
    }
    return partition;
}

// Shared by subgroup_index / subgroup_hosts: locate the hierarchy_partition group holding `hostname`.
// Compares depth-prefixes rather than searching for the id so that the grouping rule (including the
// shorter-than-depth path case) lives in exactly one place: hierarchy_partition.
uint32_t FsdQuery::subgroup_index(const std::string& hostname, uint32_t depth) const {
    const auto prefix_of = [depth](const std::vector<std::string>& path) {
        return std::vector<std::string>(path.begin(), path.begin() + std::min<size_t>(depth, path.size()));
    };
    const auto my_prefix = prefix_of(get_instance_path(hostname));

    const auto partition = hierarchy_partition(depth);
    for (size_t i = 0; i < partition.size(); ++i) {
        if (prefix_of(get_instance_path(partition[i].front())) == my_prefix) {
            return static_cast<uint32_t>(i);
        }
    }
    // hierarchy_partition covers every host in the FSD, so a host present in the index is always in some group.
    throw std::runtime_error(fmt::format("Host '{}' not found in any hierarchy subgroup at depth {}", hostname, depth));
}

std::set<std::string> FsdQuery::subgroup_hosts(const std::string& hostname, uint32_t depth) const {
    // hierarchy_partition returns by value; bind the whole partition to a named local before indexing into it.
    // Binding a reference straight to `hierarchy_partition(depth)[i]` does NOT extend the temporary's lifetime
    // (the reference binds to operator[]'s result, not the temporary), leaving `group` dangling.
    const auto partition = hierarchy_partition(depth);
    const auto& group = partition[subgroup_index(hostname, depth)];
    std::set<std::string> hosts;
    for (uint32_t id : group) {
        hosts.insert(fsd_.hosts()[id].hostname());
    }
    return hosts;
}

uint32_t FsdQuery::count_subgroup_tier_connections(const std::string& hostname, uint32_t depth) const {
    const auto hosts = subgroup_hosts(hostname, depth);
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    uint32_t count = 0;
    for (const auto& connection : fsd_.eth_connections().connection()) {
        const uint32_t host_a = connection.endpoint_a().host_id();
        const uint32_t host_b = connection.endpoint_b().host_id();
        if (host_a >= num_hosts || host_b >= num_hosts) {
            continue;
        }
        // An LCP of exactly `depth` means both endpoints share the depth-`depth` prefix, so they are necessarily in
        // the SAME subgroup — testing one endpoint's membership is sufficient.
        if (lcp_length(host_a, host_b) == depth && hosts.contains(fsd_.hosts()[host_a].hostname())) {
            ++count;
        }
    }
    return count;
}

uint32_t FsdQuery::host_id_for(const std::string& hostname) const {
    auto it = hostname_to_host_id_.find(hostname);
    if (it == hostname_to_host_id_.end()) {
        throw std::runtime_error(fmt::format("Hostname '{}' not found in factory system descriptor", hostname));
    }
    return it->second;
}

}  // namespace tt::scaleout_tools
