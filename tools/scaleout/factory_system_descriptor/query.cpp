// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "query.hpp"

#include <algorithm>
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
}

std::vector<std::string> FsdQuery::longest_common_prefix(uint32_t host_id_a, uint32_t host_id_b) const {
    const auto num_hosts = static_cast<uint32_t>(fsd_.hosts().size());
    if (host_id_a >= num_hosts || host_id_b >= num_hosts) {
        throw std::out_of_range(
            fmt::format("host_id out of range (a={}, b={}, num_hosts={})", host_id_a, host_id_b, num_hosts));
    }
    const auto& path_a = fsd_.hosts()[host_id_a].instance_path();
    const auto& path_b = fsd_.hosts()[host_id_b].instance_path();

    std::vector<std::string> prefix;
    const int limit = std::min(path_a.size(), path_b.size());
    for (int i = 0; i < limit && path_a[i] == path_b[i]; ++i) {
        prefix.push_back(path_a[i]);
    }
    return prefix;
}

std::vector<std::string> FsdQuery::longest_common_prefix(
    const std::string& hostname_a, const std::string& hostname_b) const {
    return longest_common_prefix(host_id_for(hostname_a), host_id_for(hostname_b));
}

uint32_t FsdQuery::host_id_for(const std::string& hostname) const {
    auto it = hostname_to_host_id_.find(hostname);
    if (it == hostname_to_host_id_.end()) {
        throw std::runtime_error(fmt::format("Hostname '{}' not found in factory system descriptor", hostname));
    }
    return it->second;
}

}  // namespace tt::scaleout_tools
