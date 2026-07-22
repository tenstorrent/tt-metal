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

private:
    uint32_t host_id_for(const std::string& hostname) const;

    const fsd::proto::FactorySystemDescriptor& fsd_;
    std::unordered_map<std::string, uint32_t> hostname_to_host_id_;
};

}  // namespace tt::scaleout_tools
