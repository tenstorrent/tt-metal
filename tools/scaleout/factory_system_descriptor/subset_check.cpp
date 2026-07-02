// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subset_check.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <google/protobuf/text_format.h>

#include <board/board.hpp>

#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

namespace {

// Load an FSD textproto into a hostname-keyed, endpoint-canonicalized set of channel connections.
// Keying by hostname (not the FSD's per-file host_id index) is what makes two FSDs with different
// host orderings comparable.
std::set<PhysicalChannelConnection> load_fsd_connections(const std::string& fsd_path) {
    std::ifstream fsd_file(fsd_path);
    if (!fsd_file.is_open()) {
        throw std::runtime_error("Failed to open FSD file: " + fsd_path);
    }
    std::string contents((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());

    fsd::proto::FactorySystemDescriptor fsd;
    if (!google::protobuf::TextFormat::ParseFromString(contents, &fsd)) {
        throw std::runtime_error("Failed to parse FSD protobuf from file: " + fsd_path);
    }

    const auto& hosts = fsd.hosts();
    auto endpoint = [&](const auto& e) {
        if (e.host_id() >= static_cast<uint32_t>(hosts.size())) {
            throw std::runtime_error(
                "FSD '" + fsd_path + "' references host_id " + std::to_string(e.host_id()) + " but only has " +
                std::to_string(hosts.size()) + " hosts");
        }
        return PhysicalChannelEndpoint{
            hosts[e.host_id()].hostname(), TrayId(e.tray_id()), AsicChannel{e.asic_location(), ChanId(e.chan_id())}};
    };

    std::set<PhysicalChannelConnection> connections;
    for (const auto& connection : fsd.eth_connections().connection()) {
        auto a = endpoint(connection.endpoint_a());
        auto b = endpoint(connection.endpoint_b());
        connections.emplace(std::min(a, b), std::max(a, b));
    }
    return connections;
}

}  // namespace

std::set<PhysicalChannelConnection> missing_skinny_connections(
    const std::string& skinny_fsd_path, const std::string& good_fsd_path) {
    auto skinny = load_fsd_connections(skinny_fsd_path);
    auto good = load_fsd_connections(good_fsd_path);

    std::set<PhysicalChannelConnection> missing;
    std::set_difference(skinny.begin(), skinny.end(), good.begin(), good.end(), std::inserter(missing, missing.end()));
    return missing;
}

}  // namespace tt::scaleout_tools
