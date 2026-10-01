// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// physical_descriptor_builder.cpp - Implementation of the offline FSD -> PSD builder.
//
// Ported from tenstorrent/tt-fabric-manager controller/physical_system_descriptor_builder.cpp (FSD overloads).
// Notable tt-metal adaptations vs. the Fabric Manager original:
//   - board_type string -> enum uses tt::scaleout_tools::get_board_type_from_string (the canonical board
//     library) instead of a hand-maintained table, so it stays in sync with tt::BoardType automatically.
//   - logging uses tt::LogFabric (Fabric Manager used its own tt::LogFabricManager category).

#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <google/protobuf/text_format.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include <protobuf/factory_system_descriptor.pb.h>   // tt::scaleout_tools::fsd::proto (from scaleout_tools)
#include <protobuf/physical_system_descriptor.pb.h>  // tt::fabric::proto (compiled into this lib)
#include "board/board.hpp"                           // tt::scaleout_tools::get_board_type_from_string

namespace tt::tt_metal::experimental::tt_fabric {

using ::tt::scaleout_tools::get_board_type_from_string;  // canonical board-type lookup (board/board.hpp)

namespace {

// FSD board_type string -> the uint32 the PSD proto stores (the raw tt::BoardType enum value). Reuses the
// canonical, reflection-based lookup from the board library so it never drifts from tt::BoardType.
//
// UBB_WORMHOLE is a compile-time alias of UBB (same enum value, 9). enchantum reflection (used by
// get_board_type_from_string) does not surface aliased enumerators, so it would reject "UBB_WORMHOLE" even
// though tt::BoardType defines it. Fabric Manager's FSDs use that spelling, so normalize it to the canonical
// "UBB" to preserve backward compatibility.
uint32_t fsd_board_type_to_psd(const std::string& name) {
    std::string upper = name;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if (upper == "UBB_WORMHOLE") {
        return static_cast<uint32_t>(get_board_type_from_string("UBB"));
    }
    return static_cast<uint32_t>(get_board_type_from_string(name));
}

// Partition the FSD hosts into connected components via union-find. Two hosts are connected when an
// eth_connection has its endpoints on different hosts; transitive connectivity chains through unions.
// Returns one entry per component (member host_ids ascending), components ordered by lowest member host name.
std::vector<std::vector<uint32_t>> partition_fsd_hosts_by_connectivity(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) {
    const int num_hosts = fsd.hosts_size();

    std::vector<size_t> parent(static_cast<size_t>(num_hosts));
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = i;
    }
    auto find = [&parent](size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](size_t a, size_t b) {
        size_t ra = find(a);
        size_t rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    };

    if (fsd.has_eth_connections()) {
        for (const auto& conn : fsd.eth_connections().connection()) {
            const uint32_t a = conn.endpoint_a().host_id();
            const uint32_t b = conn.endpoint_b().host_id();
            // Validate endpoint host_ids before the local-link skip, so a malformed connection (e.g. 99 -> 99)
            // is reported rather than silently dropped as if it were a same-host link.
            if (a >= static_cast<uint32_t>(num_hosts) || b >= static_cast<uint32_t>(num_hosts)) {
                throw std::runtime_error(fmt::format(
                    "FSD eth_connection references host_id {}/{} but only {} hosts are defined", a, b, num_hosts));
            }
            if (a == b) {
                continue;
            }
            unite(a, b);
        }
    }

    // Collect member host_ids per component, keyed by component root (ascending host_id order).
    std::map<size_t, std::vector<uint32_t>> components;
    for (int i = 0; i < num_hosts; ++i) {
        components[find(static_cast<size_t>(i))].push_back(static_cast<uint32_t>(i));
    }

    std::vector<std::vector<uint32_t>> ordered;
    ordered.reserve(components.size());
    for (auto& [root, members] : components) {
        ordered.push_back(std::move(members));
    }
    // Order groups by their lexicographically lowest member host NAME for deterministic output. (Members are
    // collected by host index, so the lowest hostname is not necessarily members.front().)
    auto min_hostname = [&fsd](const std::vector<uint32_t>& members) -> const std::string& {
        const std::string* best = &fsd.hosts(static_cast<int>(members.front())).hostname();
        for (uint32_t m : members) {
            const std::string& n = fsd.hosts(static_cast<int>(m)).hostname();
            if (n < *best) {
                best = &n;
            }
        }
        return *best;
    };
    std::sort(ordered.begin(), ordered.end(), [&min_hostname](const auto& a, const auto& b) {
        return min_hostname(a) < min_hostname(b);
    });
    return ordered;
}

// Build a sub-FSD restricted to `member_host_ids` (host indices into `src.hosts()`), with host_ids densely
// renumbered to 0..N-1 in the order given and board_types/eth_connections filtered and rewritten to match.
::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor build_sub_fsd(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& src,
    const std::vector<uint32_t>& member_host_ids) {
    ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor result;

    std::map<uint32_t, uint32_t> id_map;
    for (uint32_t new_id = 0; new_id < member_host_ids.size(); ++new_id) {
        const uint32_t old_id = member_host_ids[new_id];
        *result.add_hosts() = src.hosts(static_cast<int>(old_id));
        id_map[old_id] = new_id;
    }

    if (src.has_board_types()) {
        for (const auto& loc : src.board_types().board_locations()) {
            auto it = id_map.find(loc.host_id());
            if (it == id_map.end()) {
                continue;
            }
            auto* new_loc = result.mutable_board_types()->add_board_locations();
            *new_loc = loc;
            new_loc->set_host_id(it->second);
        }
    }

    if (src.has_eth_connections()) {
        for (const auto& conn : src.eth_connections().connection()) {
            auto it_a = id_map.find(conn.endpoint_a().host_id());
            auto it_b = id_map.find(conn.endpoint_b().host_id());
            if (it_a == id_map.end() || it_b == id_map.end()) {
                continue;
            }
            auto* new_conn = result.mutable_eth_connections()->add_connection();
            *new_conn = conn;
            new_conn->mutable_endpoint_a()->set_host_id(it_a->second);
            new_conn->mutable_endpoint_b()->set_host_id(it_b->second);
        }
    }

    return result;
}

}  // namespace

::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor load_factory_descriptor(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(fmt::format("Unable to open Factory System Descriptor file: {}", path));
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;
    if (!google::protobuf::TextFormat::ParseFromString(text, &fsd)) {
        throw std::runtime_error(fmt::format("Failed to parse Factory System Descriptor textproto: {}", path));
    }
    return fsd;
}

::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor filter_factory_descriptor(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd, const std::vector<std::string>& hostnames) {
    if (hostnames.empty()) {
        return fsd;  // no filter: return a full copy so callers can use the result uniformly
    }
    // Collect the host indices whose hostname is requested (original order), then reuse build_sub_fsd to
    // densely renumber hosts and filter board_types / eth_connections to that subset.
    const std::set<std::string> wanted(hostnames.begin(), hostnames.end());
    std::set<std::string> present;
    std::vector<uint32_t> member_host_ids;
    for (int i = 0; i < fsd.hosts_size(); ++i) {
        const std::string& name = fsd.hosts(i).hostname();
        if (wanted.contains(name)) {
            member_host_ids.push_back(static_cast<uint32_t>(i));
            present.insert(name);
        }
    }
    // Reject requested hostnames that aren't in the FSD (typo / wrong descriptor) rather than silently dropping.
    if (present.size() != wanted.size()) {
        std::string missing;
        for (const auto& h : wanted) {
            if (!present.contains(h)) {
                missing += (missing.empty() ? "" : ", ") + h;
            }
        }
        throw std::runtime_error(
            fmt::format("Host filter references hostnames not present in the Factory System Descriptor: {}", missing));
    }
    return build_sub_fsd(fsd, member_host_ids);
}

::tt::tt_metal::PhysicalSystemDescriptor build_physical_descriptor_from_file(
    const std::string& fsd_path, const std::vector<std::string>& host_filter) {
    // parse FSD -> (optional host filter) -> PSD proto -> wrap in the C++ PhysicalSystemDescriptor (no protos leak).
    auto fsd = load_factory_descriptor(fsd_path);
    if (!host_filter.empty()) {
        fsd = filter_factory_descriptor(fsd, host_filter);
    }
    return ::tt::tt_metal::PhysicalSystemDescriptor(build_physical_descriptor(fsd));
}

::tt::fabric::proto::PhysicalSystemDescriptor build_physical_descriptor(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) {
    ::tt::fabric::proto::PhysicalSystemDescriptor psd;

    const int num_hosts = fsd.hosts_size();
    log_info(
        tt::LogFabric,
        "Building PhysicalSystemDescriptor from FactorySystemDescriptor for {} hosts, {} board locations, {} "
        "connections",
        num_hosts,
        fsd.has_board_types() ? fsd.board_types().board_locations_size() : 0,
        fsd.has_eth_connections() ? fsd.eth_connections().connection_size() : 0);

    auto hostname_of = [&](uint32_t host_id) -> const std::string& {
        if (host_id >= static_cast<uint32_t>(num_hosts)) {
            throw std::runtime_error(
                fmt::format("FSD references host_id {} but only {} hosts are defined", host_id, num_hosts));
        }
        return fsd.hosts(static_cast<int>(host_id)).hostname();
    };

    // Build (host_id, tray_id) -> board_type uint32 lookup.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> host_tray_to_board_type;
    if (fsd.has_board_types()) {
        for (const auto& loc : fsd.board_types().board_locations()) {
            host_tray_to_board_type[{loc.host_id(), loc.tray_id()}] = fsd_board_type_to_psd(loc.board_type());
        }
    }

    // Enumerate ASICs from the union of eth_connections endpoints. ASICs that appear in board_types but have
    // zero eth connections are intentionally skipped; the topology mapper has no edges to place on them.
    using AsicKey = std::tuple<uint32_t, uint32_t, uint32_t>;  // (host_id, tray_id, asic_location)
    std::set<AsicKey> asic_keys;
    if (fsd.has_eth_connections()) {
        for (const auto& conn : fsd.eth_connections().connection()) {
            const auto& a = conn.endpoint_a();
            const auto& b = conn.endpoint_b();
            asic_keys.emplace(a.host_id(), a.tray_id(), a.asic_location());
            asic_keys.emplace(b.host_id(), b.tray_id(), b.asic_location());
        }
    }

    // Synthesize a deterministic uint64 unique_id for every ASIC. The std::set above gives a stable iteration
    // order keyed by (host_id, tray_id, asic_location), so the mapping is reproducible across runs for a given FSD.
    std::map<AsicKey, uint64_t> key_to_unique_id;
    {
        uint64_t next_id = 1;  // 0 is reserved to avoid collision with UNKNOWN defaults
        for (const auto& k : asic_keys) {
            key_to_unique_id[k] = next_id++;
        }
    }

    auto unique_id_for = [&](uint32_t host_id, uint32_t tray_id, uint32_t asic_location) {
        auto it = key_to_unique_id.find(AsicKey{host_id, tray_id, asic_location});
        if (it == key_to_unique_id.end()) {
            throw std::runtime_error(
                fmt::format("FSD endpoint ({}, {}, {}) not in synthesized ASIC set", host_id, tray_id, asic_location));
        }
        return it->second;
    };

    auto board_type_for = [&](uint32_t host_id, uint32_t tray_id) {
        auto it = host_tray_to_board_type.find({host_id, tray_id});
        if (it == host_tray_to_board_type.end()) {
            throw std::runtime_error(
                fmt::format("FSD has no board_type entry for host_id={}, tray_id={}", host_id, tray_id));
        }
        return it->second;
    };

    // host_to_mobo_name and host_to_rank (rank = FSD host index). Reject duplicate hostnames: the PSD keys
    // ranks / motherboards / graph entries by hostname, so duplicates would silently collapse hosts together.
    std::set<std::string> seen_hostnames;
    for (int host_id = 0; host_id < num_hosts; ++host_id) {
        const auto& host = fsd.hosts(host_id);
        if (!seen_hostnames.insert(host.hostname()).second) {
            throw std::runtime_error(
                fmt::format("Duplicate hostname '{}' in Factory System Descriptor", host.hostname()));
        }
        auto* mobo = psd.add_host_to_mobo_name();
        mobo->set_host_name(host.hostname());
        mobo->set_mobo_name(host.motherboard());

        auto* rank = psd.add_host_to_rank();
        rank->set_host_name(host.hostname());
        rank->set_rank(static_cast<uint32_t>(host_id));
    }

    // ASIC descriptors.
    for (const auto& [key, unique_id] : key_to_unique_id) {
        auto [host_id, tray_id, asic_location] = key;
        auto* asic_map = psd.add_asic_descriptors();
        asic_map->set_asic_id(unique_id);

        auto* desc = asic_map->mutable_asic_descriptor();
        desc->set_tray_id(tray_id);
        desc->set_asic_location(asic_location);
        desc->set_board_type(board_type_for(host_id, tray_id));
        desc->set_unique_id(unique_id);
        desc->set_host_name(hostname_of(host_id));
    }

    if (!fsd.has_eth_connections()) {
        return psd;
    }

    // Group connections per (host, src_asic_id, dst_asic_id) so a single ASIC connection edge can carry all
    // ethernet channels between that pair. Edges are emitted in both directions so the resulting graph is
    // symmetric, matching the structure produced by the runtime (HPT) overload in Fabric Manager.
    struct EthChan {
        uint32_t src_chan;
        uint32_t dst_chan;
        bool is_local;
    };
    // src_host_name -> src_asic_id -> dst_asic_id -> channels
    std::map<std::string, std::map<uint64_t, std::map<uint64_t, std::vector<EthChan>>>> outbound;
    // src_host_name -> dst_host_name -> list of exit node connections (src_asic, dst_asic, src_chan, dst_chan)
    std::map<std::string, std::map<std::string, std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>>>>
        host_outbound;
    // src_host_name -> flat list of exit node connections (for exit_node_connection_table)
    std::map<std::string, std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>>> exit_table;

    for (const auto& conn : fsd.eth_connections().connection()) {
        const auto& a = conn.endpoint_a();
        const auto& b = conn.endpoint_b();
        const bool is_local = (a.host_id() == b.host_id());

        const uint64_t asic_a = unique_id_for(a.host_id(), a.tray_id(), a.asic_location());
        const uint64_t asic_b = unique_id_for(b.host_id(), b.tray_id(), b.asic_location());

        const std::string& host_a = hostname_of(a.host_id());
        const std::string& host_b = hostname_of(b.host_id());

        outbound[host_a][asic_a][asic_b].push_back({a.chan_id(), b.chan_id(), is_local});
        outbound[host_b][asic_b][asic_a].push_back({b.chan_id(), a.chan_id(), is_local});

        if (!is_local) {
            host_outbound[host_a][host_b].emplace_back(asic_a, asic_b, a.chan_id(), b.chan_id());
            host_outbound[host_b][host_a].emplace_back(asic_b, asic_a, b.chan_id(), a.chan_id());
            exit_table[host_a].emplace_back(asic_a, asic_b, a.chan_id(), b.chan_id());
            exit_table[host_b].emplace_back(asic_b, asic_a, b.chan_id(), a.chan_id());
        }
    }

    // Ensure every host that owns at least one ASIC appears in the ASIC connectivity graph, even if its ASICs
    // have no outbound connections.
    for (const auto& [key, _] : key_to_unique_id) {
        outbound[hostname_of(std::get<0>(key))];
    }

    auto* graph = psd.mutable_system_graph();

    // ASIC connectivity graph.
    for (const auto& [hostname, asic_map] : outbound) {
        auto* host_conn = graph->add_asic_connectivity_graph();
        host_conn->set_host_name(hostname);
        for (const auto& [src_asic, dst_map] : asic_map) {
            auto* asic_graph = host_conn->add_asic_topologies();
            asic_graph->set_asic_id(src_asic);
            auto* topo = asic_graph->mutable_topology();
            for (const auto& [dst_asic, channels] : dst_map) {
                auto* edge = topo->add_asic_connections();
                edge->set_dst_asic_id(dst_asic);
                for (const auto& ch : channels) {
                    auto* eth = edge->add_eth_connections();
                    eth->set_src_chan(ch.src_chan);
                    eth->set_dst_chan(ch.dst_chan);
                    eth->set_is_local(ch.is_local);
                }
            }
        }
    }

    // Host connectivity graph — emit one entry per host, including hosts with no inter-host neighbors (isolated
    // or single-host descriptors). This matches the runtime discovery path (physical_system_discovery.cpp), so
    // PhysicalSystemDescriptor::get_host_neighbors() returns an empty list rather than failing its contains() check.
    for (int host_id = 0; host_id < num_hosts; ++host_id) {
        const std::string& src_host = fsd.hosts(host_id).hostname();
        auto* host_conn = graph->add_host_connectivity_graph();
        host_conn->set_src_host_name(src_host);
        auto it = host_outbound.find(src_host);
        if (it == host_outbound.end()) {
            continue;  // no inter-host links; keep the empty entry
        }
        for (const auto& [dst_host, conns] : it->second) {
            auto* edge = host_conn->add_host_connections();
            edge->set_dst_host_name(dst_host);
            for (const auto& [src_exit, dst_exit, src_ch, dst_ch] : conns) {
                auto* exit_conn = edge->add_exit_node_connections();
                exit_conn->set_src_exit_node(src_exit);
                exit_conn->set_dst_exit_node(dst_exit);
                auto* eth = exit_conn->mutable_eth_conn();
                eth->set_src_chan(src_ch);
                eth->set_dst_chan(dst_ch);
                eth->set_is_local(false);
            }
        }
    }

    // Exit node connection table.
    for (const auto& [hostname, conns] : exit_table) {
        auto* table = psd.add_exit_node_connection_table();
        table->set_host_name(hostname);
        for (const auto& [src_exit, dst_exit, src_ch, dst_ch] : conns) {
            auto* exit_conn = table->add_exit_connections();
            exit_conn->set_src_exit_node(src_exit);
            exit_conn->set_dst_exit_node(dst_exit);
            auto* eth = exit_conn->mutable_eth_conn();
            eth->set_src_chan(src_ch);
            eth->set_dst_chan(dst_ch);
            eth->set_is_local(false);
        }
    }

    return psd;
}

std::vector<::tt::fabric::proto::PhysicalSystemDescriptor> build_physical_descriptors(
    const ::tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) {
    std::vector<::tt::fabric::proto::PhysicalSystemDescriptor> psds;

    auto groups = partition_fsd_hosts_by_connectivity(fsd);
    log_info(tt::LogFabric, "Partitioned {} FSD host(s) into {} connected group(s)", fsd.hosts_size(), groups.size());

    psds.reserve(groups.size());
    for (const auto& members : groups) {
        psds.push_back(build_physical_descriptor(build_sub_fsd(fsd, members)));
    }
    return psds;
}

}  // namespace tt::tt_metal::experimental::tt_fabric
