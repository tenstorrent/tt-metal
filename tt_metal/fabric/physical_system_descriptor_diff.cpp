// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_node_id.hpp>

namespace tt::tt_metal {

namespace {

// One end of a cable: the chip's address plus the channel it leaves through. This, and not an
// AsicID, is what two independently labelled descriptors can agree on.
struct CableEnd {
    PhysicalNodeId node{};
    uint8_t chan = 0;

    friend bool operator==(const CableEnd&, const CableEnd&) = default;
    friend auto operator<=>(const CableEnd&, const CableEnd&) = default;
};

// A cable, identified without a direction: the ends are stored in sorted order so the record host A
// keeps for a cross-host link and the mirrored record host B keeps collapse onto one key.
struct CableKey {
    CableEnd lo;
    CableEnd hi;

    CableKey(const CableEnd& a, const CableEnd& b) : lo(std::min(a, b)), hi(std::max(a, b)) {}

    friend bool operator==(const CableKey&, const CableKey&) = default;
};

struct CableKeyHash {
    std::size_t operator()(const CableKey& key) const noexcept {
        auto mix = [](std::size_t seed, std::size_t value) {
            return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };
        std::size_t seed = std::hash<PhysicalNodeId>{}(key.lo.node);
        seed = mix(seed, std::hash<uint8_t>{}(key.lo.chan));
        seed = mix(seed, std::hash<PhysicalNodeId>{}(key.hi.node));
        seed = mix(seed, std::hash<uint8_t>{}(key.hi.chan));
        return seed;
    }
};

// What we remember about a cable: the attributes worth comparing, plus one directed observation of
// it so a delta record can be rebuilt with this descriptor's own labels. One observation is enough
// for both directions, since the reverse is just the two ends swapped.
struct CableFacts {
    AsicID src_asic{0};
    AsicID dst_asic{0};
    uint8_t src_chan = 0;
    uint8_t dst_chan = 0;
    bool is_local = false;
    PortType port_type = PortType::UNKNOWN;
};

using CableMap = std::unordered_map<CableKey, CableFacts, CableKeyHash>;

CableMap collect_cables(const PhysicalSystemDescriptor& descriptor, const PhysicalNodeIdIndex& index) {
    CableMap cables;
    for (const auto& [host_name, asic_topology] : descriptor.get_system_graph().asic_connectivity_graph) {
        for (const auto& [src_asic, edges] : asic_topology) {
            const auto src_node = index.asic_id_to_node_id.find(src_asic);
            if (src_node == index.asic_id_to_node_id.end()) {
                // An edge naming an ASIC the descriptor never described. Nothing positional to
                // compare it against, so leave it out rather than invent an address for it.
                continue;
            }
            for (const auto& [dst_asic, eth_connections] : edges) {
                const auto dst_node = index.asic_id_to_node_id.find(dst_asic);
                if (dst_node == index.asic_id_to_node_id.end()) {
                    continue;
                }
                for (const auto& eth_connection : eth_connections) {
                    const CableEnd src_end{src_node->second, eth_connection.src_chan};
                    const CableEnd dst_end{dst_node->second, eth_connection.dst_chan};
                    // First writer wins: the mirrored record carries the same attributes, and
                    // whichever direction we keep can reconstruct both.
                    cables.emplace(
                        CableKey{src_end, dst_end},
                        CableFacts{
                            src_asic,
                            dst_asic,
                            eth_connection.src_chan,
                            eth_connection.dst_chan,
                            eth_connection.is_local,
                            eth_connection.port_type});
                }
            }
        }
    }
    return cables;
}

// Append one directed record, merging into the existing entry for this destination. AsicTopology
// allows a destination to appear in several entries for the same source, and a caller comparing
// entry counts would read that as extra cables.
void add_directed_link(AsicTopology& topology, AsicID src, AsicID dst, const EthConnection& connection) {
    auto& edges = topology[src];
    const auto edge = std::find_if(
        edges.begin(), edges.end(), [dst](const AsicConnectionEdge& candidate) { return candidate.first == dst; });
    if (edge == edges.end()) {
        edges.emplace_back(dst, std::vector<EthConnection>{connection});
    } else {
        edge->second.push_back(connection);
    }
}

// Both halves of the cable, so a caller that keys on the source end sees it from either side.
void add_cable(AsicTopology& topology, const CableFacts& facts) {
    add_directed_link(
        topology,
        facts.src_asic,
        facts.dst_asic,
        EthConnection{facts.src_chan, facts.dst_chan, facts.is_local, facts.port_type});
    add_directed_link(
        topology,
        facts.dst_asic,
        facts.src_asic,
        EthConnection{facts.dst_chan, facts.src_chan, facts.is_local, facts.port_type});
}

// Deterministic output: the maps are unordered, but everything inside them is ordered so a caller
// can compare, log, or hash a delta without seeing container-order noise.
void sort_links(AsicTopology& topology) {
    for (auto& [src_asic, edges] : topology) {
        for (auto& [dst_asic, connections] : edges) {
            std::sort(connections.begin(), connections.end());
        }
        std::sort(edges.begin(), edges.end(), [](const AsicConnectionEdge& a, const AsicConnectionEdge& b) {
            return a.first < b.first;
        });
    }
}

}  // namespace

PhysicalSystemDelta diff_physical_system_descriptors(
    const PhysicalSystemDescriptor& golden, const PhysicalSystemDescriptor& candidate) {
    // Fatal on a duplicate address, which is what makes the positional join well defined.
    const auto golden_index = build_physical_node_id_index(golden);
    const auto candidate_index = build_physical_node_id_index(candidate);

    PhysicalSystemDelta delta;

    for (const auto& [node_id, golden_asic] : golden_index.node_id_to_asic_id) {
        const auto candidate_asic = candidate_index.node_id_to_asic_id.find(node_id);
        if (candidate_asic == candidate_index.node_id_to_asic_id.end()) {
            delta.missing_asics.push_back(golden_asic);
            continue;
        }
        // Same chip, described differently. Tray and location cannot differ here -- they are part
        // of the address, so a chip that moved reads as one address missing and another extra.
        if (golden.get_asic_descriptors().at(golden_asic).board_type !=
            candidate.get_asic_descriptors().at(candidate_asic->second).board_type) {
            delta.mismatched_asics.push_back(golden_asic);
        }
    }
    for (const auto& [node_id, candidate_asic] : candidate_index.node_id_to_asic_id) {
        if (!golden_index.node_id_to_asic_id.contains(node_id)) {
            delta.extra_asics.push_back(candidate_asic);
        }
    }

    const auto golden_cables = collect_cables(golden, golden_index);
    const auto candidate_cables = collect_cables(candidate, candidate_index);

    for (const auto& [key, golden_facts] : golden_cables) {
        const auto candidate_cable = candidate_cables.find(key);
        if (candidate_cable == candidate_cables.end()) {
            add_cable(delta.missing_links, golden_facts);
            continue;
        }
        const auto& candidate_facts = candidate_cable->second;
        if (golden_facts.port_type != candidate_facts.port_type || golden_facts.is_local != candidate_facts.is_local) {
            add_cable(delta.mismatched_links, golden_facts);
        }
    }
    for (const auto& [key, candidate_facts] : candidate_cables) {
        if (!golden_cables.contains(key)) {
            add_cable(delta.extra_links, candidate_facts);
        }
    }

    std::sort(delta.missing_asics.begin(), delta.missing_asics.end());
    std::sort(delta.extra_asics.begin(), delta.extra_asics.end());
    std::sort(delta.mismatched_asics.begin(), delta.mismatched_asics.end());
    sort_links(delta.missing_links);
    sort_links(delta.extra_links);
    sort_links(delta.mismatched_links);

    return delta;
}

}  // namespace tt::tt_metal
