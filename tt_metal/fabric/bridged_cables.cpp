// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/bridged_cables.hpp"

#include <algorithm>
#include <charconv>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/reflection.hpp>

namespace tt::tt_fabric::experimental {

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::EthConnection;
using tt::tt_metal::ExitNodeConnection;
using tt::tt_metal::PhysicalSystemDescriptor;
using tt::tt_metal::experimental::PhysicalNodeId;

constexpr std::string_view kGrammar =
    "TT_METAL_E2H_BRIDGE_BREAK_LINKS takes one of: list | all | cable:<i>[,<j>...] | random:<n>[:<seed>]";

// An end as the order sees it: where the chip is, not what discovery happened to call it.
struct EndKey {
    PhysicalNodeId node;
    std::uint8_t chan = 0;
    friend auto operator<=>(const EndKey&, const EndKey&) = default;
};

// Cables keyed by their (lower, higher) end, which both dedupes the two directed halves and fixes the order.
using CableMap = std::map<std::pair<EndKey, EndKey>, BridgedCable>;

// Adds the cable src <-> dst over `c`, named by `names`; skipped when either chip is not in `names`.
void add_cable(
    CableMap& cables,
    const PhysicalSystemDescriptor& names,
    AsicID src,
    AsicID dst,
    const EthConnection& c,
    BridgeReason reason) {
    const auto src_node = names.find_physical_node_id(src);
    const auto dst_node = names.find_physical_node_id(dst);
    if (!src_node || !dst_node) {
        return;
    }
    const bool cross_rank = names.get_mpi_rank_for_asic(src) != names.get_mpi_rank_for_asic(dst);
    EndKey src_key{*src_node, c.src_chan}, dst_key{*dst_node, c.dst_chan};
    CableEnd src_end{src, c.src_chan}, dst_end{dst, c.dst_chan};
    if (dst_key < src_key) {
        std::swap(src_key, dst_key);
        std::swap(src_end, dst_end);
    }
    cables.try_emplace(std::pair{src_key, dst_key}, BridgedCable{src_end, dst_end, cross_rank, c.port_type, reason});
}

std::vector<BridgedCable> values_of(const CableMap& cables) {
    const auto values = cables | std::views::values;
    return {values.begin(), values.end()};
}

std::uint64_t parse_u64(std::string_view text, std::string_view what) {
    std::uint64_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || ec != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(fmt::format("{}: '{}' is not a number for {}", kGrammar, text, what));
    }
    return value;
}

std::vector<std::string_view> split(std::string_view text, char sep) {
    std::vector<std::string_view> parts;
    while (true) {
        const auto at = text.find(sep);
        parts.push_back(text.substr(0, at));
        if (at == std::string_view::npos) {
            return parts;
        }
        text.remove_prefix(at + 1);
    }
}

// One directed half of a Downed cable, recorded the way run_local_discovery records a discovered one.
void add_directed_half(
    PhysicalSystemDescriptor& live, const BridgedCable& cable, const CableEnd& src, const CableEnd& dst) {
    const auto& asics = live.get_asic_descriptors();
    if (!asics.contains(src.asic) || !asics.contains(dst.asic)) {
        return;  // a chip missing from this descriptor (e.g. a refresh): nothing to record
    }
    const auto& src_desc = asics.at(src.asic);
    const auto& dst_desc = asics.at(dst.asic);
    const EthConnection connection{src.chan, dst.chan, /*is_local=*/!cable.cross_rank, cable.port_type};

    auto& edges = live.get_system_graph().asic_connectivity_graph[src_desc.host_name][src.asic];
    auto edge = std::find_if(edges.begin(), edges.end(), [&](const auto& e) { return e.first == dst.asic; });
    if (edge == edges.end()) {
        edges.emplace_back(dst.asic, std::vector<EthConnection>{});
        edge = std::prev(edges.end());
    }
    auto& connections = edge->second;
    if (std::any_of(
            connections.begin(), connections.end(), [&](const EthConnection& c) { return c.src_chan == src.chan; })) {
        return;  // already there: idempotent
    }
    connections.push_back(connection);
    std::sort(connections.begin(), connections.end());

    if (!cable.cross_rank) {
        return;
    }
    const ExitNodeConnection exit{.src_exit_node = src.asic, .dst_exit_node = dst.asic, .eth_conn = connection};
    auto& exits = live.get_exit_node_connection_table()[src_desc.host_name];
    if (std::find(exits.begin(), exits.end(), exit) == exits.end()) {
        exits.push_back(exit);
    }

    // The host graph groups exits by peer rank, and ranks may share a host name, so match on both.
    auto& host_edges = live.get_system_graph().host_connectivity_graph[src_desc.host_name];
    auto host_edge = std::find_if(host_edges.begin(), host_edges.end(), [&](const auto& e) {
        return e.first == dst_desc.host_name &&
               (e.second.empty() || live.get_mpi_rank_for_asic(e.second.front().dst_exit_node) == dst_desc.mpi_rank);
    });
    if (host_edge == host_edges.end()) {
        host_edges.emplace_back(dst_desc.host_name, std::vector<ExitNodeConnection>{});
        host_edge = std::prev(host_edges.end());
    }
    if (std::find(host_edge->second.begin(), host_edge->second.end(), exit) == host_edge->second.end()) {
        host_edge->second.push_back(exit);
    }
}

}  // namespace

BreakSpec parse_break_spec(std::string_view spec) {
    BreakSpec out;
    if (spec.empty() || spec == "list" || spec == "all") {
        out.mode = spec.empty() ? BreakSpec::Mode::None : spec == "list" ? BreakSpec::Mode::List : BreakSpec::Mode::All;
        return out;
    }
    const auto colon = spec.find(':');
    if (colon == std::string_view::npos) {
        throw std::invalid_argument(fmt::format("{}; got '{}'", kGrammar, spec));
    }
    const auto head = spec.substr(0, colon);
    const auto args = split(spec.substr(colon + 1), ':');

    if (head == "cable" && args.size() == 1) {
        out.mode = BreakSpec::Mode::Cables;
        for (const auto index : split(args[0], ',')) {
            out.cable_indices.push_back(static_cast<std::uint32_t>(parse_u64(index, "a cable index")));
        }
        return out;
    }
    if (head == "random" && (args.size() == 1 || args.size() == 2)) {
        out.mode = BreakSpec::Mode::Random;
        out.count = static_cast<std::uint32_t>(parse_u64(args[0], "a cable count"));
        out.seed = args.size() == 2 ? parse_u64(args[1], "a seed") : 0;
        return out;
    }
    throw std::invalid_argument(fmt::format("{}; got '{}'", kGrammar, spec));
}

std::vector<BridgedCable> enumerate_live_cables(const PhysicalSystemDescriptor& live) {
    CableMap cables;
    for (const auto& topology : live.get_system_graph().asic_connectivity_graph | std::views::values) {
        for (const auto& [src, edges] : topology) {
            for (const auto& [dst, connections] : edges) {
                for (const auto& connection : connections) {
                    add_cable(cables, live, src, dst, connection, BridgeReason::Injected);
                }
            }
        }
    }
    return values_of(cables);
}

std::vector<BridgedCable> select_injected_cables(const std::vector<BridgedCable>& live_cables, const BreakSpec& spec) {
    std::vector<std::uint32_t> picked;
    switch (spec.mode) {
        case BreakSpec::Mode::None:
        case BreakSpec::Mode::List: return {};
        case BreakSpec::Mode::All:
            picked.resize(live_cables.size());
            std::iota(picked.begin(), picked.end(), 0u);
            break;
        case BreakSpec::Mode::Cables:
            for (const auto index : spec.cable_indices) {
                if (index >= live_cables.size()) {
                    throw std::invalid_argument(
                        fmt::format("no cable {} of {}; see 'list'", index, live_cables.size()));
                }
                picked.push_back(index);
            }
            break;
        case BreakSpec::Mode::Random: {
            if (spec.count > live_cables.size()) {
                throw std::invalid_argument(
                    fmt::format("random:{} but only {} cables", spec.count, live_cables.size()));
            }
            // Partial Fisher-Yates driven by ttsl's hash: unlike std distributions, every rank makes the same picks.
            std::vector<std::uint32_t> order(live_cables.size());
            std::iota(order.begin(), order.end(), 0u);
            for (std::uint32_t i = 0; i < spec.count; ++i) {
                const auto j =
                    i + static_cast<std::uint32_t>(ttsl::hash::hash_objects(spec.seed, i) % (order.size() - i));
                std::swap(order[i], order[j]);
            }
            picked.assign(order.begin(), order.begin() + spec.count);
            break;
        }
    }

    std::sort(picked.begin(), picked.end());
    picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
    std::vector<BridgedCable> out;  // enumerate_live_cables already marks them Injected
    for (const auto index : picked) {
        out.push_back(live_cables[index]);
    }
    return out;
}

std::vector<BridgedCable> find_downed_cables(
    const PhysicalSystemDescriptor& fsd, const PhysicalSystemDescriptor& live) {
    const auto delta = tt::tt_metal::experimental::diff_physical_system_descriptors(fsd, live);
    CableMap cables;
    // Delta labels are the golden side's: join to live by address. An absent chip is not a cable fault.
    auto to_live = [&](AsicID asic) {
        const auto address = fsd.find_physical_node_id(asic);
        return address ? live.find_asic_id(*address) : std::nullopt;
    };
    for (const auto& [src, edges] : delta.missing_links) {
        for (const auto& [dst, connections] : edges) {
            const auto live_src = to_live(src);
            const auto live_dst = to_live(dst);
            if (!live_src || !live_dst) {
                continue;
            }
            for (const auto& connection : connections) {
                add_cable(cables, live, *live_src, *live_dst, connection, BridgeReason::Downed);
            }
        }
    }
    return values_of(cables);
}

std::vector<BridgedCable> merge_bridged_cables(
    std::vector<BridgedCable> downed, const std::vector<BridgedCable>& injected) {
    for (const auto& cable : injected) {
        // Both lists name a cable the same way round (lower end first).
        if (std::none_of(
                downed.begin(), downed.end(), [&](const auto& d) { return d.a == cable.a && d.b == cable.b; })) {
            downed.push_back(cable);
        }
    }
    return downed;
}

void add_bridged_cables_to_live(PhysicalSystemDescriptor& live, const std::vector<BridgedCable>& cables) {
    for (const auto& cable : cables) {
        if (cable.reason != BridgeReason::Downed) {
            continue;
        }
        add_directed_half(live, cable, cable.a, cable.b);
        add_directed_half(live, cable, cable.b, cable.a);
    }
}

void remove_cables_from_live(PhysicalSystemDescriptor& live, const std::vector<BridgedCable>& cables) {
    auto drop = [&live](const CableEnd& src, const CableEnd& dst) {
        if (!live.get_asic_descriptors().contains(src.asic)) {
            return;
        }
        const auto host = live.get_asic_descriptors().at(src.asic).host_name;
        auto same = [&](const EthConnection& c) { return c.src_chan == src.chan && c.dst_chan == dst.chan; };
        auto& edges = live.get_system_graph().asic_connectivity_graph[host][src.asic];
        for (auto& edge : edges) {
            if (edge.first == dst.asic) {
                std::erase_if(edge.second, same);
            }
        }
        std::erase_if(edges, [](const auto& e) { return e.second.empty(); });
        auto is_exit = [&](const ExitNodeConnection& e) {
            return e.src_exit_node == src.asic && e.dst_exit_node == dst.asic && same(e.eth_conn);
        };
        std::erase_if(live.get_exit_node_connection_table()[host], is_exit);
        for (auto& host_edge : live.get_system_graph().host_connectivity_graph[host]) {
            std::erase_if(host_edge.second, is_exit);
        }
    };
    for (const auto& cable : cables) {
        drop(cable.a, cable.b);
        drop(cable.b, cable.a);
    }
}

std::string describe_cable(const PhysicalSystemDescriptor& live, const BridgedCable& cable) {
    auto where = [&live](const CableEnd& end) {
        const auto node = live.find_physical_node_id(end.asic);
        return node.has_value() ? fmt::format("{} chan {}", *node, end.chan)
                                : fmt::format("<unknown> chan {}", end.chan);
    };
    return fmt::format(
        "{} <-> {} ({}, {}, {})",
        where(cable.a),
        where(cable.b),
        enchantum::to_string(cable.port_type),
        cable.cross_rank ? "cross-rank" : "same-rank",
        cable.reason == BridgeReason::Downed ? "downed" : "injected");
}

void BridgedCables::record_layout(FabricNodeId node, chan_id_t chan, BridgedRouterLayout layout) {
    std::lock_guard<std::mutex> lock(layouts_mutex_);
    layouts_[{node, chan}] = std::move(layout);
}

std::optional<BridgedRouterLayout> BridgedCables::layout(FabricNodeId node, chan_id_t chan) const {
    std::lock_guard<std::mutex> lock(layouts_mutex_);
    const auto it = layouts_.find({node, chan});
    return it == layouts_.end() ? std::nullopt : std::optional{it->second};
}

}  // namespace tt::tt_fabric::experimental
