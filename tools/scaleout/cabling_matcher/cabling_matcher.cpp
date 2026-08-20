// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cabling_matcher.hpp"

#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <google/protobuf/text_format.h>

#include <connector/connector.hpp>

#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools::matcher {

namespace {

constexpr uint32_t kUnmapped = Diagnosis::kUnmapped;

// Guard against the search space blowing up in the port-agnostic modes, where a cable can be placed
// many ways. Running out cuts the search short and says so rather than producing a wrong answer. The
// budget is shared between the candidate seed hosts, with a floor so that a cluster with very many
// hosts still gives each of them a workable share.
constexpr uint64_t kMaxSearchSteps = 8'000'000;
constexpr uint64_t kMinSeedSteps = 250'000;

// No level of the search is being abandoned.
constexpr int kNoPrune = std::numeric_limits<int>::max();

template <typename Descriptor>
Descriptor load_textproto(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    Descriptor descriptor;
    if (!google::protobuf::TextFormat::ParseFromString(contents, &descriptor)) {
        throw std::runtime_error("Failed to parse textproto file: " + path);
    }
    return descriptor;
}

cabling_generator::proto::ClusterDescriptor load_cluster_descriptor(const std::string& path) {
    return load_textproto<cabling_generator::proto::ClusterDescriptor>(path);
}

fsd::proto::FactorySystemDescriptor load_factory_system_descriptor(const std::string& path) {
    return load_textproto<fsd::proto::FactorySystemDescriptor>(path);
}

// Boards are immutable here and building one is not free, so each type is built once.
const Board& board_for(BoardType board_type) {
    static std::map<BoardType, Board> cache;
    auto it = cache.find(board_type);
    if (it == cache.end()) {
        it = cache.emplace(board_type, create_board(board_type)).first;
    }
    return it->second;
}

// Channels a cable between these two ports carries, which is a property of the boards rather than of
// any descriptor.
size_t expected_channel_count(
    const std::vector<MatchGraph::HostInfo>& hosts, const CableEndpoint& endpoint_a, const CableEndpoint& endpoint_b) {
    const Board& board_a = board_for(hosts[*endpoint_a.host_id].trays.at(endpoint_a.tray_id));
    const Board& board_b = board_for(hosts[*endpoint_b.host_id].trays.at(endpoint_b.tray_id));
    return get_asic_channel_connections(
               endpoint_a.port_type,
               board_a.get_port_channels(endpoint_a.port_type, endpoint_a.port_id),
               board_b.get_port_channels(endpoint_b.port_type, endpoint_b.port_id))
        .size();
}

std::string port_to_string(const CableEndpoint& endpoint) {
    return fmt::format(
        "host {} tray {} {} port {}",
        *endpoint.host_id,
        *endpoint.tray_id,
        enchantum::to_string(endpoint.port_type),
        *endpoint.port_id);
}

std::string cable_to_string(const ResolvedCable& cable) {
    return fmt::format("{} <-> {}", port_to_string(cable.endpoint_a), port_to_string(cable.endpoint_b));
}

void count_hosts_in_instance(const cabling_generator::proto::GraphInstance& instance, size_t& count) {
    for (const auto& [name, mapping] : instance.child_mappings()) {
        if (mapping.has_sub_instance()) {
            count_hosts_in_instance(mapping.sub_instance(), count);
        } else {
            ++count;
        }
    }
}

void build_instance(
    const cabling_generator::proto::ClusterDescriptor& source,
    const std::string& template_name,
    cabling_generator::proto::GraphInstance* out,
    uint32_t& next_host_id,
    std::vector<std::string>& stack) {
    if (std::find(stack.begin(), stack.end(), template_name) != stack.end()) {
        throw std::runtime_error(
            fmt::format("Graph template '{}' contains itself (via {})", template_name, fmt::join(stack, " -> ")));
    }
    auto it = source.graph_templates().find(template_name);
    if (it == source.graph_templates().end()) {
        std::vector<std::string> available;
        for (const auto& [name, unused] : source.graph_templates()) {
            available.push_back(name);
        }
        std::sort(available.begin(), available.end());
        throw std::runtime_error(fmt::format(
            "Graph template '{}' not found in descriptor. Available templates: {}",
            template_name,
            available.empty() ? "(none)" : fmt::format("{}", fmt::join(available, ", "))));
    }
    stack.push_back(template_name);
    out->set_template_name(template_name);
    for (const auto& child : it->second.children()) {
        auto& mapping = (*out->mutable_child_mappings())[child.name()];
        if (child.has_node_ref()) {
            mapping.set_host_id(next_host_id++);
        } else if (child.has_graph_ref()) {
            build_instance(
                source, child.graph_ref().graph_template(), mapping.mutable_sub_instance(), next_host_id, stack);
        } else {
            throw std::runtime_error(fmt::format(
                "Child '{}' of graph template '{}' is neither a node_ref nor a graph_ref",
                child.name(),
                template_name));
        }
    }
    stack.pop_back();
}

bool hosts_compatible(
    const MatchGraph::HostInfo& pattern_host, const MatchGraph::HostInfo& target_host, TraySymmetry symmetry) {
    if (pattern_host.trays.size() != target_host.trays.size()) {
        return false;
    }
    if (symmetry == TraySymmetry::None) {
        return pattern_host.trays == target_host.trays;
    }
    std::vector<BoardType> pattern_boards;
    std::vector<BoardType> target_boards;
    for (const auto& [tray, board] : pattern_host.trays) {
        pattern_boards.push_back(board);
    }
    for (const auto& [tray, board] : target_host.trays) {
        target_boards.push_back(board);
    }
    std::sort(pattern_boards.begin(), pattern_boards.end());
    std::sort(target_boards.begin(), target_boards.end());
    return pattern_boards == target_boards;
}

// Searches for placements of one connected component of the pattern.
//
// Cables are visited in BFS order from a seed host, so every cable has at least one end on a host
// that is already mapped by the time it is placed. That end anchors the cable: it fixes which target
// host and tray the cable must leave from, which narrows the target cables that could serve it to
// the handful on that tray -- exactly one under strict port identity, which is why the strict search
// is a linear propagation rather than a search. Placing a cable then forces the host and tray at its
// far end, so the mapping grows outward from the seed with no guessing beyond the candidate cable
// itself.
class Search {
public:
    Search(
        const MatchGraph& pattern,
        const MatchGraph& target,
        const MatchOptions& options,
        std::vector<uint32_t> component_hosts) :
        pattern_(pattern), target_(target), options_(options), component_hosts_(std::move(component_hosts)) {}

    ComponentResult run() {
        ComponentResult result;
        result.pattern_hosts = component_hosts_;
        if (component_hosts_.empty()) {
            return result;
        }

        pick_seed();
        build_order();

        host_map_.assign(pattern_.hosts().size(), kUnmapped);
        host_rmap_.assign(target_.hosts().size(), kUnmapped);
        tray_map_.assign(pattern_.hosts().size(), {});
        tray_rmap_.assign(pattern_.hosts().size(), {});
        cable_map_.assign(pattern_.cables().size(), 0);
        target_cable_used_.assign(target_.cables().size(), false);

        std::set<std::pair<uint32_t, TrayId>> slots;
        for (uint32_t pattern_host : component_hosts_) {
            for (size_t cable_idx : pattern_.cables_at(pattern_host)) {
                const auto& cable = pattern_.cables()[cable_idx];
                slots.insert({*cable.endpoint_a.host_id, cable.endpoint_a.tray_id});
                slots.insert({*cable.endpoint_b.host_id, cable.endpoint_b.tray_id});
            }
        }
        needed_slots_ = slots.size();

        // Every target host the seed could stand for is tried in turn, and each gets its own share of
        // the step budget. A single share is plenty to find the placements around one seed host, and
        // sharing keeps a seed that has fallen into an expensive corner of the search from spending
        // the whole budget and leaving the rest of the cluster unexamined.
        size_t candidate_seeds = 0;
        for (const auto& target_host : target_.hosts()) {
            candidate_seeds += hosts_compatible(pattern_.hosts()[seed_], target_host, options_.tray_symmetry) ? 1 : 0;
        }
        budget_ = std::max(kMinSeedSteps, kMaxSearchSteps / std::max<size_t>(candidate_seeds, 1));

        for (uint32_t target_host = 0; target_host < target_.hosts().size() && !stopped_at_limit_; ++target_host) {
            if (!hosts_compatible(pattern_.hosts()[seed_], target_.hosts()[target_host], options_.tray_symmetry)) {
                continue;
            }
            host_map_[seed_] = target_host;
            host_rmap_[target_host] = seed_;
            mapped_hosts_ = 1;
            steps_ = 0;
            stop_ = false;
            recurse(0);
            mapped_hosts_ = 0;
            prune_to_ = kNoPrune;
            host_rmap_[target_host] = kUnmapped;
            host_map_[seed_] = kUnmapped;
        }

        for (auto& [host_set, accumulated] : found_) {
            Match match = accumulated.canonical;
            match.role_assignments = accumulated.role_assignments.size();
            result.matches.push_back(std::move(match));
        }
        result.num_host_sets = found_.size();
        result.stopped_at_limit = stopped_at_limit_;
        result.exhausted_budget = exhausted_budget_;
        if (result.matches.empty()) {
            result.diagnosis = diagnosis_;
        }
        return result;
    }

private:
    struct Accumulated {
        Match canonical;
        std::set<std::vector<uint32_t>> role_assignments;
    };

    void pick_seed() {
        seed_ = component_hosts_.front();
        size_t best_degree = 0;
        for (uint32_t host : component_hosts_) {
            size_t degree = pattern_.cables_at(host).size();
            if (degree > best_degree) {
                best_degree = degree;
                seed_ = host;
            }
        }
    }

    // Cables that decide something -- reaching a new host, or using a tray of a host for the first
    // time -- go before cables that only need to be accounted for. Everything after the last
    // deciding cable is a pure assignment problem, which is much cheaper to solve than to search.
    void build_order() {
        std::vector<bool> host_seen(pattern_.hosts().size(), false);
        std::vector<bool> cable_seen(pattern_.cables().size(), false);
        std::set<std::pair<uint32_t, TrayId>> slot_seen;
        std::deque<uint32_t> queue{seed_};
        host_seen[seed_] = true;

        auto decides = [&](const ResolvedCable& cable) {
            for (const auto& endpoint : {cable.endpoint_a, cable.endpoint_b}) {
                if (!host_seen[*endpoint.host_id] || !slot_seen.contains({*endpoint.host_id, endpoint.tray_id})) {
                    return true;
                }
            }
            return false;
        };
        auto take = [&](size_t cable_idx) {
            cable_seen[cable_idx] = true;
            order_.push_back(cable_idx);
            const auto& cable = pattern_.cables()[cable_idx];
            for (const auto& endpoint : {cable.endpoint_a, cable.endpoint_b}) {
                slot_seen.insert({*endpoint.host_id, endpoint.tray_id});
                if (!host_seen[*endpoint.host_id]) {
                    host_seen[*endpoint.host_id] = true;
                    queue.push_back(*endpoint.host_id);
                }
            }
        };

        while (!queue.empty()) {
            uint32_t host = queue.front();
            queue.pop_front();
            for (int pass = 0; pass < 2; ++pass) {
                for (size_t cable_idx : pattern_.cables_at(host)) {
                    if (cable_seen[cable_idx]) {
                        continue;
                    }
                    bool wanted = decides(pattern_.cables()[cable_idx]);
                    if ((pass == 0) == wanted) {
                        take(cable_idx);
                    }
                }
            }
        }
    }

    // Every pattern host mapped and every tray it uses pinned to a target tray, so no cable left to
    // place can decide anything new.
    bool bindings_complete() const { return mapped_hosts_ == component_hosts_.size() && bound_slots_ == needed_slots_; }

    // Injectively assign the remaining pattern cables to distinct unused target cables. With hosts
    // and trays already pinned, each pattern cable's candidates are just the free target cables
    // between the two mapped trays whose ports correspond, so this is bipartite matching (Kuhn's
    // augmenting paths) rather than a search. On failure, first_unmatched names a pattern cable that
    // no assignment can serve.
    bool assign_remaining(size_t index, size_t& first_unmatched) {
        std::vector<std::vector<size_t>> candidates(order_.size() - index);
        for (size_t position = index; position < order_.size(); ++position) {
            const auto& cable = pattern_.cables()[order_[position]];
            const CableEndpoint& a = cable.endpoint_a;
            const CableEndpoint& b = cable.endpoint_b;
            uint32_t target_host_a = host_map_[*a.host_id];
            TrayId target_tray_a = tray_map_[*a.host_id].at(a.tray_id);
            uint32_t target_host_b = host_map_[*b.host_id];
            TrayId target_tray_b = tray_map_[*b.host_id].at(b.tray_id);
            for (size_t target_cable_idx : target_.cables_at(target_host_a, target_tray_a)) {
                if (target_cable_used_[target_cable_idx]) {
                    continue;
                }
                const auto& target_cable = target_.cables()[target_cable_idx];
                for (int side = 0; side < 2; ++side) {
                    const CableEndpoint& target_a = side == 0 ? target_cable.endpoint_a : target_cable.endpoint_b;
                    const CableEndpoint& target_b = side == 0 ? target_cable.endpoint_b : target_cable.endpoint_a;
                    if (*target_a.host_id != target_host_a || target_a.tray_id != target_tray_a) {
                        continue;
                    }
                    if (*target_b.host_id != target_host_b || target_b.tray_id != target_tray_b) {
                        continue;
                    }
                    if (ports_match(a, target_a) && ports_match(b, target_b)) {
                        candidates[position - index].push_back(target_cable_idx);
                        break;
                    }
                }
            }
            if (candidates[position - index].empty()) {
                first_unmatched = position;
                return false;
            }
        }

        std::map<size_t, size_t> target_to_pattern;
        std::set<size_t> visited;
        auto augment = [&](auto& self, size_t pattern_slot) -> bool {
            for (size_t target_cable_idx : candidates[pattern_slot]) {
                if (!visited.insert(target_cable_idx).second) {
                    continue;
                }
                auto it = target_to_pattern.find(target_cable_idx);
                if (it == target_to_pattern.end() || self(self, it->second)) {
                    target_to_pattern[target_cable_idx] = pattern_slot;
                    return true;
                }
            }
            return false;
        };
        for (size_t slot = 0; slot < candidates.size(); ++slot) {
            visited.clear();
            if (!augment(augment, slot)) {
                first_unmatched = index + slot;
                return false;
            }
        }
        for (const auto& [target_cable_idx, pattern_slot] : target_to_pattern) {
            cable_map_[order_[index + pattern_slot]] = target_cable_idx;
        }
        return true;
    }

    // Does a target port stand in for a pattern port? Both endpoints are known to be on hosts with
    // the same board layout, so ASIC locations are directly comparable.
    bool ports_match(const CableEndpoint& pattern_end, const CableEndpoint& target_end) const {
        if (pattern_end.port_type != target_end.port_type) {
            return false;
        }
        switch (options_.port_identity) {
            case PortIdentity::Strict: return pattern_end.port_id == target_end.port_id;
            case PortIdentity::Relaxed: return true;
            case PortIdentity::Chip: {
                if (pattern_end.port_id == target_end.port_id) {
                    return true;
                }
                const auto& pattern_asics = asics_for_port(
                    pattern_.hosts()[*pattern_end.host_id].trays.at(pattern_end.tray_id),
                    pattern_end.port_type,
                    pattern_end.port_id);
                const auto& target_asics = asics_for_port(
                    target_.hosts()[*target_end.host_id].trays.at(target_end.tray_id),
                    target_end.port_type,
                    target_end.port_id);
                return std::any_of(pattern_asics.begin(), pattern_asics.end(), [&](uint32_t asic) {
                    return target_asics.contains(asic);
                });
            }
        }
        return false;
    }

    // Target trays the pattern tray of an already-mapped host could be using.
    std::vector<TrayId> candidate_trays(uint32_t pattern_host, TrayId pattern_tray) const {
        const auto& bound = tray_map_[pattern_host];
        if (auto it = bound.find(pattern_tray); it != bound.end()) {
            return {it->second};
        }
        const auto& pattern_trays = pattern_.hosts()[pattern_host].trays;
        const auto& target_trays = target_.hosts()[host_map_[pattern_host]].trays;
        BoardType board = pattern_trays.at(pattern_tray);
        if (options_.tray_symmetry == TraySymmetry::None) {
            auto it = target_trays.find(pattern_tray);
            if (it == target_trays.end() || it->second != board) {
                return {};
            }
            return {pattern_tray};
        }
        std::vector<TrayId> candidates;
        for (const auto& [tray, target_board] : target_trays) {
            if (target_board == board && !tray_rmap_[pattern_host].contains(tray)) {
                candidates.push_back(tray);
            }
        }
        return candidates;
    }

    void bind_tray(uint32_t pattern_host, TrayId pattern_tray, TrayId target_tray) {
        tray_map_[pattern_host][pattern_tray] = target_tray;
        tray_rmap_[pattern_host][target_tray] = pattern_tray;
        ++bound_slots_;
    }

    void unbind_tray(uint32_t pattern_host, TrayId pattern_tray, TrayId target_tray) {
        tray_map_[pattern_host].erase(pattern_tray);
        tray_rmap_[pattern_host].erase(target_tray);
        --bound_slots_;
    }

    // True when this level and everything under it should be abandoned because a completed match has
    // already been recorded for every decision made at or below the level that is being retried.
    bool abandon(size_t index) {
        if (prune_to_ == kNoPrune) {
            return false;
        }
        if (static_cast<int>(index) > prune_to_) {
            return true;
        }
        prune_to_ = kNoPrune;
        return false;
    }

    void recurse(size_t index) {
        if (stop_) {
            return;
        }
        if (index == order_.size()) {
            record_match();
            return;
        }
        if (!have_diagnosis_ || index > deepest_) {
            have_diagnosis_ = true;
            deepest_ = index;
            diagnosis_ = diagnose(index);
        }
        if (bindings_complete()) {
            size_t unmatched = index;
            if (assign_remaining(index, unmatched)) {
                record_match();
            } else if (!have_diagnosis_ || unmatched > deepest_) {
                have_diagnosis_ = true;
                deepest_ = unmatched;
                diagnosis_ = diagnose(unmatched);
                diagnosis_->reason +=
                    " (no way of dealing out the remaining cables avoids this, having pinned every host and tray)";
            }
            return;
        }

        size_t pattern_cable_idx = order_[index];
        const auto& pattern_cable = pattern_.cables()[pattern_cable_idx];
        bool anchor_is_a = host_map_[*pattern_cable.endpoint_a.host_id] != kUnmapped;
        const CableEndpoint& anchor = anchor_is_a ? pattern_cable.endpoint_a : pattern_cable.endpoint_b;
        const CableEndpoint& far = anchor_is_a ? pattern_cable.endpoint_b : pattern_cable.endpoint_a;
        uint32_t anchor_pattern_host = *anchor.host_id;
        uint32_t anchor_target_host = host_map_[anchor_pattern_host];

        for (TrayId anchor_target_tray : candidate_trays(anchor_pattern_host, anchor.tray_id)) {
            bool anchor_tray_was_bound = tray_map_[anchor_pattern_host].contains(anchor.tray_id);
            if (!anchor_tray_was_bound) {
                bind_tray(anchor_pattern_host, anchor.tray_id, anchor_target_tray);
            }

            for (size_t target_cable_idx : target_.cables_at(anchor_target_host, anchor_target_tray)) {
                if (target_cable_used_[target_cable_idx]) {
                    continue;
                }
                if (++steps_ > budget_) {
                    exhausted_budget_ = true;
                    stop_ = true;
                    break;
                }
                const auto& target_cable = target_.cables()[target_cable_idx];
                for (int side = 0; side < 2; ++side) {
                    const CableEndpoint& target_anchor = side == 0 ? target_cable.endpoint_a : target_cable.endpoint_b;
                    const CableEndpoint& target_far = side == 0 ? target_cable.endpoint_b : target_cable.endpoint_a;
                    if (*target_anchor.host_id != anchor_target_host || target_anchor.tray_id != anchor_target_tray) {
                        continue;
                    }
                    if (!ports_match(anchor, target_anchor)) {
                        continue;
                    }
                    try_place(index, pattern_cable_idx, far, target_far, target_cable_idx);
                    if (stop_ || abandon(index)) {
                        break;
                    }
                }
                if (stop_ || abandon(index)) {
                    break;
                }
            }

            if (!anchor_tray_was_bound) {
                unbind_tray(anchor_pattern_host, anchor.tray_id, anchor_target_tray);
            }
            if (stop_ || abandon(index)) {
                return;
            }
        }
    }

    // Commit the far end of a candidate cable if it is consistent with the mapping so far, recurse,
    // then roll back.
    void try_place(
        size_t index,
        size_t pattern_cable_idx,
        const CableEndpoint& far,
        const CableEndpoint& target_far,
        size_t target_cable_idx) {
        uint32_t far_pattern_host = *far.host_id;
        uint32_t far_target_host = *target_far.host_id;
        bool bound_host = false;
        if (host_map_[far_pattern_host] == kUnmapped) {
            if (host_rmap_[far_target_host] != kUnmapped) {
                return;
            }
            if (!hosts_compatible(
                    pattern_.hosts()[far_pattern_host], target_.hosts()[far_target_host], options_.tray_symmetry)) {
                return;
            }
            host_map_[far_pattern_host] = far_target_host;
            host_rmap_[far_target_host] = far_pattern_host;
            host_binding_levels_.push_back(static_cast<int>(index));
            ++mapped_hosts_;
            bound_host = true;
        } else if (host_map_[far_pattern_host] != far_target_host) {
            return;
        }

        bool bound_tray = false;
        auto rollback = [&]() {
            if (bound_tray) {
                unbind_tray(far_pattern_host, far.tray_id, target_far.tray_id);
            }
            if (bound_host) {
                host_binding_levels_.pop_back();
                --mapped_hosts_;
                host_rmap_[far_target_host] = kUnmapped;
                host_map_[far_pattern_host] = kUnmapped;
            }
        };

        auto bound = tray_map_[far_pattern_host].find(far.tray_id);
        if (bound != tray_map_[far_pattern_host].end()) {
            if (bound->second != target_far.tray_id) {
                rollback();
                return;
            }
        } else {
            const auto& target_trays = target_.hosts()[far_target_host].trays;
            auto target_tray = target_trays.find(target_far.tray_id);
            if (target_tray == target_trays.end() ||
                target_tray->second != pattern_.hosts()[far_pattern_host].trays.at(far.tray_id) ||
                tray_rmap_[far_pattern_host].contains(target_far.tray_id) ||
                (options_.tray_symmetry == TraySymmetry::None && far.tray_id != target_far.tray_id)) {
                rollback();
                return;
            }
            bind_tray(far_pattern_host, far.tray_id, target_far.tray_id);
            bound_tray = true;
        }

        if (!ports_match(far, target_far)) {
            rollback();
            return;
        }

        target_cable_used_[target_cable_idx] = true;
        cable_map_[pattern_cable_idx] = target_cable_idx;
        recurse(index + 1);
        target_cable_used_[target_cable_idx] = false;
        rollback();
    }

    // Role assignment of the current state: the target host each pattern host of this component
    // plays, in pattern host order. Two placements onto the same set of target hosts differ only in
    // this, which is what makes it the right key for counting a pattern's automorphisms.
    std::vector<uint32_t> role_assignment_of(const std::vector<uint32_t>& host_map) const {
        std::vector<uint32_t> roles;
        roles.reserve(component_hosts_.size());
        for (uint32_t pattern_host : component_hosts_) {
            roles.push_back(host_map[pattern_host]);
        }
        return roles;
    }

    void record_match() {
        // The answer is which target hosts play which pattern roles. Which target cable serves which
        // pattern cable, and which target tray stands in for which pattern tray, are not part of it,
        // so once a placement is complete there is nothing to learn from the other ways of dealing
        // out the same cables and trays. Abandon everything back to the last host decision, which is
        // where a different answer could still come from -- and cannot come from anywhere deeper,
        // since every host is already mapped by this point. Without this, the modes that let a cable
        // land on several ports spend their whole budget enumerating equivalent placements.
        if (!options_.search_every_placement) {
            prune_to_ = host_binding_levels_.empty() ? -1 : host_binding_levels_.back();
        }

        std::vector<uint32_t> role_assignment = role_assignment_of(host_map_);
        std::vector<uint32_t> host_set = role_assignment;
        std::sort(host_set.begin(), host_set.end());

        auto it = found_.find(host_set);
        if (it == found_.end()) {
            it = found_.emplace(host_set, Accumulated{}).first;
        }
        if (it->second.role_assignments.empty() ||
            role_assignment < role_assignment_of(it->second.canonical.host_map)) {
            it->second.canonical.host_map = host_map_;
            it->second.canonical.tray_map = tray_map_;
            it->second.canonical.cable_map = cable_map_;
        }
        it->second.role_assignments.insert(std::move(role_assignment));

        // Stop as soon as the caller has as many host sets as it asked for, rather than going on to
        // prove there are no others. Under the port-agnostic identity modes that proof is the
        // expensive part, so --max-matches 1 is how to ask a cheap "does it fit anywhere" question.
        if (options_.max_matches != 0 && found_.size() >= options_.max_matches) {
            stopped_at_limit_ = true;
            stop_ = true;
        }
    }

    // Explain the state the search is in at the given step: which pattern cable is next, where its
    // anchored end lands in the target, and what the target has there.
    Diagnosis diagnose(size_t index) const {
        Diagnosis diagnosis;
        diagnosis.cables_placed = index;
        diagnosis.partial_host_map = host_map_;
        size_t pattern_cable_idx = order_[index];
        diagnosis.pattern_cable = pattern_cable_idx;
        const auto& pattern_cable = pattern_.cables()[pattern_cable_idx];
        bool anchor_is_a = host_map_[*pattern_cable.endpoint_a.host_id] != kUnmapped;
        const CableEndpoint& anchor = anchor_is_a ? pattern_cable.endpoint_a : pattern_cable.endpoint_b;
        const CableEndpoint& far = anchor_is_a ? pattern_cable.endpoint_b : pattern_cable.endpoint_a;
        uint32_t anchor_target_host = host_map_[*anchor.host_id];
        diagnosis.anchor_host = anchor_target_host;

        // With trays held fixed the anchor tray is known even before anything binds it.
        const auto& bound = tray_map_[*anchor.host_id];
        auto tray = bound.find(anchor.tray_id);
        TrayId anchor_target_tray = anchor.tray_id;
        if (tray != bound.end()) {
            anchor_target_tray = tray->second;
        } else if (options_.tray_symmetry != TraySymmetry::None) {
            diagnosis.reason = fmt::format(
                "no cable on target host {} could serve it, on any tray still free to stand in for tray {}",
                anchor_target_host,
                *anchor.tray_id);
            return diagnosis;
        }
        diagnosis.anchor_tray = anchor_target_tray;

        for (size_t target_cable_idx : target_.cables_at(anchor_target_host, anchor_target_tray)) {
            const auto& target_cable = target_.cables()[target_cable_idx];
            for (int side = 0; side < 2; ++side) {
                const CableEndpoint& target_anchor = side == 0 ? target_cable.endpoint_a : target_cable.endpoint_b;
                const CableEndpoint& target_far = side == 0 ? target_cable.endpoint_b : target_cable.endpoint_a;
                if (*target_anchor.host_id != anchor_target_host || target_anchor.tray_id != anchor_target_tray ||
                    !ports_match(anchor, target_anchor)) {
                    continue;
                }
                diagnosis.rejections.push_back(Diagnosis::Rejection{
                    .candidate = target_anchor,
                    .cable = target_cable,
                    .reason = why_not(target_cable_idx, far, target_far)});
            }
        }
        if (diagnosis.rejections.empty()) {
            switch (options_.port_identity) {
                case PortIdentity::Strict: diagnosis.reason = "the target has no cable on that port"; break;
                case PortIdentity::Chip:
                    diagnosis.reason = "the target has no cable on any port of that tray reaching the same ASIC";
                    break;
                case PortIdentity::Relaxed: diagnosis.reason = "the target has no cable on that tray"; break;
            }
        }
        return diagnosis;
    }

    // Why one particular target cable could not serve the pattern cable being placed.
    std::string why_not(size_t target_cable_idx, const CableEndpoint& far, const CableEndpoint& target_far) const {
        if (target_cable_used_[target_cable_idx]) {
            return "already carrying another pattern cable";
        }
        uint32_t far_mapped = host_map_[*far.host_id];
        if (far_mapped != kUnmapped && far_mapped != *target_far.host_id) {
            return fmt::format(
                "far end is target host {}, but pattern host {} is already mapped to target host {}",
                *target_far.host_id,
                *far.host_id,
                far_mapped);
        }
        if (far_mapped == kUnmapped && host_rmap_[*target_far.host_id] != kUnmapped) {
            return fmt::format(
                "far end is target host {}, already taken by pattern host {}",
                *target_far.host_id,
                host_rmap_[*target_far.host_id]);
        }
        if (!ports_match(far, target_far)) {
            return fmt::format(
                "far end is {}, but the pattern needs {}", port_to_string(target_far), port_to_string(far));
        }
        auto far_bound = tray_map_[*far.host_id].find(far.tray_id);
        if (far_bound != tray_map_[*far.host_id].end() && far_bound->second != target_far.tray_id) {
            return fmt::format(
                "far end is tray {} of target host {}, but pattern tray {} there is already taken to be tray {}",
                *target_far.tray_id,
                *target_far.host_id,
                *far.tray_id,
                *far_bound->second);
        }
        if (far_mapped == kUnmapped &&
            !hosts_compatible(
                pattern_.hosts()[*far.host_id], target_.hosts()[*target_far.host_id], options_.tray_symmetry)) {
            return fmt::format(
                "far end is target host {}, whose boards do not match pattern host {}",
                *target_far.host_id,
                *far.host_id);
        }
        return "no reason found at this port; the placement failed further on";
    }

    const MatchGraph& pattern_;
    const MatchGraph& target_;
    const MatchOptions& options_;
    std::vector<uint32_t> component_hosts_;

    uint32_t seed_ = 0;
    std::vector<size_t> order_;

    std::vector<uint32_t> host_map_;
    std::vector<uint32_t> host_rmap_;
    std::vector<std::map<TrayId, TrayId>> tray_map_;
    std::vector<std::map<TrayId, TrayId>> tray_rmap_;
    std::vector<size_t> cable_map_;
    std::vector<bool> target_cable_used_;

    std::map<std::vector<uint32_t>, Accumulated> found_;
    // Search levels at which a pattern host was bound to a target host, innermost last.
    std::vector<int> host_binding_levels_;
    size_t needed_slots_ = 0;
    size_t bound_slots_ = 0;
    size_t mapped_hosts_ = 0;
    int prune_to_ = kNoPrune;
    size_t deepest_ = 0;
    bool have_diagnosis_ = false;
    std::optional<Diagnosis> diagnosis_;
    uint64_t steps_ = 0;
    uint64_t budget_ = kMaxSearchSteps;
    bool stopped_at_limit_ = false;
    bool exhausted_budget_ = false;
    bool stop_ = false;
};

}  // namespace

const std::set<uint32_t>& asics_for_port(BoardType board_type, PortType port_type, PortId port_id) {
    static std::map<std::tuple<BoardType, PortType, PortId>, std::set<uint32_t>> cache;
    auto key = std::make_tuple(board_type, port_type, port_id);
    auto it = cache.find(key);
    if (it == cache.end()) {
        Board board = create_board(board_type);
        std::set<uint32_t> asics;
        for (const auto& channel : board.get_port_channels(port_type, port_id)) {
            asics.insert(channel.asic_location);
        }
        it = cache.emplace(std::move(key), std::move(asics)).first;
    }
    return it->second;
}

std::vector<std::string> list_graph_templates(const std::string& cabling_path) {
    auto descriptor = load_cluster_descriptor(cabling_path);
    std::vector<std::string> names;
    for (const auto& [name, unused] : descriptor.graph_templates()) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

size_t count_hosts(const cabling_generator::proto::ClusterDescriptor& descriptor) {
    size_t count = 0;
    count_hosts_in_instance(descriptor.root_instance(), count);
    return count;
}

cabling_generator::proto::ClusterDescriptor synthesize_pattern_descriptor(
    const cabling_generator::proto::ClusterDescriptor& source, const std::string& template_name) {
    cabling_generator::proto::ClusterDescriptor pattern = source;
    pattern.clear_root_instance();
    uint32_t next_host_id = 0;
    std::vector<std::string> stack;
    build_instance(source, template_name, pattern.mutable_root_instance(), next_host_id, stack);
    if (next_host_id == 0) {
        throw std::runtime_error(fmt::format("Graph template '{}' contains no nodes", template_name));
    }
    return pattern;
}

MatchGraph MatchGraph::from_generator(const CablingGenerator& generator, TierScope tier, std::string label) {
    MatchGraph graph;
    graph.label_ = std::move(label);

    size_t num_hosts = generator.get_num_hosts();
    const auto& deployment_hosts = generator.get_deployment_hosts();
    graph.hosts_.reserve(num_hosts);
    for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
        HostInfo info;
        info.name =
            host_id < deployment_hosts.size() ? deployment_hosts[host_id].hostname : fmt::format("host_{}", host_id);
        std::vector<std::string> parts;
        for (const auto& [tray_id, board] : generator.get_node(HostId(host_id)).boards) {
            info.trays.emplace(tray_id, board.get_board_type());
            parts.push_back(fmt::format("{}:{}", *tray_id, enchantum::to_string(board.get_board_type())));
        }
        info.signature = fmt::format("{}", fmt::join(parts, ","));
        graph.hosts_.push_back(std::move(info));
    }

    for (auto& cable : generator.get_cables()) {
        if (tier == TierScope::OwnLevel && cable.depth != 0) {
            continue;
        }
        graph.cables_.push_back(std::move(cable));
    }

    graph.index_cables();
    return graph;
}

void MatchGraph::index_cables() {
    cables_by_host_.assign(hosts_.size(), {});
    cables_by_host_tray_.clear();
    for (size_t index = 0; index < cables_.size(); ++index) {
        const auto& cable = cables_[index];
        for (const auto& endpoint : {cable.endpoint_a, cable.endpoint_b}) {
            auto& by_host = cables_by_host_[*endpoint.host_id];
            if (by_host.empty() || by_host.back() != index) {
                by_host.push_back(index);
            }
            auto& by_tray = cables_by_host_tray_[{*endpoint.host_id, endpoint.tray_id}];
            if (by_tray.empty() || by_tray.back() != index) {
                by_tray.push_back(index);
            }
        }
    }
}

MatchGraph MatchGraph::load(
    const std::string& cabling_path,
    const std::string& deployment_path,
    const std::string& template_name,
    TierScope tier,
    std::string label) {
    if (!deployment_path.empty()) {
        if (!template_name.empty()) {
            throw std::runtime_error(
                "A graph template is instantiated with synthetic hosts, so it cannot be combined with a "
                "deployment descriptor");
        }
        return from_generator(CablingGenerator(cabling_path, deployment_path), tier, std::move(label));
    }

    if (std::filesystem::is_directory(cabling_path)) {
        throw std::runtime_error(
            "A directory of cabling descriptors is only merged against a deployment descriptor; pass "
            "--deployment (or --pattern-deployment) alongside it: " +
            cabling_path);
    }
    auto descriptor = load_cluster_descriptor(cabling_path);
    if (!template_name.empty()) {
        descriptor = synthesize_pattern_descriptor(descriptor, template_name);
    }
    size_t num_hosts = count_hosts(descriptor);
    if (num_hosts == 0) {
        throw std::runtime_error("Descriptor has no hosts: " + cabling_path);
    }
    std::vector<std::string> hostnames;
    hostnames.reserve(num_hosts);
    for (size_t host_id = 0; host_id < num_hosts; ++host_id) {
        hostnames.push_back(fmt::format("host_{}", host_id));
    }
    return from_generator(CablingGenerator(descriptor, hostnames), tier, std::move(label));
}

MatchGraph MatchGraph::from_fsd(const std::string& fsd_path, std::string label) {
    fsd::proto::FactorySystemDescriptor fsd = load_factory_system_descriptor(fsd_path);

    MatchGraph graph;
    graph.label_ = std::move(label);

    // Hosts are positional: the i-th entry of hosts is host_id i, which is what connection endpoints
    // and board locations refer to. A board location for a host beyond that list has no host to
    // belong to, so it decides the count when hosts is absent altogether.
    size_t num_hosts = fsd.hosts_size();
    for (const auto& location : fsd.board_types().board_locations()) {
        num_hosts = std::max<size_t>(num_hosts, location.host_id() + 1);
    }
    if (num_hosts == 0) {
        throw std::runtime_error("Factory system descriptor has no hosts: " + fsd_path);
    }

    graph.hosts_.resize(num_hosts);
    for (uint32_t host_id = 0; host_id < num_hosts; ++host_id) {
        graph.hosts_[host_id].name =
            host_id < static_cast<uint32_t>(fsd.hosts_size()) && !fsd.hosts(host_id).hostname().empty()
                ? fsd.hosts(host_id).hostname()
                : fmt::format("host_{}", host_id);
    }
    for (const auto& location : fsd.board_types().board_locations()) {
        BoardType board_type = get_board_type_from_string(location.board_type());
        graph.hosts_[location.host_id()].trays.emplace(TrayId(location.tray_id()), board_type);
    }
    for (auto& host : graph.hosts_) {
        std::vector<std::string> parts;
        for (const auto& [tray_id, board_type] : host.trays) {
            parts.push_back(fmt::format("{}:{}", *tray_id, enchantum::to_string(board_type)));
        }
        host.signature = fmt::format("{}", fmt::join(parts, ","));
    }

    // Fold the channels back into their ports. Each port connection collects the channel connections
    // that belong to it, so that the graph is in the same terms as a cabling descriptor's cables.
    std::map<std::pair<CableEndpoint, CableEndpoint>, size_t> channels_seen;
    size_t split_traces = 0;
    for (const auto& connection : fsd.eth_connections().connection()) {
        auto resolve = [&](const fsd::proto::FactorySystemDescriptor::EndPoint& endpoint) {
            if (endpoint.host_id() >= num_hosts) {
                throw std::runtime_error(fmt::format(
                    "{}: a connection refers to host {}, but the descriptor only has {} hosts",
                    fsd_path,
                    endpoint.host_id(),
                    num_hosts));
            }
            const auto& trays = graph.hosts_[endpoint.host_id()].trays;
            auto tray = trays.find(TrayId(endpoint.tray_id()));
            if (tray == trays.end()) {
                throw std::runtime_error(fmt::format(
                    "{}: a connection uses tray {} of host {}, which has no board type declared",
                    fsd_path,
                    endpoint.tray_id(),
                    endpoint.host_id()));
            }
            const Board& board = board_for(tray->second);
            AsicChannel channel{.asic_location = endpoint.asic_location(), .channel_id = ChanId(endpoint.chan_id())};
            const Port* port = nullptr;
            try {
                port = &board.get_port_for_asic_channel(channel);
            } catch (const std::exception&) {
                throw std::runtime_error(fmt::format(
                    "{}: host {} tray {} is a {}, which has no port carrying ASIC {} channel {}",
                    fsd_path,
                    endpoint.host_id(),
                    endpoint.tray_id(),
                    enchantum::to_string(tray->second),
                    endpoint.asic_location(),
                    endpoint.chan_id()));
            }
            return CableEndpoint{
                .host_id = HostId(endpoint.host_id()),
                .tray_id = TrayId(endpoint.tray_id()),
                .port_type = port->port_type,
                .port_id = port->port_id};
        };

        CableEndpoint endpoint_a = resolve(connection.endpoint_a());
        CableEndpoint endpoint_b = resolve(connection.endpoint_b());

        // A trace is wiring inside a board rather than a cable, and the descriptor side does not
        // report those either.
        if (endpoint_a.port_type == PortType::TRACE || endpoint_b.port_type == PortType::TRACE) {
            split_traces += endpoint_a.port_type == endpoint_b.port_type ? 0 : 1;
            continue;
        }
        channels_seen[std::minmax(endpoint_a, endpoint_b)]++;
    }

    // A cable's channel count is decided by the two ports it joins, so it is known independently of
    // what the descriptor happens to contain, and a disagreement either way is worth reporting.
    // channels_seen is keyed by the port pair, so the cables come out sorted.
    std::vector<std::string> short_of_channels;
    std::vector<std::string> over_channels;
    for (const auto& [ports, channels] : channels_seen) {
        const auto& [endpoint_a, endpoint_b] = ports;
        size_t expected = expected_channel_count(graph.hosts_, endpoint_a, endpoint_b);
        if (channels != expected) {
            (channels < expected ? short_of_channels : over_channels)
                .push_back(fmt::format(
                    "{} <-> {} ({} of {} channels)",
                    port_to_string(endpoint_a),
                    port_to_string(endpoint_b),
                    channels,
                    expected));
        }
        graph.cables_.push_back(
            ResolvedCable{.endpoint_a = endpoint_a, .endpoint_b = endpoint_b, .depth = 0, .declared_at = ""});
    }

    auto note = [&graph](const std::string& text, const std::vector<std::string>& cables) {
        constexpr size_t kMaxListed = 4;
        std::vector<std::string> listed(cables.begin(), cables.begin() + std::min(cables.size(), kMaxListed));
        graph.notes_.push_back(fmt::format(
            "{} of {} cables {}: {}{}",
            cables.size(),
            graph.cables_.size(),
            text,
            fmt::join(listed, "; "),
            cables.size() > kMaxListed ? fmt::format("; and {} more", cables.size() - kMaxListed) : ""));
    };
    if (!short_of_channels.empty()) {
        note("are missing channels and were taken to be present all the same", short_of_channels);
    }
    if (!over_channels.empty()) {
        note("have more channels than their ports can carry, so the descriptor lists some twice", over_channels);
    }
    if (split_traces != 0) {
        graph.notes_.push_back(fmt::format(
            "{} connections join a board-internal trace to a cabled port, which no cabling can produce; they were "
            "dropped",
            split_traces));
    }

    graph.index_cables();
    return graph;
}

const std::vector<size_t>& MatchGraph::cables_at(uint32_t host_id) const {
    static const std::vector<size_t> empty;
    return host_id < cables_by_host_.size() ? cables_by_host_[host_id] : empty;
}

const std::vector<size_t>& MatchGraph::cables_at(uint32_t host_id, TrayId tray_id) const {
    static const std::vector<size_t> empty;
    auto it = cables_by_host_tray_.find({host_id, tray_id});
    return it == cables_by_host_tray_.end() ? empty : it->second;
}

std::vector<uint32_t> MatchGraph::isolated_hosts() const {
    std::vector<uint32_t> isolated;
    for (uint32_t host_id = 0; host_id < hosts_.size(); ++host_id) {
        if (cables_at(host_id).empty()) {
            isolated.push_back(host_id);
        }
    }
    return isolated;
}

std::vector<std::vector<uint32_t>> MatchGraph::components() const {
    std::vector<std::vector<uint32_t>> components;
    std::vector<bool> seen(hosts_.size(), false);
    for (uint32_t start = 0; start < hosts_.size(); ++start) {
        if (seen[start] || cables_at(start).empty()) {
            continue;
        }
        std::vector<uint32_t> component;
        std::deque<uint32_t> queue{start};
        seen[start] = true;
        while (!queue.empty()) {
            uint32_t host = queue.front();
            queue.pop_front();
            component.push_back(host);
            for (size_t cable_idx : cables_at(host)) {
                const auto& cable = cables_[cable_idx];
                for (uint32_t end : {*cable.endpoint_a.host_id, *cable.endpoint_b.host_id}) {
                    if (!seen[end]) {
                        seen[end] = true;
                        queue.push_back(end);
                    }
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }
    return components;
}

MatchResult match(const MatchGraph& pattern, const MatchGraph& target, const MatchOptions& options) {
    MatchResult result;
    result.isolated_pattern_hosts = pattern.isolated_hosts();

    if (pattern.cables().empty()) {
        throw std::runtime_error(
            "The pattern has no cables, so every set of hosts of the right size would match. Pick a template "
            "that declares connections, or widen --tier.");
    }

    auto components = pattern.components();
    if (components.size() > 1 && !options.allow_disconnected) {
        std::vector<std::string> sizes;
        for (const auto& component : components) {
            sizes.push_back(fmt::format("{} hosts", component.size()));
        }
        throw std::runtime_error(fmt::format(
            "The pattern's cables split it into {} disconnected components ({}). Nothing ties the components to "
            "each other, so the matches would be the cross product of their independent placements. Pass "
            "--allow-disconnected to match each component on its own.",
            components.size(),
            fmt::join(sizes, ", ")));
    }

    if (options.mode == MatchMode::Exact) {
        if (pattern.hosts().size() != target.hosts().size()) {
            result.exact_mismatch = fmt::format(
                "exact match needs equal host counts: pattern has {}, target has {}",
                pattern.hosts().size(),
                target.hosts().size());
        } else if (pattern.cables().size() != target.cables().size()) {
            result.exact_mismatch = fmt::format(
                "exact match needs equal cable counts: pattern has {}, target has {}",
                pattern.cables().size(),
                target.cables().size());
        }
    }

    result.matched = result.exact_mismatch.empty();
    for (auto& component : components) {
        Search search(pattern, target, options, component);
        auto component_result = search.run();
        if (component_result.matches.empty()) {
            result.matched = false;
        }
        result.components.push_back(std::move(component_result));
    }
    return result;
}

std::string to_string(PortIdentity identity) {
    switch (identity) {
        case PortIdentity::Strict: return "strict";
        case PortIdentity::Chip: return "chip";
        case PortIdentity::Relaxed: return "relaxed";
    }
    return "unknown";
}

std::string to_string(TraySymmetry symmetry) {
    return symmetry == TraySymmetry::None ? "trays fixed" : "trays interchangeable";
}

std::string to_string(MatchMode mode) { return mode == MatchMode::Contains ? "contains" : "exact"; }

std::string to_string(TierScope tier) { return tier == TierScope::Full ? "full" : "own-level"; }

bool MatchResult::inconclusive() const {
    if (matched) {
        return false;
    }
    return std::any_of(components.begin(), components.end(), [](const ComponentResult& component) {
        return component.matches.empty() && component.exhausted_budget;
    });
}

std::string format_result(
    const MatchGraph& pattern, const MatchGraph& target, const MatchResult& result, const MatchOptions& options) {
    std::ostringstream out;
    out << fmt::format(
        "Pattern: {} ({} hosts, {} cables)\n", pattern.label(), pattern.hosts().size(), pattern.cables().size());
    out << fmt::format(
        "Target:  {} ({} hosts, {} cables)\n", target.label(), target.hosts().size(), target.cables().size());
    out << fmt::format(
        "Mode:    {}, port identity {}, {}\n\n",
        to_string(options.mode),
        to_string(options.port_identity),
        to_string(options.tray_symmetry));

    for (const auto& [graph, side] : {std::pair{&pattern, "Pattern"}, std::pair{&target, "Target"}}) {
        for (const auto& note : graph->notes()) {
            out << fmt::format("Note ({}): {}\n", side, note);
        }
    }
    if (!pattern.notes().empty() || !target.notes().empty()) {
        out << "\n";
    }

    if (!result.isolated_pattern_hosts.empty()) {
        std::vector<std::string> names;
        for (uint32_t host_id : result.isolated_pattern_hosts) {
            names.push_back(pattern.hosts()[host_id].name);
        }
        out << fmt::format(
            "Note: {} pattern host(s) have no cables and were left out of the search ({}). Any compatible target "
            "host would serve for them.\n\n",
            names.size(),
            fmt::join(names, ", "));
    }

    if (!result.exact_mismatch.empty()) {
        out << "NO MATCH\n  " << result.exact_mismatch << "\n";
        return out.str();
    }

    if (result.matched) {
        out << "MATCH\n";
    } else if (result.inconclusive()) {
        out << "INCONCLUSIVE\n  The search budget ran out before every placement had been tried, so this is not a "
               "proof that the pattern does not fit.\n";
    } else {
        out << "NO MATCH\n";
    }

    for (size_t index = 0; index < result.components.size(); ++index) {
        const auto& component = result.components[index];
        if (result.components.size() > 1) {
            out << fmt::format("\nComponent {} ({} pattern hosts)\n", index + 1, component.pattern_hosts.size());
        }
        if (component.matches.empty()) {
            if (component.diagnosis) {
                const auto& diagnosis = *component.diagnosis;
                out << fmt::format(
                    "  Furthest attempt placed {} of {} pattern cables.\n",
                    diagnosis.cables_placed,
                    pattern.cables().size());
                out << fmt::format(
                    "  Stuck on pattern cable {}\n", cable_to_string(pattern.cables()[diagnosis.pattern_cable]));
                out << fmt::format(
                    "    anchored on target host {}{}\n",
                    target.hosts()[diagnosis.anchor_host].name,
                    diagnosis.anchor_tray ? fmt::format(" tray {}", **diagnosis.anchor_tray) : "");
                if (diagnosis.rejections.empty()) {
                    out << fmt::format("    {}\n", diagnosis.reason);
                } else {
                    out << fmt::format(
                        "    {} target cable(s) could have served that end, and none worked:\n",
                        diagnosis.rejections.size());
                    constexpr size_t kMaxShown = 8;
                    for (size_t shown = 0; shown < diagnosis.rejections.size() && shown < kMaxShown; ++shown) {
                        const auto& rejection = diagnosis.rejections[shown];
                        out << fmt::format(
                            "      port {} -> {}: {}\n",
                            *rejection.candidate.port_id,
                            cable_to_string(rejection.cable),
                            rejection.reason);
                    }
                    if (diagnosis.rejections.size() > kMaxShown) {
                        out << fmt::format("      ... and {} more\n", diagnosis.rejections.size() - kMaxShown);
                    }
                }
            }
            continue;
        }

        std::string qualifier;
        if (component.stopped_at_limit) {
            qualifier = " (stopped at --max-matches; there may be more)";
        } else if (component.exhausted_budget) {
            qualifier = " (search budget exhausted; there may be more)";
        }
        out << fmt::format("  {} distinct target host set(s){}\n", component.num_host_sets, qualifier);
        for (size_t match_index = 0; match_index < component.matches.size(); ++match_index) {
            const auto& match = component.matches[match_index];
            std::vector<std::string> roles;
            bool trays_moved = false;
            for (uint32_t pattern_host : component.pattern_hosts) {
                roles.push_back(fmt::format(
                    "{} -> {}", pattern.hosts()[pattern_host].name, target.hosts()[match.host_map[pattern_host]].name));
                for (const auto& [from, to] : match.tray_map[pattern_host]) {
                    trays_moved = trays_moved || from != to;
                }
            }
            out << fmt::format("  #{}: {}\n", match_index + 1, fmt::join(roles, ", "));
            if (match.role_assignments > 1) {
                out << fmt::format(
                    "      {}{} role assignments onto this host set\n",
                    (component.stopped_at_limit || component.exhausted_budget) ? "at least " : "",
                    match.role_assignments);
            }
            if (trays_moved) {
                for (uint32_t pattern_host : component.pattern_hosts) {
                    std::vector<std::string> trays;
                    for (const auto& [from, to] : match.tray_map[pattern_host]) {
                        trays.push_back(fmt::format("{}->{}", *from, *to));
                    }
                    out << fmt::format(
                        "      trays on {}: {}\n", pattern.hosts()[pattern_host].name, fmt::join(trays, " "));
                }
            }
        }
    }
    return out.str();
}

}  // namespace tt::scaleout_tools::matcher
