// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <board/board.hpp>
#include <cabling_generator/cabling_generator.hpp>

namespace tt::scaleout_tools::cabling_generator::proto {
class ClusterDescriptor;
}

namespace tt::scaleout_tools::matcher {

// How much freedom the matcher has when calling a pattern cable and a target cable the same
// connection. Every mode requires the port type to agree; they differ in what they do with port ids.
enum class PortIdentity {
    // Port ids must agree. A factory descriptor's cable is matched only by the same cable.
    Strict,
    // Ports are compared by the ASIC they reach, read from the board's port-to-channel map. A port
    // whose channels span two ASICs counts as reaching either one (fixed or configured to one of
    // two), so two ports match when there is at least one ASIC they can both reach.
    Chip,
    // Port ids are ignored; only the host and tray of each endpoint matter.
    Relaxed,
};

// Whether trays within a host may be relabelled. Factory descriptors number trays by physical slot,
// so the default holds them fixed; Full searches every tray bijection that preserves board types,
// which is what finds a scheme cabled onto a different set of trays.
enum class TraySymmetry { None, Full };

// Contains: the pattern must appear somewhere in the target. Exact: it must additionally account for
// every host and every cable of the target.
enum class MatchMode { Contains, Exact };

// Full: the pattern is every cable in the root template's subtree. OwnLevel: only the cables that
// template declares itself, which is how to ask whether the outer tiers agree while ignoring a
// mismatch further in.
enum class TierScope { Full, OwnLevel };

struct MatchOptions {
    PortIdentity port_identity = PortIdentity::Strict;
    TraySymmetry tray_symmetry = TraySymmetry::None;
    MatchMode mode = MatchMode::Contains;
    // Number of distinct target host sets to report; the search stops once this many are found, so 1
    // asks only whether the pattern fits anywhere. 0 searches for all of them, which under the
    // port-agnostic identity modes is far and away the most expensive thing to ask for.
    size_t max_matches = 16;
    // Match each connected component of the pattern separately instead of rejecting a pattern whose
    // cables do not tie all of its hosts together.
    bool allow_disconnected = false;
    // Visit every placement instead of abandoning the ways of dealing out the same cables and trays
    // that are equivalent to one already recorded. Costs a great deal and changes nothing, so this is
    // here for tests that check the pruning against the unpruned search.
    bool search_every_placement = false;
};

// A cluster reduced to what matching needs: hosts with their board layout, and cables. Host ids
// index hosts() directly, matching the dense 0..N-1 space CablingGenerator assigns.
class MatchGraph {
public:
    struct HostInfo {
        std::string name;
        std::map<TrayId, BoardType> trays;
        // Board types in tray order, for candidate pre-filtering. Two hosts can only stand in for
        // each other if these agree, so a pattern leaf declaring a UBB galaxy never matches a P150.
        std::string signature;
    };

    // Load a cabling descriptor into a graph. deployment_path may be empty, in which case hosts are
    // named host_0..host_N-1 and cabling_path must be a single file. template_name, when non-empty,
    // names a graph_template to instantiate as the root instead of the descriptor's own
    // root_instance; see synthesize_pattern_descriptor.
    static MatchGraph load(
        const std::string& cabling_path,
        const std::string& deployment_path,
        const std::string& template_name,
        TierScope tier,
        std::string label);

    static MatchGraph from_generator(const CablingGenerator& generator, TierScope tier, std::string label);

    const std::string& label() const { return label_; }
    const std::vector<HostInfo>& hosts() const { return hosts_; }
    const std::vector<ResolvedCable>& cables() const { return cables_; }

    // Cable indices touching a host, and touching a specific tray of a host.
    const std::vector<size_t>& cables_at(uint32_t host_id) const;
    const std::vector<size_t>& cables_at(uint32_t host_id, TrayId tray_id) const;

    // Hosts with no cables, which every host of a compatible target would match.
    std::vector<uint32_t> isolated_hosts() const;

    // Connected components over the cables, as pattern host id sets. Isolated hosts are excluded.
    std::vector<std::vector<uint32_t>> components() const;

private:
    std::string label_;
    std::vector<HostInfo> hosts_;
    std::vector<ResolvedCable> cables_;
    std::vector<std::vector<size_t>> cables_by_host_;
    std::map<std::pair<uint32_t, TrayId>, std::vector<size_t>> cables_by_host_tray_;
};

// One placement of the pattern into the target.
struct Match {
    std::vector<uint32_t> host_map;                  // pattern host id -> target host id
    std::vector<std::map<TrayId, TrayId>> tray_map;  // per pattern host; identity unless trays move
    std::vector<size_t> cable_map;                   // pattern cable index -> target cable index
    // Distinct role assignments found onto this same set of target hosts, this one included.
    uint64_t role_assignments = 1;
};

// Where the search got stuck, taken from the attempt that placed the most cables. Reported when
// nothing matched: the matcher always knows which pattern cable it could not place, which target
// cables it weighed for that cable, and what was wrong with each of them.
struct Diagnosis {
    size_t cables_placed = 0;
    size_t pattern_cable = 0;
    // Target host the anchored end of that cable landed on, and the tray when one is determined.
    uint32_t anchor_host = 0;
    std::optional<TrayId> anchor_tray;

    // A target cable that could have served the anchored end, and why it did not work out. Under the
    // port-agnostic identity modes there is usually more than one, and reporting a single one of them
    // reads as if it were the only option.
    struct Rejection {
        CableEndpoint candidate;  // the target port the anchored end would have used
        ResolvedCable cable;      // the cable the target has there
        std::string reason;
    };
    std::vector<Rejection> rejections;
    // Why there was nothing to try at all. Set only when rejections is empty.
    std::string reason;
    // Pattern host -> target host at the point of failure; kUnmapped where undecided.
    std::vector<uint32_t> partial_host_map;
    static constexpr uint32_t kUnmapped = ~0u;
};

struct ComponentResult {
    std::vector<uint32_t> pattern_hosts;
    std::vector<Match> matches;
    size_t num_host_sets = 0;
    bool stopped_at_limit = false;
    bool exhausted_budget = false;
    std::optional<Diagnosis> diagnosis;
};

struct MatchResult {
    bool matched = false;
    // One entry per connected component of the pattern. A connected pattern has exactly one.
    std::vector<ComponentResult> components;
    // Pattern hosts dropped for having no cables to constrain them.
    std::vector<uint32_t> isolated_pattern_hosts;
    // Set when Exact was asked for and the counts rule it out, independent of any placement.
    std::string exact_mismatch;

    // A failure is only a proof that the pattern does not fit if the search ran to completion.
    // Otherwise all that is known is that nothing turned up in the steps available.
    bool inconclusive() const;
};

MatchResult match(const MatchGraph& pattern, const MatchGraph& target, const MatchOptions& options);

// Turn a graph_template into a loadable ClusterDescriptor by instantiating it as the root and
// assigning host ids 0..k-1 to its leaf nodes in template child order. A graph_template is not
// loadable on its own: it has no root_instance, so nothing binds its leaves to hosts.
cabling_generator::proto::ClusterDescriptor synthesize_pattern_descriptor(
    const cabling_generator::proto::ClusterDescriptor& source, const std::string& template_name);

// Leaf nodes under a descriptor's root_instance, i.e. how many hostnames a CablingGenerator needs.
size_t count_hosts(const cabling_generator::proto::ClusterDescriptor& descriptor);

// Names of the graph templates a cabling descriptor defines, sorted. These are the candidates for
// MatchGraph::load's template_name.
std::vector<std::string> list_graph_templates(const std::string& cabling_path);

// ASIC locations a port can reach. More than one means the port is read as reaching either of them;
// see PortIdentity::Chip.
const std::set<uint32_t>& asics_for_port(BoardType board_type, PortType port_type, PortId port_id);

std::string to_string(PortIdentity identity);
std::string to_string(TraySymmetry symmetry);
std::string to_string(MatchMode mode);
std::string to_string(TierScope tier);

// Human-readable report of a match run, as printed by run_cabling_matcher.
std::string format_result(
    const MatchGraph& pattern, const MatchGraph& target, const MatchResult& result, const MatchOptions& options);

}  // namespace tt::scaleout_tools::matcher
