// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fsd_host_filter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <protobuf/factory_system_descriptor.pb.h>

#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

namespace {

// Enough names to recognize which system you are looking at, without printing a datacenter.
constexpr std::size_t kNamesToPrint = 5;

std::string sample(const std::set<std::string>& names) {
    std::vector<std::string> head(names.begin(), std::next(names.begin(), std::min(kNamesToPrint, names.size())));
    return names.size() > kNamesToPrint ? fmt::format("{}, ... ({} total)", fmt::join(head, ", "), names.size())
                                        : fmt::format("{}", fmt::join(head, ", "));
}

// FNV-1a. Explicit rather than std::hash because this value is compared across processes: it has to be a
// function of the bytes alone, not of anything the standard library is free to vary.
std::uint64_t fnv1a(std::string_view bytes, std::uint64_t seed = 0xcbf29ce484222325ULL) {
    std::uint64_t hash = seed;
    for (const char byte : bytes) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

}  // namespace

std::vector<std::string> fsd_host_filter_from_live(
    const std::string& fsd_path, const ::tt::tt_metal::PhysicalSystemDescriptor& live) {
    // Canonical live host set, and the spellings that produced each name so a collision can name them.
    std::map<std::string, std::vector<std::string>> live_spellings;
    for (const auto& hostname : live.get_all_hostnames()) {
        live_spellings[::tt::tt_metal::canonical_host_for_node_id(hostname)].push_back(hostname);
    }

    // Two live hosts under one canonical name make the address join ambiguous: cables from one machine
    // would be attached to the other. Nothing downstream can recover from that, so stop here.
    std::vector<std::string> ambiguous;
    for (const auto& [canonical, spellings] : live_spellings) {
        if (spellings.size() > 1) {
            ambiguous.push_back(fmt::format("{} <- {}", canonical, fmt::join(spellings, ", ")));
        }
    }
    if (!ambiguous.empty()) {
        throw std::runtime_error(fmt::format(
            "Live host names are not distinct after canonicalization: {}. The factory-descriptor join is "
            "keyed on the canonical name, so it cannot tell these hosts apart.",
            fmt::join(ambiguous, "; ")));
    }

    std::vector<std::string> hosts;
    hosts.reserve(live_spellings.size());
    for (const auto& [canonical, spellings] : live_spellings) {
        hosts.push_back(canonical);
    }

    const auto fsd = load_factory_descriptor(fsd_path);

    std::set<std::string> fsd_hosts;
    for (int i = 0; i < fsd.hosts_size(); ++i) {
        fsd_hosts.insert(::tt::tt_metal::canonical_host_for_node_id(fsd.hosts(i).hostname()));
    }

    // Zero overlap gets its own message. It is a different operator problem from a partial mismatch --
    // the wrong descriptor rather than a stale one -- and the fix is different, so a list of every live
    // host "missing" would bury the useful signal.
    const bool any_overlap = std::any_of(
        hosts.begin(), hosts.end(), [&fsd_hosts](const std::string& host) { return fsd_hosts.contains(host); });
    if (!any_overlap) {
        throw std::runtime_error(fmt::format(
            "No live host appears in the Factory System Descriptor '{}', so it does not describe this system. "
            "Live hosts: {}. Descriptor hosts: {}.",
            fsd_path,
            sample(std::set<std::string>(hosts.begin(), hosts.end())),
            sample(fsd_hosts)));
    }

    // Everything else an ingest would reject: hosts absent from the descriptor, and descriptor hosts that
    // collide under canonicalization. Run as a dry run so this function throws for every reason the ingest
    // would, which is what lets the caller decide collectively before committing.
    validate_host_filter(fsd, hosts);

    return hosts;
}

std::uint64_t checksum_sorted_host_list(const std::vector<std::string>& hosts) {
    std::vector<std::string> sorted(hosts);
    std::sort(sorted.begin(), sorted.end());
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const auto& host : sorted) {
        // NUL-separated so {"ab", "c"} and {"a", "bc"} do not hash alike.
        hash = fnv1a(host, hash);
        hash = fnv1a(std::string_view("\0", 1), hash);
    }
    return hash;
}

std::uint64_t fsd_fingerprint(const std::string& fsd_path) {
    std::ifstream file(fsd_path, std::ios::binary);
    if (!file.is_open()) {
        return 0;
    }
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    std::array<char, 64 * 1024> buffer{};
    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        hash = fnv1a(std::string_view(buffer.data(), static_cast<std::size_t>(file.gcount())), hash);
    }
    return hash;
}

void agree_or_throw_fsd_host_filter(
    const ::tt::tt_metal::distributed::multihost::DistributedContext& ctx,
    std::uint64_t host_checksum,
    std::uint64_t descriptor_fingerprint,
    bool local_ok) {
    // One buffer, two reductions. Comparing each value's min against its max detects disagreement without
    // needing to know what any other rank actually holds, which is what keeps the message rank-independent.
    std::array<std::uint64_t, 3> local{static_cast<std::uint64_t>(local_ok), host_checksum, descriptor_fingerprint};
    std::array<std::uint64_t, 3> mins{};
    std::array<std::uint64_t, 3> maxes{};
    using ::tt::tt_metal::distributed::multihost::ReduceOp;
    ctx.all_reduce(ttsl::Span<std::uint64_t>(local), ttsl::Span<std::uint64_t>(mins), ReduceOp::MIN);
    ctx.all_reduce(ttsl::Span<std::uint64_t>(local), ttsl::Span<std::uint64_t>(maxes), ReduceOp::MAX);

    const bool every_rank_ok = mins[0] != 0;
    const bool checksums_agree = mins[1] == maxes[1];
    const bool fingerprints_agree = mins[2] == maxes[2];
    if (every_rank_ok && checksums_agree && fingerprints_agree) {
        return;
    }

    // Built only from reduced values, so every rank formats the same bytes.
    const auto message = fmt::format(
        "Factory System Descriptor host filter is not identical on every rank (local_ok_min={}, "
        "host_checksum min={} max={}, fsd_fingerprint min={} max={}). Every rank fails together with this "
        "message. There is no per-rank fallback to live mapping: one rank mapping on the factory descriptor "
        "while another maps on live makes their downed-link sets disagree, and any collective gated on that "
        "deadlocks.",
        mins[0],
        mins[1],
        maxes[1],
        mins[2],
        maxes[2]);
    log_error(tt::LogFabric, "{}", message);
    throw std::runtime_error(message);
}

void align_factory_descriptor_with_live(
    ::tt::tt_metal::PhysicalSystemDescriptor& fsd, const ::tt::tt_metal::PhysicalSystemDescriptor& live) {
    std::map<std::string, std::string> live_by_canonical;
    for (const auto& hostname : live.get_all_hostnames()) {
        live_by_canonical.emplace(::tt::tt_metal::canonical_host_for_node_id(hostname), hostname);
    }

    auto& fsd_ranks = fsd.get_host_to_rank_map();
    for (auto& [fsd_hostname, rank] : fsd_ranks) {
        const auto live_hostname = live_by_canonical.find(::tt::tt_metal::canonical_host_for_node_id(fsd_hostname));
        TT_FATAL(
            live_hostname != live_by_canonical.end(),
            "Factory descriptor host '{}' has no live counterpart. The descriptor should already have been "
            "filtered to the live host set.",
            fsd_hostname);
        rank = live.get_rank_for_hostname(live_hostname->second);
    }

    // This process's own host, under the descriptor's spelling of it.
    const auto my_canonical = ::tt::tt_metal::canonical_host_for_node_id(live.my_host_name());
    std::string my_fsd_hostname;
    for (const auto& [fsd_hostname, rank] : fsd_ranks) {
        if (::tt::tt_metal::canonical_host_for_node_id(fsd_hostname) == my_canonical) {
            my_fsd_hostname = fsd_hostname;
            break;
        }
    }
    TT_FATAL(
        !my_fsd_hostname.empty(),
        "The local host '{}' is not in the factory descriptor, so there is nothing here to map onto.",
        live.my_host_name());

    fsd.set_discovery_data(
        my_fsd_hostname, live.get_rank_for_hostname(live.my_host_name()), live.get_all_hostnames_unique());
}

void throw_on_fsd_chips_absent_from_live(
    const ::tt::tt_metal::PhysicalSystemDescriptor& fsd, const ::tt::tt_metal::PhysicalSystemDescriptor& live) {
    const auto fsd_index = ::tt::tt_metal::build_physical_node_id_index(fsd);
    const auto live_index = ::tt::tt_metal::build_physical_node_id_index(live);

    std::set<std::string> live_hosts;
    for (const auto& hostname : live.get_all_hostnames()) {
        live_hosts.insert(::tt::tt_metal::canonical_host_for_node_id(hostname));
    }

    std::vector<std::string> absent;
    for (const auto& [node_id, asic_id] : fsd_index.node_id_to_asic_id) {
        if (!live_hosts.contains(std::string(::tt::tt_metal::host_id_view(node_id)))) {
            continue;  // discovery never looked at this host -- see the header
        }
        if (!live_index.node_id_to_asic_id.contains(node_id)) {
            absent.push_back(fmt::format("{}", node_id));
        }
    }
    if (absent.empty()) {
        return;
    }
    // Sorted because the index is unordered: without this, two ranks would report the same chips in
    // different orders and the operator would see several distinct-looking failures.
    std::sort(absent.begin(), absent.end());
    throw std::runtime_error(fmt::format(
        "Factory System Descriptor: {} chip(s) it expects are absent from the live cluster: {}. A missing "
        "chip is not a downed-link case -- check the allocation and the boards.",
        absent.size(),
        fmt::join(absent, ", ")));
}

}  // namespace tt::tt_metal::experimental::tt_fabric
