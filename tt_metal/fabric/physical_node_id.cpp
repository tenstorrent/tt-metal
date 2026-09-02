// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/physical_node_id.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

constexpr uint32_t kMaxPackedComponent = 0xffff;

// A trailing "_<digits>" is the uniqueness suffix run_local_discovery appends as
// hostname + "_" + rank. Only a run of digits counts, so a host id that genuinely ends in
// "_something" is left alone.
std::string_view strip_rank_suffix(std::string_view host_id) {
    const std::size_t underscore = host_id.find_last_of('_');
    if (underscore == std::string_view::npos || underscore + 1 == host_id.size()) {
        return host_id;
    }
    const std::string_view suffix = host_id.substr(underscore + 1);
    const bool all_digits = std::all_of(
        suffix.begin(), suffix.end(), [](const char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
    return all_digits ? host_id.substr(0, underscore) : host_id;
}

}  // namespace

std::string canonical_host_for_node_id(std::string_view host_id, bool hosts_unique) {
    if (!hosts_unique) {
        host_id = strip_rank_suffix(host_id);
    }

    // First DNS label: an FQDN and its short name have to canonicalize to the same string, because
    // one side of a join may report either.
    host_id = host_id.substr(0, host_id.find('.'));

    std::string canonical(host_id);
    std::transform(canonical.begin(), canonical.end(), canonical.begin(), [](const char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    return canonical;
}

PhysicalNodeId make_physical_node_id(std::string_view host_id, TrayID tray, ASICLocation loc, bool hosts_unique) {
    const std::string canonical = canonical_host_for_node_id(host_id, hosts_unique);

    TT_FATAL(
        !canonical.empty(),
        "Cannot build a PhysicalNodeId with an empty host id (from \"{}\"): the host id is part of the ASIC's "
        "address, and an unset one would collide with every other host.",
        host_id);
    TT_FATAL(
        canonical.size() < kPhysicalHostNameLen,
        "Host id \"{}\" is {} characters, which does not fit a PhysicalNodeId (limit is {}). It is not truncated, "
        "because a truncated host id would silently collide with its neighbours.",
        canonical,
        canonical.size(),
        kPhysicalHostNameLen - 1);
    TT_FATAL(*tray <= kMaxPackedComponent, "Tray id {} does not fit in 16 bits (host id \"{}\").", *tray, canonical);
    TT_FATAL(
        *loc <= kMaxPackedComponent,
        "ASIC location {} does not fit in 16 bits (host id \"{}\", tray {}).",
        *loc,
        canonical,
        *tray);

    PhysicalNodeId id{};  // Zeroes the host_id buffer, so the unused tail is NUL padding.
    std::memcpy(id.host_id, canonical.data(), canonical.size());
    id.tray = tray;
    id.loc = loc;
    return id;
}

PhysicalNodeFields decode_physical_node_id(const PhysicalNodeId& id) {
    return PhysicalNodeFields{std::string(host_id_view(id)), id.tray, id.loc};
}

std::ostream& operator<<(std::ostream& os, const PhysicalNodeId& id) {
    return os << "(" << host_id_view(id) << ", tray " << *id.tray << ", loc " << *id.loc << ")";
}

}  // namespace tt::tt_metal

auto fmt::formatter<tt::tt_metal::PhysicalNodeId>::format(
    const tt::tt_metal::PhysicalNodeId& id, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(ctx.out(), "({}, tray {}, loc {})", tt::tt_metal::host_id_view(id), *id.tray, *id.loc);
}
