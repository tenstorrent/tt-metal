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

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

constexpr uint32_t kMaxPackedComponent = 0xffff;

// True when every dot-separated label is a legal DNS label, i.e. the dots are domain separators
// rather than ordinary characters. Synthesized host ids such as "dual_glx_2.5d_torus_cluster_desc"
// (used when a mock cluster descriptor leaves host_id empty) also contain dots, and cutting those
// at the first dot would merge distinct hosts into one address.
bool looks_like_fqdn(std::string_view host_id) {
    if (host_id.find('.') == std::string_view::npos) {
        return false;
    }
    for (std::size_t start = 0;;) {
        const std::size_t dot = host_id.find('.', start);
        const std::size_t end = dot == std::string_view::npos ? host_id.size() : dot;
        const std::string_view label = host_id.substr(start, end - start);
        if (label.empty()) {
            return false;
        }
        const bool dns_label = std::all_of(label.begin(), label.end(), [](const char c) {
            return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-';
        });
        if (!dns_label) {
            return false;
        }
        if (dot == std::string_view::npos) {
            return true;
        }
        start = dot + 1;
    }
}

}  // namespace

std::string canonical_host_for_node_id(std::string_view host_id) {
    // A leading dot is an empty first label, so there is no name here to address. Rejected up front
    // because the FQDN strip below no longer runs unconditionally and would otherwise let it pass.
    if (!host_id.empty() && host_id.front() == '.') {
        return {};
    }

    // First DNS label: an FQDN and its short name have to canonicalize to the same string, because
    // one side of a join may report either. Only when the dots really are domain separators.
    if (looks_like_fqdn(host_id)) {
        host_id = host_id.substr(0, host_id.find('.'));
    }

    std::string canonical(host_id);
    std::transform(canonical.begin(), canonical.end(), canonical.begin(), [](const char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });
    return canonical;
}

PhysicalNodeId make_physical_node_id(std::string_view host_id, TrayID tray, ASICLocation loc) {
    const std::string canonical = canonical_host_for_node_id(host_id);

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

PhysicalNodeId node_id_from_asic_descriptor(const ASICDescriptor& descriptor) {
    return make_physical_node_id(descriptor.host_name, descriptor.tray_id, descriptor.asic_location);
}

PhysicalNodeIdIndex build_physical_node_id_index(const PhysicalSystemDescriptor& descriptor) {
    PhysicalNodeIdIndex index;
    const auto& asic_descriptors = descriptor.get_asic_descriptors();
    index.node_id_to_asic_id.reserve(asic_descriptors.size());
    index.asic_id_to_node_id.reserve(asic_descriptors.size());

    for (const auto& [asic_id, asic_descriptor] : asic_descriptors) {
        const PhysicalNodeId node_id = node_id_from_asic_descriptor(asic_descriptor);
        const auto [it, inserted] = index.node_id_to_asic_id.emplace(node_id, asic_id);
        TT_FATAL(
            inserted,
            "Two ASICs in the physical system descriptor share the address {}: ids {} and {}. An address names one "
            "chip, so this would merge them into a single topology node.",
            node_id,
            it->second,
            asic_id);
        index.asic_id_to_node_id.emplace(asic_id, node_id);
    }
    return index;
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
