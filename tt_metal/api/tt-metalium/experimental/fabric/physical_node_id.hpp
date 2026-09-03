// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal {

// The logical address of an ASIC is (cluster_id, tray, loc). PhysicalNodeId packs that address so the
// same chip gets the same id whether it came from a factory system descriptor or from live/mock
// discovery, which is what lets the topology solver produce the same placement on both paths.
//
// The first component is a cluster id, not a hostname: it identifies the group of accelerators
// connected to a common host / controller / root complex. Its value is currently that group's
// hostname -- see the UMD cluster descriptor's cluster_id field, which is where it comes from.

// UMD's CLUSTER_ID_MAX_LENGTH, so every cluster id UMD will hand us fits. The length is set by mock
// cluster-descriptor filenames, which stand in as cluster ids when the descriptor leaves cluster_id
// empty ("sc36_32x4_revc_subtorus_aisled_cluster_desc_bh-glx-120-d05u20_rank_7.yaml" is 73); real
// cluster ids are much shorter -- "bh-glx-110-c01u02" (18), "sjc1-tt-qb-01" (13).
//
// The buffer is NUL-padded, not NUL-terminated: an id of exactly this length fills it with no room
// for a terminator, so read it through cluster_id_view. Never truncated -- a truncated id would
// silently collide with its neighbours.
inline constexpr std::size_t kPhysicalClusterIdLen = 128;

struct PhysicalNodeId {
    char cluster_id[kPhysicalClusterIdLen]{};  // NUL-padded C buffer, not std::array / std::string
    TrayID tray{0};
    ASICLocation loc{0};

    // Defaulted comparison recurses into the C array element-wise, so the struct stays a POD with
    // no handwritten compare. Order is cluster_id bytes, then tray, then loc.
    friend bool operator==(const PhysicalNodeId&, const PhysicalNodeId&) = default;
    friend auto operator<=>(const PhysicalNodeId&, const PhysicalNodeId&) = default;
};

// Hashing the whole object is only valid if there is no padding to pick up garbage from.
static_assert(
    sizeof(PhysicalNodeId) == kPhysicalClusterIdLen + sizeof(TrayID) + sizeof(ASICLocation),
    "PhysicalNodeId has padding; std::hash<PhysicalNodeId> hashes the whole object");
static_assert(
    std::has_unique_object_representations_v<PhysicalNodeId>,
    "PhysicalNodeId is not byte-comparable; std::hash<PhysicalNodeId> hashes the whole object");

// Lowercase, strip a trailing "_<rank>" when hosts are not unique, then take the first DNS label.
// The rank suffix is what run_local_discovery appends when two ranks report the same host
// (hostname + "_" + rank), so it is only stripped when hosts_unique is false.
//
// This is the one canonicalization. The FSD host filter uses it too, so a mix of FQDN and short
// names for the same machine still lands on one string.
std::string canonical_cluster_id_for_node_id(std::string_view cluster_id, bool hosts_unique = true);

// Fatal if the canonical cluster id is empty or longer than kPhysicalClusterIdLen characters (never
// truncated -- a truncated id would silently collide with its neighbours), or if tray or loc does
// not fit in 16 bits.
PhysicalNodeId make_physical_node_id(std::string_view cluster_id, TrayID tray, ASICLocation loc, bool hosts_unique = true);

// Declared here rather than alongside ASICDescriptor so there is one place that builds a node id.
// Forward declared to keep the physical system descriptor out of this header.
struct ASICDescriptor;

// The id of the ASIC a descriptor describes, built from the three address components the descriptor
// already carries. Both the factory-system-descriptor builder and live discovery fill host_name,
// tray_id and asic_location, so this is the single seam through which a descriptor becomes a
// solver key -- pass every descriptor through it instead of reading unique_id, which is 1..N file
// order on the factory path and a UMD chip id on the live path.
//
// hosts_unique comes from PhysicalSystemDescriptor::get_all_hostnames_unique(); it decides whether a
// trailing "_<rank>" on host_name is discovery's uniqueness suffix and must be stripped.
PhysicalNodeId node_id_from_asic_descriptor(const ASICDescriptor& descriptor, bool hosts_unique = true);

// cluster_id is the NUL-trimmed canonical string stored in the id.
struct PhysicalNodeFields {
    std::string cluster_id;
    TrayID tray{0};
    ASICLocation loc{0};
};
PhysicalNodeFields decode_physical_node_id(const PhysicalNodeId& id);

// True for the value-initialized id. make_physical_node_id never returns one: an empty cluster id is
// already fatal.
inline bool is_unset(const PhysicalNodeId& id) { return id == PhysicalNodeId{}; }

// The cluster id as a view into the id's own buffer. The buffer is NUL-padded rather than
// NUL-terminated, so a full-length id has no terminator to find and the view is bounded by the
// buffer instead. Prefer this over decoding when a string_view is enough; do not let
// id.cluster_id decay to a char* and outlive the id.
inline std::string_view cluster_id_view(const PhysicalNodeId& id) {
    const std::string_view buffer(id.cluster_id, kPhysicalClusterIdLen);
    return buffer.substr(0, buffer.find('\0'));
}

std::ostream& operator<<(std::ostream& os, const PhysicalNodeId& id);

}  // namespace tt::tt_metal

namespace std {
template <>
struct hash<tt::tt_metal::PhysicalNodeId> {
    std::size_t operator()(const tt::tt_metal::PhysicalNodeId& id) const noexcept {
        // Whole object. Value-initialization zeroes the unused cluster_id bytes, and the static_asserts
        // above rule out padding, so equal ids always hash equally. This is a container hash only --
        // it is not the node's identity.
        return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&id), sizeof(id)));
    }
};
}  // namespace std

template <>
struct fmt::formatter<tt::tt_metal::PhysicalNodeId> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::PhysicalNodeId& id, format_context& ctx) const -> format_context::iterator;
};
