// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_findings.hpp"

#include <string_view>
#include <type_traits>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::decode {

namespace {

Finding make_finding(FindingCode code, std::string path, std::string message) {
    return Finding{.code = code, .paths = {std::move(path)}, .message = std::move(message)};
}

// The finding for why a router isn't read in full. The snapshot's reason is one of the router's.
FindingCode get_router_unread_finding_code(UnreadCode code) {
    switch (code) {
        case UnreadCode::NOT_IN_RECORD:
        case UnreadCode::CHIP_MISSING:
        case UnreadCode::CHIP_ARCH_MISMATCH:
        case UnreadCode::CHANNEL_NOT_FOUND:
        case UnreadCode::CORE_NOT_FOUND: return FindingCode::ROUTER_NOT_READ;
        case UnreadCode::L1_READ_FAILED: return FindingCode::L1_READ_FAILED;
        case UnreadCode::REGISTER_READ_FAILED: return FindingCode::REGISTER_READ_FAILED;
        case UnreadCode::REGISTER_NOT_CAPTURED:
        case UnreadCode::OUTSIDE_CAPTURED_L1:
        case UnreadCode::TYPE_LARGER_THAN_ELEMENT: break;
    }
    TT_THROW("Fabric decode: {} is a region's reason, not a router's", manifest::lower_enum_name(code));
}

// The finding for an unread region, or null when the region is unread only because its router is, which the router's
// own finding says once.
std::optional<FindingCode> find_region_unread_finding_code(UnreadCode code) {
    switch (code) {
        case UnreadCode::REGISTER_NOT_CAPTURED: return FindingCode::REGISTER_NOT_CAPTURED;
        case UnreadCode::OUTSIDE_CAPTURED_L1: return FindingCode::OUTSIDE_CAPTURED_L1;
        case UnreadCode::TYPE_LARGER_THAN_ELEMENT: return FindingCode::TYPE_LARGER_THAN_ELEMENT;
        case UnreadCode::NOT_IN_RECORD:
        case UnreadCode::CHIP_MISSING:
        case UnreadCode::CHIP_ARCH_MISMATCH:
        case UnreadCode::CHANNEL_NOT_FOUND:
        case UnreadCode::CORE_NOT_FOUND:
        case UnreadCode::L1_READ_FAILED:
        case UnreadCode::REGISTER_READ_FAILED: return std::nullopt;
    }
    TT_THROW("Fabric decode: unknown unread code {}", static_cast<int>(code));
}

// What capture recorded of the router: its full read's failure, its rounds' first failure, and a core mismatch.
void add_capture_findings(const RouterSnapshot& snapshot, std::vector<Finding>& findings) {
    const CapturedRouter& router = snapshot.get_router();
    if (const std::optional<Unread>& reason = snapshot.get_unread_reason()) {
        findings.push_back(make_finding(get_router_unread_finding_code(reason->code), router.path, reason->message));
    }
    if (router.record == nullptr) {
        return;
    }
    for (const capture::RoundRecord& round : router.record->rounds) {
        for (const capture::ReadRecord& read : round.reads) {
            if (read.error.has_value()) {
                findings.push_back(make_finding(
                    FindingCode::ROUND_READ_FAILED,
                    router.path,
                    fmt::format("round {}, the read at {:#x}: {}", round.round, read.address, *read.error)));
            }
        }
    }
    if (router.record->target.location == capture::CoreLocation::CORE_MISMATCH) {
        findings.push_back(make_finding(
            FindingCode::CORE_MISMATCH,
            router.path,
            "capture read the manifest's core, though the reader puts the router's channel on another"));
    }
}

// The regions unread for a reason of their own.
void add_region_findings(const RouterValues& values, std::vector<Finding>& findings) {
    const std::string& router_path = values.get_snapshot().get_router().path;
    for (const auto& [path, unread] : values.get_unread_regions()) {
        if (const std::optional<FindingCode> code = find_region_unread_finding_code(unread.code)) {
            findings.push_back(make_finding(*code, fmt::format("{}/{}", router_path, path), unread.message));
        }
    }
}

// An enum region whose reads hold a value no enumerator names: one finding per region, with the first such value.
// Enums inside structs are shown by their struct's interpreter, with a null name.
void add_unknown_enumerator_finding(
    const RouterValues& values, std::string_view path, std::string_view type_name, std::vector<Finding>& findings) {
    manifest::visit_registered_type(type_name, [&]<typename T>(std::type_identity<T>) {
        if constexpr (std::is_enum_v<T>) {
            const RegionReads<T>* reads = values.find_reads<T>(path);
            if (reads == nullptr) {
                return;
            }
            for (const RegionRead<T>& read : *reads) {
                for (const T value : read.elements) {
                    if (!find_enum_name(value).has_value()) {
                        findings.push_back(make_finding(
                            FindingCode::UNKNOWN_ENUMERATOR,
                            fmt::format("{}/{}", values.get_snapshot().get_router().path, path),
                            fmt::format(
                                "{:#x} names no {}", static_cast<std::underlying_type_t<T>>(value), type_name)));
                        return;
                    }
                }
            }
        }
    });
}

void add_unknown_enumerator_findings(
    const RouterValues& values, const manifest::Arch& arch, std::vector<Finding>& findings) {
    manifest::for_each_router_region(
        *values.get_snapshot().get_router().router, [&](std::string_view path, const auto& region) {
            add_unknown_enumerator_finding(values, path, region.type, findings);
        });
    manifest::for_each_arch_area(arch, [&](std::string_view name, const manifest::content::L1& area) {
        add_unknown_enumerator_finding(values, make_arch_area_path(name), area.type, findings);
    });
}

// Each VC's connected-but-not-live and live-but-not-connected directions.
void add_direction_findings(
    const std::string& router_path,
    const std::optional<std::vector<VcDirections>>& directions,
    std::vector<Finding>& findings) {
    if (!directions.has_value()) {
        return;
    }
    for (const VcDirections& vc : *directions) {
        if (const auto bits = static_cast<uint8_t>(vc.connected & ~vc.live); bits != 0) {
            findings.push_back(make_finding(
                FindingCode::CONNECTED_NOT_LIVE,
                router_path,
                fmt::format("vc{} is connected to directions {:#04x} it never forwards to", vc.vc, bits)));
        }
        if (const auto bits = static_cast<uint8_t>(vc.live & ~vc.connected); bits != 0) {
            findings.push_back(make_finding(
                FindingCode::LIVE_NOT_CONNECTED,
                router_path,
                fmt::format("vc{} forwards to directions {:#04x} with no sibling", vc.vc, bits)));
        }
    }
}

}  // namespace

FindingSeverity get_finding_severity(FindingCode code) {
    switch (code) {
        case FindingCode::ROUTER_NOT_READ:
        case FindingCode::L1_READ_FAILED:
        case FindingCode::REGISTER_READ_FAILED:
        case FindingCode::ROUTER_NOT_IN_MANIFEST:
        case FindingCode::REGISTER_NOT_CAPTURED:
        case FindingCode::OUTSIDE_CAPTURED_L1:
        case FindingCode::TYPE_LARGER_THAN_ELEMENT: return FindingSeverity::ERROR;
        case FindingCode::ROUND_READ_FAILED:
        case FindingCode::CORE_MISMATCH:
        case FindingCode::UNKNOWN_ENUMERATOR:
        case FindingCode::ROUTING_TABLE_COPIES_DIFFER:
        case FindingCode::ROUTING_TABLE_SHAPE_MISMATCH:
        case FindingCode::ROUTING_TABLE_MESH_MEMBERS_DIFFER: return FindingSeverity::WARNING;
        case FindingCode::ROUTING_TABLE_IDENTITY_MISMATCH:
        case FindingCode::ROUTING_TABLE_SHAPE_INVALID: return FindingSeverity::ERROR;
        // Both differences happen on a correct router (decision 20): a corner chip is live toward a missing sibling.
        case FindingCode::CONNECTED_NOT_LIVE:
        case FindingCode::LIVE_NOT_CONNECTED: return FindingSeverity::INFO;
    }
    TT_THROW("Fabric decode: unknown finding code {}", static_cast<int>(code));
}

FindingOrigin get_finding_origin(FindingCode code) {
    switch (code) {
        case FindingCode::ROUTER_NOT_READ:
        case FindingCode::L1_READ_FAILED:
        case FindingCode::REGISTER_READ_FAILED:
        case FindingCode::ROUND_READ_FAILED:
        case FindingCode::CORE_MISMATCH:
        case FindingCode::ROUTER_NOT_IN_MANIFEST: return FindingOrigin::CAPTURE;
        case FindingCode::REGISTER_NOT_CAPTURED:
        case FindingCode::OUTSIDE_CAPTURED_L1:
        case FindingCode::TYPE_LARGER_THAN_ELEMENT:
        case FindingCode::UNKNOWN_ENUMERATOR:
        case FindingCode::CONNECTED_NOT_LIVE:
        case FindingCode::LIVE_NOT_CONNECTED:
        case FindingCode::ROUTING_TABLE_COPIES_DIFFER:
        case FindingCode::ROUTING_TABLE_IDENTITY_MISMATCH:
        case FindingCode::ROUTING_TABLE_SHAPE_MISMATCH:
        case FindingCode::ROUTING_TABLE_SHAPE_INVALID:
        case FindingCode::ROUTING_TABLE_MESH_MEMBERS_DIFFER: return FindingOrigin::DECODE;
    }
    TT_THROW("Fabric decode: unknown finding code {}", static_cast<int>(code));
}

std::vector<Finding> find_router_findings(
    const RouterValues& values,
    const manifest::Arch& arch,
    const std::optional<std::vector<VcDirections>>& directions) {
    std::vector<Finding> findings;
    add_capture_findings(values.get_snapshot(), findings);
    add_region_findings(values, findings);
    add_unknown_enumerator_findings(values, arch, findings);
    add_direction_findings(values.get_snapshot().get_router().path, directions, findings);
    return findings;
}

std::vector<Finding> find_pairing_findings(const RouterPairing& pairing) {
    std::vector<Finding> findings;
    for (const std::string& path : pairing.record_routers_not_in_manifest) {
        findings.push_back(make_finding(
            FindingCode::ROUTER_NOT_IN_MANIFEST, path, "the capture record has the router, but the manifest doesn't"));
    }
    return findings;
}

json make_finding_json(const Finding& finding) {
    json out;
    out["code"] = manifest::lower_enum_name(finding.code);
    out["severity"] = manifest::lower_enum_name(get_finding_severity(finding.code));
    out["origin"] = manifest::lower_enum_name(get_finding_origin(finding.code));
    out["paths"] = finding.paths;
    out["message"] = finding.message;
    return out;
}

}  // namespace tt::tt_fabric::decode
