// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Findings: what decode reports about a rank beside its values, each with a code from a closed set. A finding never
// aborts decode; it's written in the decoded file's top-level analysis.findings.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_inputs.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_router_analysis.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// What a finding is about. The decoded file writes it by lower_enum_name, and lists every code in
// analysis.vocabulary.finding_code.
enum class FindingCode : uint8_t {
    // -- What capture recorded. --

    // Capture read nothing of the router: no record entry, its chip missing or another architecture, or its core not
    // found. One per router.
    ROUTER_NOT_READ,
    // The full read's L1 read failed. One per router, not one per region.
    L1_READ_FAILED,
    // A register read of the full read failed, so the L1 and the registers after it weren't read. One per router.
    REGISTER_READ_FAILED,
    // A round's read failed, so the router's later rounds weren't read. Its full read still was.
    ROUND_READ_FAILED,
    // Capture read the manifest's core, though the reader puts the router's channel on another.
    CORE_MISMATCH,
    // The record has a router the manifest doesn't.
    ROUTER_NOT_IN_MANIFEST,

    // -- What decode found. --

    // The manifest names a stream register capture didn't read.
    REGISTER_NOT_CAPTURED,
    // A region isn't inside the L1 capture read.
    OUTSIDE_CAPTURED_L1,
    // A region's elements are smaller than its type, so its layout isn't this build's.
    TYPE_LARGER_THAN_ELEMENT,
    // An enum region holds a value that names no enumerator.
    UNKNOWN_ENUMERATOR,
    // A VC is connected to a sibling it never forwards to: expected on an E/W router that can't turn.
    CONNECTED_NOT_LIVE,
    // A VC forwards to a direction it has no sibling for: expected on a corner chip, since a correct
    // route never asks for it; a packet routed there blocks.
    LIVE_NOT_CONNECTED,
    // A chip's routers hold routing tables that differ beyond state_manager, the router's own. One per chip.
    ROUTING_TABLE_COPIES_DIFFER,
    // A routing table's mesh id, device id or, in 2D, coordinates aren't the manifest chip's. One per group of routers
    // with the same table.
    ROUTING_TABLE_IDENTITY_MISMATCH,
    // A 2D routing table's mesh shape isn't the manifest mesh's.
    ROUTING_TABLE_SHAPE_MISMATCH,
    // A 2D routing table's mesh shape can't be a 2D route table's, so its action vectors and trees aren't shown.
    ROUTING_TABLE_SHAPE_INVALID,
    // Chips of one mesh whose 2D action vectors or 1D path table differ from the mesh's first chip's. One per mesh.
    ROUTING_TABLE_MESH_MEMBERS_DIFFER,
};

// How much a finding matters. Written by lower_enum_name.
enum class FindingSeverity : uint8_t {
    // Something decode should have shown is missing or wrong.
    ERROR,
    // Shown, but may mislead.
    WARNING,
    // A fact worth knowing that isn't a problem by itself.
    INFO,
};

// Which stage the finding comes from. Written by lower_enum_name.
enum class FindingOrigin : uint8_t {
    CAPTURE,
    DECODE,
};

struct Finding {
    FindingCode code = FindingCode::ROUTER_NOT_READ;
    // What it's about: a router's path, such as "M0/C7/E0", or a region's, such as "M0/C7/E0/fields/erisc0/edm_status".
    std::vector<std::string> paths;
    // For a person: capture's error text where there is one, else what decode found.
    std::string message;
};

// Each code's severity and origin, so every finding of a code has the same ones.
FindingSeverity get_finding_severity(FindingCode code);
FindingOrigin get_finding_origin(FindingCode code);

// Everything decode finds about one router: what capture recorded of it, its unread regions, enum regions that name
// no enumerator, and its directions' differences.
std::vector<Finding> find_router_findings(
    const RouterValues& values,
    const manifest::Arch& arch,
    const std::optional<std::vector<VcDirections>>& directions);

// The record's routers the manifest doesn't have.
std::vector<Finding> find_pairing_findings(const RouterPairing& pairing);

json make_finding_json(const Finding& finding);

}  // namespace tt::tt_fabric::decode
