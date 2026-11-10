// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Routing-table analysis: every router core has its own copy of its chip's routing table. This groups each chip's
// copies by their bytes, so the chip's table is written once (analysis.routing on its node) with the routers that hold
// it, and finds where a copy, a chip's identity or its shape isn't what the manifest says.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_findings.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// A router's routing table, from its full read.
struct RoutingTableRead {
    // We interpret the bytes into this type, which is what the router kernel uses as well.
    routing_l1_info_t table;
    // Where the table's bytes start in the .bin.
    uint64_t at = 0;
};

// The router's routing table as its full read has it. Null when the router has no full read of it: it wasn't read, or
// the area wasn't.
std::optional<RoutingTableRead> find_routing_table_read(const RouterValues& values);

// One router's table, with what analysis needs to place it.
struct RouterRoutingTable {
    manifest::RouterRef ref;
    // Such as "M0/C7/E0".
    std::string path;
    const manifest::Router* router = nullptr;
    RoutingTableRead read;
};

// One group of a chip's routers whose tables are the same apart from state_manager, the router's own.
struct RoutingTableCopy {
    // Router paths that correspond to this copy, in the order they were given. The first one's read is kept.
    std::vector<std::string> routers;
    // The first router's read: its table, with its state_manager, and where its bytes are in the .bin.
    RoutingTableRead read;
    // The first router's, which the table's JSON is written in the context of.
    manifest::RouterRef read_router_ref;
    const manifest::Router* read_manifest_router = nullptr;
};

// One chip's copies. One is the normal case.
struct ChipRouting {
    FabricNodeId chip{MeshId{0}, 0};
    // In the order the routers were given.
    std::vector<RoutingTableCopy> copies;
};

struct RoutingAnalysis {
    // In the order the chips' first routers were given.
    std::vector<ChipRouting> chips;
    // The index into its chip's copies that holds each router's table, by router path. A router with no table isn't
    // in it. Example: "M0/C7/E0" -> 0, where 0 is an index into the chip's routing table copies vector.
    std::map<std::string, uint32_t, std::less<>> copy_by_router;
};

// Groups each chip's routers' tables, and adds the findings to `findings`.
RoutingAnalysis analyze_routing_tables(
    std::span<const RouterRoutingTable> tables, const manifest::ManifestFacts& facts, std::vector<Finding>& findings);

// A chip's analysis.routing: `{ "copies": [{ "routers", "read_router", "at", "table" }] }`, each `table` written by
// make_chip_routing_table_json.
json make_chip_routing_json(const ChipRouting& routing, const manifest::ManifestFacts& facts);

// A router's analysis.routing_table: `{ "copy": n }`, the index of its chip's copy that holds its table. Null when
// the router has none.
json make_router_routing_table_json(std::optional<uint32_t> copy);

}  // namespace tt::tt_fabric::decode
