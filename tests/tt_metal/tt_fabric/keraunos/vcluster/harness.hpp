// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Harness routers on devices that share a CCE (SPEC-0012): the checks a `router: harness` manifest must pass, and the
// plan the harness computes from it (routing tables, layouts, runtime arguments, predicted per-link counts). No
// device access; run_harness() in harness_run.hpp touches the device.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/vc_node_layout.hpp"

namespace tt::tt_fabric::vcluster {

// The row of SPEC-0012's H table a finding belongs to; the value is the row number.
enum class HarnessCheck : uint32_t {
    SlotNotOne = 1,        // H1: a device has not exactly one slot, or its slot's port is not 0
    RegionNotOne = 2,      // H2: a device has not exactly one region
    RegionPlacement = 3,   // H3: a region's base is below PackageFacts::unreserved_base, or base or size is not a
                           // multiple of vc_node::kAlign
    EthernetLink = 4,      // H4: a link's backend is ethernet
    LinkPlacement = 5,     // H5: a local link's ends are not on one CCE, or a noc link's ends are on one tile
    Ports = 6,             // H6: a device's ports are not 0..k-1 each named once, k is above vc_node::kMaxPorts, or
                           // the nodes of one CCE need more than kMaxArgWordsPerCce runtime-argument words
    Disconnected = 7,      // H7: some device cannot reach another over the links
    RegionTooSmall = 8,    // H8: a region is smaller than the node's layout for this run
    FabricChipNotDense = 9,  // H9: the devices' fabric_chip values are not 0 to N-1 with none missing
};

// One failing instance of an H row. A manifest can produce several findings for one row.
struct HarnessFinding {
    HarnessCheck check;
    std::string detail;  // names the device, link or CCE and the offending value; for people, not parsed
};

// What one run sends: packets_per_flow packets from every device to every other, each with payload_bytes of payload.
struct RunShape {
    uint32_t packets_per_flow = 0;  // at least 1
    uint32_t payload_bytes = 0;     // a multiple of vc_node::kPayloadGranule
};

// H6's bound on the runtime-argument words of the nodes of one CCE: half of MEM_IERISC_KERNEL_CONFIG_SIZE, in words.
constexpr uint32_t kMaxArgWordsPerCce = 256;

// Runs SPEC-0012's H rows against `manifest`, which must have `router: harness` and must have passed
// check_manifest(`manifest`, `facts`) with no finding. Returns every failing instance of every H row, in row order;
// empty means the manifest can run with `shape`. Reads no device. Throws std::runtime_error (TT_FATAL) only on a
// caller bug: a manifest without `router: harness` or with no devices; `facts` with no cores, 0 harts, 0 SRAM bytes
// or more than 2^32, or a 0 unreserved_base; `shape` with 0 packets_per_flow or a payload_bytes that is not a multiple of
// vc_node::kPayloadGranule; or a manifest that check_manifest() would have refused in a way the rows depend on (a
// device with no router slot, a slot or region with no matching region or slot on its CCE, a link naming a device
// that does not exist).
std::vector<HarnessFinding> check_harness_manifest(
    const Manifest& manifest, const PackageFacts& facts, const RunShape& shape);

// What one link end of a node connects to.
struct PlannedPort {
    LinkBackend backend = LinkBackend::Noc;  // Noc or Local
    uint32_t peer_device = 0;                // the peer's fabric_chip
    uint32_t peer_port = 0;                  // the port of the peer this link is
};

// One node: one device of the manifest, on its one slot's hart.
struct PlannedNode {
    std::string name;
    uint32_t device = 0;  // fabric_chip
    uint32_t tile = 0;
    uint32_t hart = 0;
    uint64_t region_base = 0;
    std::vector<PlannedPort> ports;         // index = port number
    std::vector<uint32_t> next_port;        // index = fabric_chip; vc_node::kNoPort for this device
    std::vector<uint64_t> predicted_sent;   // index = port: packets this node sends on it in the run
    std::vector<uint64_t> predicted_received;  // index = port: packets this node takes from its ring in the run
    vc_node::Layout layout{};
};

struct HarnessPlan {
    RunShape shape;
    uint32_t ring_slots = 0;            // packets_per_flow * N * (N - 1)
    std::vector<PlannedNode> nodes;     // index = fabric_chip
};

// The plan for running `manifest` with `shape`. next_port[d] is the first port on a shortest path to d, by
// breadth-first search over the links, ties broken by the lowest port; the predicted counts follow every flow along
// those routes. Precondition: check_manifest() and check_harness_manifest() reported nothing for these arguments. Throws std::runtime_error (TT_FATAL) when a precondition the plan
// depends on does not hold.
HarnessPlan plan_harness(const Manifest& manifest, const RunShape& shape);

// The runtime arguments of the node of fabric_chip `device`, in vc_node::Arg order, for a run tagged `nonce`. Throws
// std::runtime_error (TT_FATAL) when `device` is not a node of `plan`.
std::vector<uint32_t> node_runtime_args(const HarnessPlan& plan, uint32_t device, uint32_t nonce);

}  // namespace tt::tt_fabric::vcluster
