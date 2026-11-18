// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SPEC-0012 § Verification 1, no device: the `router: harness` manifest's checks (check_manifest's harness form and
// rows H1 to H9), the plan's routing tables and per-link counts against the hand-worked numbers of § Design ("The
// mesh, and its numbers"), the region layout and the node's runtime arguments (§ The node).

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt_metal/tt_fabric/keraunos/vcluster/harness.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/vc_node_layout.hpp"

namespace tt::tt_fabric::vcluster {
namespace {

namespace fs = std::filesystem;
using vc_node::kNoPort;

// mmk 26ww45 as SPEC-0011 and -0012 describe it: CCE0 of tiles 0 to 3, 8 harts and 4 MiB a CCE, tiles 0-1 and 2-3
// wired, and the unreserved base 0xaca00 of SPEC-0012 H3.
PackageFacts mmk() { return {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, 8, 0x400000, {{0, 1}, {2, 3}}, 0xaca00}; }

// § Design's mesh: D0 tile 0 hart 0, D1 tile 0 hart 1, D2 tile 1 hart 0, D3 tile 1 hart 1; each device's local link
// is its port 0 and its noc link its port 1. Each region is 64 KiB, above the unreserved base, and disjoint on its
// CCE.
constexpr std::string_view kMesh = R"yaml(
version: 1
router: harness
devices:
  - name: D0
    fabric_chip: 0
    slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 0}]
    regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]
  - name: D1
    fabric_chip: 1
    slots: [{tile: 0, cce: 0, hart: 1, role: router, port: 0}]
    regions: [{tile: 0, cce: 0, base: 0xc0000, size: 0x10000}]
  - name: D2
    fabric_chip: 2
    slots: [{tile: 1, cce: 0, hart: 0, role: router, port: 0}]
    regions: [{tile: 1, cce: 0, base: 0xb0000, size: 0x10000}]
  - name: D3
    fabric_chip: 3
    slots: [{tile: 1, cce: 0, hart: 1, role: router, port: 0}]
    regions: [{tile: 1, cce: 0, base: 0xc0000, size: 0x10000}]
links:
  - {a: {device: D0, port: 0}, b: {device: D1, port: 0}, backend: local}
  - {a: {device: D2, port: 0}, b: {device: D3, port: 0}, backend: local}
  - {a: {device: D0, port: 1}, b: {device: D2, port: 1}, backend: noc}
  - {a: {device: D1, port: 1}, b: {device: D3, port: 1}, backend: noc}
)yaml";

// § Verification 3: two devices on harts 0 and 1 of tile 0's CCE0, one local link.
constexpr std::string_view kPair = R"yaml(
version: 1
router: harness
devices:
  - name: D0
    fabric_chip: 0
    slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 0}]
    regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]
  - name: D1
    fabric_chip: 1
    slots: [{tile: 0, cce: 0, hart: 1, role: router, port: 0}]
    regions: [{tile: 0, cce: 0, base: 0xc0000, size: 0x10000}]
links:
  - {a: {device: D0, port: 0}, b: {device: D1, port: 0}, backend: local}
)yaml";

// § Verification 3 and 4: 5 packets a flow, 256 bytes of payload.
constexpr RunShape kShape{5, 256};

std::string edited(std::string_view base, std::initializer_list<std::pair<std::string_view, std::string_view>> edits) {
    std::string text(base);
    for (const auto& [from, to] : edits) {
        const size_t at = text.find(from);
        if (at == std::string::npos || text.find(from, at + 1) != std::string::npos) {
            throw std::logic_error("edit: '" + std::string(from) + "' is not in the manifest exactly once");
        }
        text.replace(at, from.size(), to);
    }
    return text;
}

using Rows = std::set<uint32_t>;

Rows manifest_rows(const Manifest& m, const PackageFacts& facts = mmk()) {
    Rows rows;
    for (const auto& finding : check_manifest(m, facts)) {
        rows.insert(static_cast<uint32_t>(finding.check));
    }
    return rows;
}

// The H rows of `yaml`, which must load and pass check_manifest, so that an H row is the harness checker's alone.
Rows h_rows(std::string_view yaml, const RunShape& shape = kShape, const PackageFacts& facts = mmk()) {
    const Manifest m = parse_manifest(yaml);
    EXPECT_EQ(manifest_rows(m, facts), Rows{}) << "check_manifest refuses the manifest";
    Rows rows;
    for (const auto& finding : check_harness_manifest(m, facts, shape)) {
        rows.insert(static_cast<uint32_t>(finding.check));
    }
    return rows;
}

// --- Accepted manifests ------------------------------------------------------------------------------------------

TEST(VclusterHarness, MeshAndPairAccepted) {
    const Manifest mesh = parse_manifest(kMesh);
    EXPECT_EQ(mesh.router, RouterKind::Harness);
    EXPECT_EQ(h_rows(kMesh), Rows{});
    EXPECT_EQ(h_rows(kPair), Rows{});
}

// § Manifest: `router` absent, the manifest is SPEC-0011's.
TEST(VclusterHarness, RouterAbsentIsMetal) {
    const Manifest m = parse_manifest(edited(kPair, {{"router: harness\n", ""}}));
    EXPECT_EQ(m.router, RouterKind::Metal);
}

// --- check_manifest under router: harness ------------------------------------------------------------------------

// § Manifest: row 1 is reduced to cce 1. A router on hart 1 and a region that is part of a CCE are values of the
// harness, and the SPEC-0011 manifest reports them as row 1.
TEST(VclusterHarness, Row1ReducedToCce1) {
    const Manifest harness = parse_manifest(kPair);
    EXPECT_EQ(manifest_rows(harness), Rows{});
    const Manifest metal = parse_manifest(edited(kPair, {{"router: harness\n", ""}}));
    EXPECT_EQ(manifest_rows(metal), Rows{1});
    EXPECT_EQ(
        manifest_rows(parse_manifest(edited(
            kPair,
            {{"slots: [{tile: 0, cce: 0, hart: 1, role: router, port: 0}]",
              "slots: [{tile: 0, cce: 1, hart: 1, role: router, port: 0}]"},
             {"regions: [{tile: 0, cce: 0, base: 0xc0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 1, base: 0xc0000, size: 0x10000}]"}}))),
        Rows{1});
}

// § Manifest: row 7's port clauses are H6's. Link ends naming ports the router slots do not have pass row 7, as the
// mesh's port 1 ends do; a link naming no device, or one device at both ends, is still row 7.
TEST(VclusterHarness, Row7KeepsDeviceClauses) {
    EXPECT_EQ(manifest_rows(parse_manifest(kMesh)), Rows{});
    EXPECT_EQ(
        manifest_rows(parse_manifest(edited(kPair, {{"b: {device: D1, port: 0}", "b: {device: D9, port: 0}"}}))),
        Rows{7});
    EXPECT_EQ(
        manifest_rows(parse_manifest(edited(kPair, {{"b: {device: D1, port: 0}", "b: {device: D0, port: 1}"}}))),
        Rows{7});
}

// § Manifest: rows 6 and 13 as written. A device whose one slot is a worker on hart 2 has no router (row 13); its CCE
// has D0's router on hart 0, so row 6 holds nothing.
TEST(VclusterHarness, Rows6And13AsWritten) {
    EXPECT_EQ(
        manifest_rows(parse_manifest(edited(
            kPair,
            {{"slots: [{tile: 0, cce: 0, hart: 1, role: router, port: 0}]",
              "slots: [{tile: 0, cce: 0, hart: 2, role: worker}]"}}))),
        Rows{13});
}

// --- H1 to H9, each alone ----------------------------------------------------------------------------------------

// H1: a device with two slots (the second on its own hart and port, so no SPEC-0011 row holds); a slot's port not 0.
TEST(VclusterHarness, H1) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 0}]",
              "slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 0}, {tile: 0, cce: 0, hart: 2, role: router, "
              "port: 1}]"}})),
        Rows{1});
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 0}]",
              "slots: [{tile: 0, cce: 0, hart: 0, role: router, port: 1}]"}})),
        Rows{1});
}

// H2: a device with two regions on its CCE.
TEST(VclusterHarness, H2) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}, {tile: 0, cce: 0, base: 0xd0000, size: "
              "0x10000}]"}})),
        Rows{2});
}

// H3: a base below the unreserved base; a base, and a size, not a multiple of 64. Each region still fits the layout
// (H8) and overlaps no other (row 4). The base at exactly 0xaca00 passes.
TEST(VclusterHarness, H3) {
    for (const std::string_view region : {
             "regions: [{tile: 0, cce: 0, base: 0xa0000, size: 0x10000}]",
             "regions: [{tile: 0, cce: 0, base: 0xb0020, size: 0xffc0}]",
             "regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0xfff0}]",
         }) {
        SCOPED_TRACE(region);
        EXPECT_EQ(
            h_rows(edited(kMesh, {{"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]", region}})), Rows{3});
    }
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 0, base: 0xaca00, size: 0x10000}]"}})),
        Rows{});
}

// H4: an ethernet link, between tiles 0 and 1, which bring-up wired, so row 8 holds nothing.
TEST(VclusterHarness, H4) {
    EXPECT_EQ(
        h_rows(
            edited(kMesh, {{"b: {device: D2, port: 1}, backend: noc", "b: {device: D2, port: 1}, backend: ethernet"}})),
        Rows{4});
}

// H5: a local link whose ends are on two CCEs; a noc link whose ends are on one tile.
TEST(VclusterHarness, H5) {
    EXPECT_EQ(
        h_rows(edited(kMesh, {{"b: {device: D2, port: 1}, backend: noc", "b: {device: D2, port: 1}, backend: local"}})),
        Rows{5});
    EXPECT_EQ(
        h_rows(edited(kMesh, {{"b: {device: D1, port: 0}, backend: local", "b: {device: D1, port: 0}, backend: noc"}})),
        Rows{5});
}

// H6: a device's ports are not 0 to k-1 each named once (a gap; a port named twice).
TEST(VclusterHarness, H6PortsNotDense) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh, {{"{a: {device: D0, port: 1}, b: {device: D2", "{a: {device: D0, port: 2}, b: {device: D2"}})),
        Rows{6});
    EXPECT_EQ(
        h_rows(edited(
            kMesh, {{"{a: {device: D0, port: 1}, b: {device: D2", "{a: {device: D0, port: 0}, b: {device: D2"}})),
        Rows{6});
}

// The pair with `count` local links between D0 and D1, ports 0 to count-1 at each end.
std::string pair_with_links(uint32_t count) {
    std::string links = "links:\n";
    for (uint32_t p = 0; p < count; ++p) {
        links += "  - {a: {device: D0, port: " + std::to_string(p) + "}, b: {device: D1, port: " + std::to_string(p) +
                 "}, backend: local}\n";
    }
    return edited(
        kPair, {{"links:\n  - {a: {device: D0, port: 0}, b: {device: D1, port: 0}, backend: local}\n", links}});
}

// H6: k above 4, the degree of a node of a 2D mesh. 4 ports pass. The pair's regions fit 5 ports at 1 packet a flow
// of 16 bytes (H8 holds nothing).
TEST(VclusterHarness, H6TooManyPorts) {
    const RunShape small{1, 16};
    EXPECT_EQ(h_rows(pair_with_links(4), small), Rows{});
    EXPECT_EQ(h_rows(pair_with_links(5), small), Rows{6});
}

// 16 devices, harts 0 to 7 of tile 0 and of tile 1, each tile's eight in a ring of local links (ports 0 and 1), and
// noc links between hart h of the two tiles (port 2) for each h in `noc_harts`. A node's runtime-argument words are
// 8 + N + 3k: 30 with two ports, 33 with three.
std::string sixteen(const std::vector<uint32_t>& noc_harts) {
    std::string text = "version: 1\nrouter: harness\ndevices:\n";
    for (uint32_t tile = 0; tile < 2; ++tile) {
        for (uint32_t hart = 0; hart < 8; ++hart) {
            const uint32_t chip = tile * 8 + hart;
            text += "  - name: T" + std::to_string(tile) + "H" + std::to_string(hart) +
                    "\n    fabric_chip: " + std::to_string(chip) + "\n    slots: [{tile: " + std::to_string(tile) +
                    ", cce: 0, hart: " + std::to_string(hart) +
                    ", role: router, port: 0}]\n    regions: [{tile: " + std::to_string(tile) +
                    ", cce: 0, base: " + std::to_string(0xb0000 + hart * 0x10000) + ", size: 65536}]\n";
        }
    }
    text += "links:\n";
    for (uint32_t tile = 0; tile < 2; ++tile) {
        for (uint32_t hart = 0; hart < 8; ++hart) {
            text += "  - {a: {device: T" + std::to_string(tile) + "H" + std::to_string(hart) +
                    ", port: 0}, b: {device: T" + std::to_string(tile) + "H" + std::to_string((hart + 1) % 8) +
                    ", port: 1}, backend: local}\n";
        }
    }
    for (const uint32_t hart : noc_harts) {
        text += "  - {a: {device: T0H" + std::to_string(hart) + ", port: 2}, b: {device: T1H" + std::to_string(hart) +
                ", port: 2}, backend: noc}\n";
    }
    return text;
}

// H6: the nodes of one CCE need more than 256 runtime-argument words. With a noc link at every hart each CCE needs
// 8 x 33 = 264; with one, 33 + 7 x 30 = 243, which passes. 1 packet a flow of 16 bytes keeps the layout inside the
// 64 KiB regions (H8).
TEST(VclusterHarness, H6ArgumentWordsPerCce) {
    const RunShape small{1, 16};
    EXPECT_EQ(vc_node::num_args(16, 3), 33u);
    EXPECT_EQ(vc_node::num_args(16, 2), 30u);
    EXPECT_EQ(h_rows(sixteen({0}), small), Rows{});
    EXPECT_EQ(h_rows(sixteen({0, 1, 2, 3, 4, 5, 6, 7}), small), Rows{6});
}

// H7: the mesh without its noc links is two islands.
TEST(VclusterHarness, H7) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"  - {a: {device: D0, port: 1}, b: {device: D2, port: 1}, backend: noc}\n"
              "  - {a: {device: D1, port: 1}, b: {device: D3, port: 1}, backend: noc}\n",
              ""}})),
        Rows{7});
}

// H8: a region smaller than the node's layout for the run; the mesh's layout is 38976 bytes, which passes.
TEST(VclusterHarness, H8) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 0, base: 0xb0000, size: 38912}]"}})),
        Rows{8});
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 0, base: 0xb0000, size: 38976}]"}})),
        Rows{});
}

// H9 (SPEC-0012 at docs 0b75d2a): fabric_chip values not 0 to N-1 with none missing.
TEST(VclusterHarness, H9) { EXPECT_EQ(h_rows(edited(kMesh, {{"fabric_chip: 3", "fabric_chip: 4"}})), Rows{9}); }

// Every H row that holds is reported: H4 and H8 together.
TEST(VclusterHarness, EveryHRowReported) {
    EXPECT_EQ(
        h_rows(edited(
            kMesh,
            {{"b: {device: D2, port: 1}, backend: noc", "b: {device: D2, port: 1}, backend: ethernet"},
             {"regions: [{tile: 0, cce: 0, base: 0xb0000, size: 0x10000}]",
              "regions: [{tile: 0, cce: 0, base: 0xb0000, size: 38912}]"}})),
        (Rows{4, 8}));
}

// --- The plan ----------------------------------------------------------------------------------------------------

// § Design: next_port is the first port on a shortest path, lowest port first, so a packet for the opposite corner
// leaves on port 0: D0 to D3 by D1, D1 to D2 by D0, D2 to D1 by D3, D3 to D0 by D2.
TEST(VclusterHarness, MeshRoutingTables) {
    const HarnessPlan plan = plan_harness(parse_manifest(kMesh), kShape);
    ASSERT_EQ(plan.nodes.size(), 4u);
    const std::vector<std::vector<uint32_t>> expected = {
        {kNoPort, 0, 1, 0}, {0, kNoPort, 0, 1}, {1, 0, kNoPort, 0}, {0, 1, 0, kNoPort}};
    for (uint32_t d = 0; d < 4; ++d) {
        EXPECT_EQ(plan.nodes[d].next_port, expected[d]) << "D" << d;
    }
}

// § Design: with 5 packets a flow each of the 8 directed links carries 10, every port's sent and received is 10;
// ring_slots is 60.
TEST(VclusterHarness, MeshCounts) {
    const HarnessPlan plan = plan_harness(parse_manifest(kMesh), kShape);
    EXPECT_EQ(plan.ring_slots, 60u);
    uint64_t total = 0;
    for (const auto& node : plan.nodes) {
        ASSERT_EQ(node.predicted_sent.size(), 2u) << node.name;
        ASSERT_EQ(node.predicted_received.size(), 2u) << node.name;
        for (uint32_t p = 0; p < 2; ++p) {
            EXPECT_EQ(node.predicted_sent[p], 10u) << node.name << " port " << p;
            EXPECT_EQ(node.predicted_received[p], 10u) << node.name << " port " << p;
            total += node.predicted_sent[p];
        }
    }
    EXPECT_EQ(total, 80u);
}

// The mesh's ports: each device's port 0 is its local link and port 1 its noc link, to the peer port the link's other
// end names; the node sits on its slot's tile and hart.
TEST(VclusterHarness, MeshPorts) {
    const HarnessPlan plan = plan_harness(parse_manifest(kMesh), kShape);
    const uint32_t local_peer[] = {1, 0, 3, 2};
    const uint32_t noc_peer[] = {2, 3, 0, 1};
    for (uint32_t d = 0; d < 4; ++d) {
        const PlannedNode& node = plan.nodes[d];
        EXPECT_EQ(node.device, d);
        EXPECT_EQ(node.tile, d / 2) << node.name;
        EXPECT_EQ(node.hart, d % 2) << node.name;
        ASSERT_EQ(node.ports.size(), 2u) << node.name;
        EXPECT_EQ(node.ports[0].backend, LinkBackend::Local) << node.name;
        EXPECT_EQ(node.ports[0].peer_device, local_peer[d]) << node.name;
        EXPECT_EQ(node.ports[0].peer_port, 0u) << node.name;
        EXPECT_EQ(node.ports[1].backend, LinkBackend::Noc) << node.name;
        EXPECT_EQ(node.ports[1].peer_device, noc_peer[d]) << node.name;
        EXPECT_EQ(node.ports[1].peer_port, 1u) << node.name;
    }
}

// § Design: "Another port numbering changes the transit paths and gives 15 and 5 on some links." D0 with its noc link
// as port 0 and its local link as port 1: D0 to D3 now goes by D2 and D1 to D2 by D0 (lowest port first), so D0 to D2
// and D2 to D3 carry 15, D0 to D1 and D1 to D3 carry 5, and the other four 10. Worked from the routing rule, not
// quoted from the spec.
TEST(VclusterHarness, OtherNumberingCounts) {
    const HarnessPlan plan = plan_harness(
        parse_manifest(edited(
            kMesh,
            {{"{a: {device: D0, port: 0}, b: {device: D1", "{a: {device: D0, port: 1}, b: {device: D1"},
             {"{a: {device: D0, port: 1}, b: {device: D2", "{a: {device: D0, port: 0}, b: {device: D2"}})),
        kShape);
    // sent per [device][port]; D0's port 0 is now its noc link to D2, port 1 its local link to D1.
    const std::vector<std::vector<uint64_t>> sent = {{15, 5}, {10, 5}, {15, 10}, {10, 10}};
    const std::vector<std::vector<uint64_t>> received = {{10, 10}, {5, 10}, {10, 15}, {15, 5}};
    for (uint32_t d = 0; d < 4; ++d) {
        EXPECT_EQ(plan.nodes[d].predicted_sent, sent[d]) << "D" << d;
        EXPECT_EQ(plan.nodes[d].predicted_received, received[d]) << "D" << d;
    }
}

// § Design, the pair: each port's sent and received is 5; ring_slots is 10.
TEST(VclusterHarness, PairPlan) {
    const HarnessPlan plan = plan_harness(parse_manifest(kPair), kShape);
    EXPECT_EQ(plan.ring_slots, 10u);
    ASSERT_EQ(plan.nodes.size(), 2u);
    EXPECT_EQ(plan.nodes[0].next_port, (std::vector<uint32_t>{kNoPort, 0}));
    EXPECT_EQ(plan.nodes[1].next_port, (std::vector<uint32_t>{0, kNoPort}));
    for (const auto& node : plan.nodes) {
        EXPECT_EQ(node.predicted_sent, (std::vector<uint64_t>{5})) << node.name;
        EXPECT_EQ(node.predicted_received, (std::vector<uint64_t>{5})) << node.name;
    }
}

// --- Layout and runtime arguments --------------------------------------------------------------------------------

// § Region layout and § Design: a slot is 320 bytes for 256 of payload, a port block 19264, the mesh's region 38976.
// The pair's 3712 is worked from the same table (64 + 64 + (64 + 10 x 320) + 320), not quoted.
TEST(VclusterHarness, Layout) {
    const vc_node::Layout mesh = vc_node::layout(2, 60, 256);
    EXPECT_EQ(mesh.slot_bytes, 320u);
    EXPECT_EQ(mesh.port_block_bytes, 19264u);
    EXPECT_EQ(mesh.control, 0u);
    EXPECT_EQ(mesh.staging, 64u);
    EXPECT_EQ(mesh.port_blocks, 128u);
    EXPECT_EQ(mesh.port_block(1), 128u + 19264u);
    EXPECT_EQ(mesh.ring_slot(1, 2), 128u + 19264u + 64u + 2u * 320u);
    EXPECT_EQ(mesh.staging_word(3), 64u + 24u);
    EXPECT_EQ(mesh.source, 128u + 2u * 19264u);
    EXPECT_EQ(mesh.total, 38976u);
    EXPECT_EQ(vc_node::layout(1, 10, 256).total, 3712u);
    EXPECT_EQ(vc_node::layout(1, 1, 0).slot_bytes, 64u);  // a header alone rounds up to one 64-byte block

    const HarnessPlan plan = plan_harness(parse_manifest(kMesh), kShape);
    for (const auto& node : plan.nodes) {
        EXPECT_EQ(node.layout.total, 38976u) << node.name;
    }
}

// § The node, runtime arguments: 8 words, next_port for each device, then kind, peer_tile and peer_port_base a port.
// D0 of the mesh: port 0 is local to D1 (port 0 of D1's region at 0xc0000), port 1 noc to D2 on tile 1 (port 1 of
// D2's region at 0xb0000).
TEST(VclusterHarness, RuntimeArgs) {
    const HarnessPlan plan = plan_harness(parse_manifest(kMesh), kShape);
    const uint32_t nonce = 0x5a17c0de;
    const std::vector<uint32_t> expected = {
        0,
        4,
        0xb0000,
        2,
        5,
        256,
        60,
        nonce,  // device, num_devices, region_base, num_ports, ppf, payload, ring, nonce
        kNoPort,
        0,
        1,
        0,  // next_port
        vc_node::kPortKindLocal,
        0,
        0xc0000 + 128,  // port 0
        vc_node::kPortKindNoc,
        1,
        0xb0000 + 128 + 19264};  // port 1
    EXPECT_EQ(node_runtime_args(plan, 0, nonce), expected);
    EXPECT_EQ(expected.size(), vc_node::num_args(4, 2));
    const auto d3 = node_runtime_args(plan, 3, nonce);
    ASSERT_EQ(d3.size(), vc_node::num_args(4, 2));
    EXPECT_EQ(d3[vc_node::kArgDevice], 3u);
    EXPECT_EQ(d3[vc_node::kArgRegionBase], 0xc0000u);
    // D3's port 1 is its noc link to D1, port 1 of D1's region on tile 0.
    const uint32_t port1 = vc_node::kArgNextPortBase + 4 + vc_node::kArgsPerPort;
    EXPECT_EQ(d3[port1 + vc_node::kPortArgKind], vc_node::kPortKindNoc);
    EXPECT_EQ(d3[port1 + vc_node::kPortArgPeerTile], 0u);
    EXPECT_EQ(d3[port1 + vc_node::kPortArgPeerPortBase], 0xc0000u + 128 + 19264);
}

// § Verification 3 and 4 name the manifests the emulator runs; they are the pair and the mesh of § Design.
TEST(VclusterHarness, ManifestFiles) {
    const fs::path dir = fs::path(__FILE__).parent_path() / "manifests";
    for (const auto& [name, nodes] : {std::pair{"pair_local.yaml", 2u}, std::pair{"mesh2x2.yaml", 4u}}) {
        SCOPED_TRACE(name);
        const Manifest m = load_manifest(dir / name);
        EXPECT_EQ(manifest_rows(m), Rows{});
        EXPECT_TRUE(check_harness_manifest(m, mmk(), kShape).empty());
        const HarnessPlan plan = plan_harness(m, kShape);
        ASSERT_EQ(plan.nodes.size(), nodes);
        for (const auto& node : plan.nodes) {
            for (size_t p = 0; p < node.predicted_sent.size(); ++p) {
                EXPECT_EQ(node.predicted_sent[p], nodes == 2 ? 5u : 10u) << node.name;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric::vcluster
