// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The virtual-cluster manifest's loader and its checks before the device is touched, from SPEC-0011 § Manifest and
// § Checks before the device is touched, and its § Verification 1: one accepted manifest, one manifest
// per row that trips that row and no other, one per kind of load error. No device. Rows 9 to 11 need a capture and
// are not covered.
//
// Every rejected manifest is the accepted line with one edit, so the edit is the only thing that differs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_metal/tt_fabric/keraunos/vcluster/manifest.hpp"

namespace tt::tt_fabric::vcluster {
namespace {

constexpr uint64_t kSram = 0x400000;

// SPEC-0011 § Contract: the Phase 2 line, A on tile 0, M on tiles 1 and 2, B on tile 3, one link each backend. A to M
// is on tiles 0 and 1, M to B on tiles 2 and 3: both pairs are wired, so either backend is allowed on either link.
constexpr std::string_view kLine = R"yaml(
version: 1
devices:
  - name: A
    fabric_chip: 0
    slots:
      - {tile: 0, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 0, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}
  - name: M
    fabric_chip: 1
    slots:
      - {tile: 1, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 2, cce: 0, hart: 0, role: router, port: 1}
    regions:
      - {tile: 1, cce: 0, base: 0x0, size: 0x400000}
      - {tile: 2, cce: 0, base: 0x0, size: 0x400000}
  - name: B
    fabric_chip: 2
    slots:
      - {tile: 3, cce: 0, hart: 0, role: router, port: 0}
      - {tile: 3, cce: 0, hart: 1, role: worker}
    regions:
      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}
links:
  - {a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}
  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}
)yaml";

// mmk 26ww45 as SPEC-0011 describes it: one CCE0 on each of tiles 0 to 3, tiles 0-1 and 2-3 wired (row 8). The hart
// count is the test's choice; the rows read it from here.
PackageFacts mmk() { return {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, 4, kSram, {{0, 1}, {2, 3}}}; }

// mmk() with a fifth CCE0 on tile 4, for a CCE no other device uses.
PackageFacts mmk_with_tile_4() {
    PackageFacts facts = mmk();
    facts.cores.push_back({4, 0});
    return facts;
}

// kLine with each `from` replaced by its `to`. Each `from` must occur exactly once, so an edit cannot land elsewhere.
std::string edit(std::initializer_list<std::pair<std::string_view, std::string_view>> edits) {
    std::string text(kLine);
    for (const auto& [from, to] : edits) {
        const size_t at = text.find(from);
        if (at == std::string::npos || text.find(from, at + 1) != std::string::npos) {
            throw std::logic_error("edit: '" + std::string(from) + "' is not in the line exactly once");
        }
        text.replace(at, from.size(), to);
    }
    return text;
}

// SPEC-0011 § Checks: "The loader fails, listing every load error it found, and reports no row." The loader throws
// ManifestLoadError. Returns its errors; fails the test when `yaml` loads.
std::vector<std::string> load_errors(std::string_view yaml) {
    try {
        parse_manifest(yaml);
    } catch (const ManifestLoadError& error) {
        EXPECT_FALSE(error.errors().empty());
        return error.errors();
    }
    ADD_FAILURE() << "loaded, expected a load error";
    return {};
}

std::string joined(const std::vector<std::string>& errors) {
    std::string text;
    for (const auto& error : errors) {
        text += "\n  " + error;
    }
    return text;
}

// `yaml` fails to load with exactly one error, which names `needle`: the key or value the one edit broke. A needle is
// a key or a value, not wording, so the check survives a reworded message.
void expect_load_error(std::string_view yaml, std::string_view needle) {
    const auto errors = load_errors(yaml);
    ASSERT_EQ(errors.size(), 1u) << joined(errors);
    EXPECT_NE(errors[0].find(needle), std::string::npos) << "no '" << needle << "' in" << joined(errors);
}

Manifest load(std::string_view yaml) {
    try {
        return parse_manifest(yaml);
    } catch (const ManifestLoadError& error) {
        ADD_FAILURE() << "load failed: " << error.what();
        throw;
    }
}

// The findings check_manifest() reports for `yaml`, which must load.
std::vector<ManifestFinding> findings(std::string_view yaml, const PackageFacts& facts = mmk()) {
    return check_manifest(load(yaml), facts);
}

// The rows check_manifest() reports for `yaml`.
std::set<uint32_t> rows(std::string_view yaml, const PackageFacts& facts = mmk()) {
    std::set<uint32_t> found;
    for (const auto& finding : findings(yaml, facts)) {
        found.insert(static_cast<uint32_t>(finding.check));
    }
    return found;
}

size_t count_of(const std::vector<ManifestFinding>& found, ManifestCheck row) {
    return std::count_if(found.begin(), found.end(), [&](const ManifestFinding& f) { return f.check == row; });
}

using Rows = std::set<uint32_t>;

// --- Accepted ----------------------------------------------------------------------------------------------------

// § Verification 1: the line loads as written and passes every row.
TEST(VclusterManifest, LineLoadsAsWritten) {
    const Manifest m = load(kLine);
    EXPECT_EQ(m.version, 1u);
    EXPECT_FALSE(m.capture.has_value());
    ASSERT_EQ(m.devices.size(), 3u);
    EXPECT_EQ(m.devices[0].name, "A");
    EXPECT_EQ(m.devices[1].name, "M");
    EXPECT_EQ(m.devices[2].fabric_chip, 2u);
    ASSERT_EQ(m.devices[1].slots.size(), 2u);
    const Slot& m_tile2 = m.devices[1].slots[1];
    EXPECT_EQ(m_tile2.tile, 2u);
    EXPECT_EQ(m_tile2.hart, 0u);
    EXPECT_EQ(m_tile2.role, SlotRole::Router);
    EXPECT_EQ(m_tile2.port, std::optional<uint32_t>(1));
    EXPECT_EQ(m.devices[0].slots[1].role, SlotRole::Worker);
    EXPECT_FALSE(m.devices[0].slots[1].port.has_value());
    ASSERT_EQ(m.devices[1].regions.size(), 2u);
    EXPECT_EQ(m.devices[1].regions[1].base, 0u);
    EXPECT_EQ(m.devices[1].regions[1].size, kSram);  // 0x400000 read as hex
    ASSERT_EQ(m.links.size(), 2u);
    EXPECT_EQ(m.links[0].backend, LinkBackend::Noc);
    EXPECT_EQ(m.links[1].backend, LinkBackend::Ethernet);
    EXPECT_EQ(m.links[1].a.device, "M");
    EXPECT_EQ(m.links[1].a.port, 1u);
    EXPECT_EQ(m.links[1].b.device, "B");
}

TEST(VclusterManifest, LinePasses) { EXPECT_EQ(rows(kLine), Rows{}); }

// § Checks: "`capture` is optional." Named, it loads; rows 9 to 11 are not judged here.
TEST(VclusterManifest, CaptureNamedLoads) {
    const Manifest m = load(edit({{"version: 1\n", "version: 1\ncapture: line3.capture\n"}}));
    EXPECT_EQ(m.capture, std::filesystem::path("line3.capture"));
}

// The loader's contract: load_manifest() reads the file and resolves a relative `capture` against its directory.
TEST(VclusterManifest, LoadManifestFromFile) {
    const auto dir = std::filesystem::path(::testing::TempDir()) / "vcluster_manifest_test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "line3.yaml";
    std::ofstream(path) << edit({{"version: 1\n", "version: 1\ncapture: line3.capture\n"}});
    const Manifest m = load_manifest(path);
    EXPECT_EQ(m.capture, dir / "line3.capture");
    EXPECT_EQ(m.devices.size(), 3u);
    std::filesystem::remove_all(dir);
}

// An absolute `capture` is kept as written.
TEST(VclusterManifest, LoadManifestAbsoluteCapture) {
    const auto dir = std::filesystem::path(::testing::TempDir()) / "vcluster_manifest_abs_test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "line3.yaml";
    std::ofstream(path) << edit({{"version: 1\n", "version: 1\ncapture: /captures/line3.capture\n"}});
    const Manifest m = load_manifest(path);
    EXPECT_EQ(m.capture, std::filesystem::path("/captures/line3.capture"));
    std::filesystem::remove_all(dir);
}

// Row 8 judges only ethernet links: both links on the NOC pass, and so does a NOC link between unwired tiles.
TEST(VclusterManifest, NocLinksPass) {
    EXPECT_EQ(rows(edit({{"backend: ethernet", "backend: noc"}})), Rows{});
    EXPECT_EQ(
        rows(edit(
            {{"b: {device: M, port: 0}, backend: noc", "b: {device: M, port: 1}, backend: noc"},
             {"{a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}",
              "{a: {device: M, port: 0}, b: {device: B, port: 0}, backend: noc}"}})),
        Rows{});
}

// § Checks: "A region on a CCE where its device has no slot is allowed."
TEST(VclusterManifest, RegionWithoutSlotPasses) {
    const std::string yaml = edit(
        {{"      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n",
          "      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n      - {tile: 4, cce: 0, base: 0x0, size: "
          "0x400000}\n"}});
    EXPECT_EQ(rows(yaml, mmk_with_tile_4()), Rows{});
}

// --- Stage A: row 1, reported alone ------------------------------------------------------------------------------

// Row 1: `cce` 1. (0, 1) is no core either (row 2), and its worker has no router on hart 0 (row 6): stage A ends the
// check, so only row 1 is reported.
TEST(VclusterManifest, Row1Cce1) {
    EXPECT_EQ(
        rows(edit({{"{tile: 0, cce: 0, hart: 1, role: worker}", "{tile: 0, cce: 1, hart: 1, role: worker}"}})),
        Rows{1});
}

// Row 1: a router on a hart other than 0. A's worker then has no router on hart 0 (row 6); stage A ends the check.
TEST(VclusterManifest, Row1RouterOffHart0) {
    EXPECT_EQ(
        rows(edit(
            {{"{tile: 0, cce: 0, hart: 0, role: router, port: 0}",
              "{tile: 0, cce: 0, hart: 2, role: router, port: 0}"}})),
        Rows{1});
}

// Row 1: a region inside its CCE's SRAM that is not the whole of it, at base 0, at an offset, and of size 0.
TEST(VclusterManifest, Row1PartialRegion) {
    for (const std::string_view region : {
             "{tile: 0, cce: 0, base: 0x0, size: 0x200000}",
             "{tile: 0, cce: 0, base: 0x100000, size: 0x100000}",
             "{tile: 0, cce: 0, base: 0x0, size: 0x0}",
             "{tile: 0, cce: 0, base: 0x400000, size: 0x0}",
         }) {
        EXPECT_EQ(rows(edit({{"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", region}})), Rows{1}) << region;
    }
}

// Row 1: backend `local`.
TEST(VclusterManifest, Row1BackendLocal) { EXPECT_EQ(rows(edit({{"backend: noc", "backend: local"}})), Rows{1}); }

// --- Stage B: rows 2 and 12 --------------------------------------------------------------------------------------

// Row 2: a slot's (tile, cce) is not a core. The slot also has no region (row 5) and no router (row 6), which are
// stage C and not judged.
TEST(VclusterManifest, Row2SlotNotACore) {
    EXPECT_EQ(
        rows(edit({{"{tile: 0, cce: 0, hart: 1, role: worker}", "{tile: 5, cce: 0, hart: 1, role: worker}"}})),
        Rows{2});
}

// Row 2: `cce` 2 is not reserved for Phase 3, so it is row 2, not row 1.
TEST(VclusterManifest, Row2Cce2) {
    EXPECT_EQ(
        rows(edit({{"{tile: 0, cce: 0, hart: 1, role: worker}", "{tile: 0, cce: 2, hart: 1, role: worker}"}})),
        Rows{2});
}

// Row 2: a region's (tile, cce) is not a core, and nothing else (§ Checks: row 2 only).
TEST(VclusterManifest, Row2RegionNotACore) {
    EXPECT_EQ(
        rows(edit(
            {{"      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n",
              "      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n      - {tile: 5, cce: 0, base: 0x0, size: "
              "0x400000}\n"}})),
        Rows{2});
}

// Row 12: two devices share a name. Links to M are then ambiguous (row 7, stage C, not judged).
TEST(VclusterManifest, Row12DuplicateName) {
    EXPECT_EQ(rows(edit({{"name: B", "name: M"}, {"b: {device: B, port: 0}", "b: {device: M, port: 0}"}})), Rows{12});
}

TEST(VclusterManifest, Row12DuplicateFabricChip) {
    EXPECT_EQ(rows(edit({{"fabric_chip: 2", "fabric_chip: 1"}})), Rows{12});
}

// Row 12: two routers of one device share a port. M port 1 then names nothing (row 7, stage C, not judged).
TEST(VclusterManifest, Row12DuplicatePort) {
    EXPECT_EQ(
        rows(edit(
            {{"{tile: 2, cce: 0, hart: 0, role: router, port: 1}",
              "{tile: 2, cce: 0, hart: 0, role: router, port: 0}"}})),
        Rows{12});
}

// Rows 2 and 12 in one stage: both reported.
TEST(VclusterManifest, Rows2And12Together) {
    EXPECT_EQ(
        rows(edit(
            {{"fabric_chip: 2", "fabric_chip: 1"},
             {"{tile: 0, cce: 0, hart: 1, role: worker}", "{tile: 5, cce: 0, hart: 1, role: worker}"}})),
        (Rows{2, 12}));
}

// Stage A ends the check before stage B: row 1 with a row-12 duplicate reports row 1 alone.
TEST(VclusterManifest, Row1EndsBeforeStageB) {
    EXPECT_EQ(rows(edit({{"fabric_chip: 2", "fabric_chip: 1"}, {"backend: noc", "backend: local"}})), Rows{1});
}

// --- Stage C: rows 3 to 8 and 13 ---------------------------------------------------------------------------------

// Row 3: two slots name one (tile, cce, hart), "reported once for the second and each later slot": two slots on a
// hart give one finding, three give two.
TEST(VclusterManifest, Row3SharedHart) {
    const std::string two = edit(
        {{"      - {tile: 3, cce: 0, hart: 1, role: worker}\n",
          "      - {tile: 3, cce: 0, hart: 1, role: worker}\n      - {tile: 3, cce: 0, hart: 1, role: worker}\n"}});
    EXPECT_EQ(rows(two), Rows{3});
    EXPECT_EQ(count_of(findings(two), ManifestCheck::Slot), 1u);
    const std::string three = edit(
        {{"      - {tile: 3, cce: 0, hart: 1, role: worker}\n",
          "      - {tile: 3, cce: 0, hart: 1, role: worker}\n      - {tile: 3, cce: 0, hart: 1, role: worker}\n"
          "      - {tile: 3, cce: 0, hart: 1, role: worker}\n"}});
    EXPECT_EQ(rows(three), Rows{3});
    EXPECT_EQ(count_of(findings(three), ManifestCheck::Slot), 2u);
}

// Row 3: a hart not below the CCE's hart count (4 in mmk()); hart 3 is the last that passes.
TEST(VclusterManifest, Row3HartBeyondCount) {
    EXPECT_EQ(
        rows(edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 4, role: worker}"}})),
        Rows{3});
    EXPECT_EQ(
        rows(edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 3, role: worker}"}})), Rows{});
}

// Row 4: two regions overlap, here B owning tile 0's CCE as A does. B has no slot there, which is allowed.
TEST(VclusterManifest, Row4Overlap) {
    EXPECT_EQ(
        rows(edit(
            {{"      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}\n",
              "      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}\n      - {tile: 0, cce: 0, base: 0x0, size: "
              "0x400000}\n"}})),
        Rows{4});
}

// Row 4: `base + size` beyond the CCE's SRAM, from base 0 and from an offset.
TEST(VclusterManifest, Row4BeyondSram) {
    for (const std::string_view region : {
             "{tile: 0, cce: 0, base: 0x0, size: 0x400001}",
             "{tile: 0, cce: 0, base: 0x200000, size: 0x400000}",
             "{tile: 0, cce: 0, base: 0x400000, size: 0x1}",
         }) {
        EXPECT_EQ(rows(edit({{"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", region}})), Rows{4}) << region;
    }
}

// Row 5: a slot on a CCE its device owns no region of; and § Checks: "A device with a router and no region fails row
// 5."
TEST(VclusterManifest, Row5SlotWithoutRegion) {
    EXPECT_EQ(rows(edit({{"      - {tile: 2, cce: 0, base: 0x0, size: 0x400000}\n", ""}})), Rows{5});
    EXPECT_EQ(
        rows(edit({{"    regions:\n      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}\n", "    regions: []\n"}})),
        Rows{5});
}

// Row 6: a CCE with a worker and no router on hart 0; and a worker on hart 0. Tile 4 is a CCE of A's with nothing
// else on it, so no other row is touched.
TEST(VclusterManifest, Row6WorkerWithoutRouter) {
    const auto with_tile_4_worker = [](std::string_view hart) {
        const std::string slot = "      - {tile: 4, cce: 0, hart: " + std::string(hart) + ", role: worker}\n";
        const std::string slots = "      - {tile: 0, cce: 0, hart: 1, role: worker}\n" + slot;
        return edit(
            {{"      - {tile: 0, cce: 0, hart: 1, role: worker}\n", slots},
             {"      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n",
              "      - {tile: 0, cce: 0, base: 0x0, size: 0x400000}\n      - {tile: 4, cce: 0, base: 0x0, size: "
              "0x400000}\n"}});
    };
    EXPECT_EQ(rows(with_tile_4_worker("1"), mmk_with_tile_4()), Rows{6});
    EXPECT_EQ(rows(with_tile_4_worker("0"), mmk_with_tile_4()), Rows{6});
}

// Row 6: a worker on hart 0 beside a router there. It also shares the router's hart (row 3); both are stage C.
TEST(VclusterManifest, Row6WorkerOnRoutersHart) {
    EXPECT_EQ(
        rows(edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 0, role: worker}"}})),
        (Rows{3, 6}));
}

// Row 7, each clause alone. A link end naming no device reports that and nothing else for that end; B's port 0, now
// in no link, is the other finding. The link is ethernet, and row 8 skips a link with an unresolved end.
TEST(VclusterManifest, Row7UnknownDevice) {
    const std::string yaml = edit({{"b: {device: B, port: 0}", "b: {device: C, port: 0}"}});
    EXPECT_EQ(rows(yaml), Rows{7});
    EXPECT_EQ(count_of(findings(yaml), ManifestCheck::Link), 2u);
}

TEST(VclusterManifest, Row7UnknownPort) {
    EXPECT_EQ(rows(edit({{"b: {device: B, port: 0}", "b: {device: B, port: 3}"}})), Rows{7});
}

// A second link from A port 0 to B port 0: both ports are in two links.
TEST(VclusterManifest, Row7PortTwice) {
    EXPECT_EQ(
        rows(edit({{"links:\n", "links:\n  - {a: {device: A, port: 0}, b: {device: B, port: 0}, backend: noc}\n"}})),
        Rows{7});
}

// M at both ends; A's and B's ports are linked to each other, so every port is in exactly one link.
TEST(VclusterManifest, Row7OneDeviceBothEnds) {
    EXPECT_EQ(
        rows(edit(
            {{"{a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}",
              "{a: {device: A, port: 0}, b: {device: B, port: 0}, backend: noc}"},
             {"{a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}",
              "{a: {device: M, port: 1}, b: {device: M, port: 0}, backend: noc}"}})),
        Rows{7});
}

// Row 7: a router port in no link; and § Checks: "With no links, every router port fails row 7."
TEST(VclusterManifest, Row7RouterPortWithoutLink) {
    EXPECT_EQ(
        rows(edit({{"  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}\n", ""}})), Rows{7});
    EXPECT_EQ(
        rows(edit(
            {{"links:\n  - {a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}\n"
              "  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}\n",
              "links: []\n"}})),
        Rows{7});
}

// Row 8: an ethernet link between tiles 1 and 3, which bring-up did not wire. A to M moves to M's tile 2 on the NOC.
TEST(VclusterManifest, Row8EthernetTilesNotWired) {
    EXPECT_EQ(
        rows(edit(
            {{"b: {device: M, port: 0}, backend: noc", "b: {device: M, port: 1}, backend: noc"},
             {"{a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}",
              "{a: {device: M, port: 0}, b: {device: B, port: 0}, backend: ethernet}"}})),
        Rows{8});
}

// Row 8 reads the wired pairs from the facts: without {2, 3}, the line's M to B ethernet link fails.
TEST(VclusterManifest, Row8FromFacts) {
    PackageFacts facts = mmk();
    facts.ethernet_tile_pairs = {{0, 1}};
    EXPECT_EQ(rows(kLine, facts), Rows{8});
}

// Row 8: a wired pair matches in either order. The line's M to B link joins tiles 2 and 3, and the A to M link, made
// ethernet, tiles 0 and 1; each pair is stored the other way round.
TEST(VclusterManifest, Row8PairEitherOrder) {
    PackageFacts facts = mmk();
    facts.ethernet_tile_pairs = {{3, 2}, {1, 0}};
    EXPECT_EQ(rows(kLine, facts), Rows{});
    EXPECT_EQ(rows(edit({{"backend: noc", "backend: ethernet"}}), facts), Rows{});
}

// Row 13: a device with no slots, and a device with workers only (whose worker also has no router, row 6).
TEST(VclusterManifest, Row13DeviceWithoutRouter) {
    const std::string no_slots =
        edit({{"links:\n", "  - name: C\n    fabric_chip: 3\n    slots: []\n    regions: []\nlinks:\n"}});
    EXPECT_EQ(rows(no_slots), Rows{13});
    const std::string workers_only = edit(
        {{"links:\n",
          "  - name: C\n    fabric_chip: 3\n    slots:\n      - {tile: 4, cce: 0, hart: 1, role: worker}\n"
          "    regions:\n      - {tile: 4, cce: 0, base: 0x0, size: 0x400000}\nlinks:\n"}});
    EXPECT_EQ(rows(workers_only, mmk_with_tile_4()), (Rows{6, 13}));
}

// § Checks: every stage C row that holds is reported. One manifest trips rows 3, 4, 5, 6, 7, 8 and 13.
TEST(VclusterManifest, StageCReportsEveryRow) {
    const std::string yaml = edit({
        {"links:\n",
         "  - name: C\n    fabric_chip: 3\n    slots:\n      - {tile: 4, cce: 0, hart: 1, role: worker}\n"
         "    regions:\n      - {tile: 4, cce: 0, base: 0x0, size: 0x400000}\nlinks:\n"},          // rows 6 and 13
        {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 4, role: worker}"},  // row 3
        {"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", "{tile: 0, cce: 0, base: 0x0, size: 0x400001}"},  // row 4
        {"      - {tile: 2, cce: 0, base: 0x0, size: 0x400000}\n", ""},                                    // row 5
        {"b: {device: M, port: 0}, backend: noc", "b: {device: M, port: 1}, backend: noc"},  // row 7: M port 0 unlinked
        {"{a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}",
         "{a: {device: A, port: 0}, b: {device: B, port: 0}, backend: ethernet}"},  // row 8: tiles 0, 3; row 7: A port
                                                                                    // 0 twice
    });
    EXPECT_EQ(rows(yaml, mmk_with_tile_4()), (Rows{3, 4, 5, 6, 7, 8, 13}));
}

// Stage B ends the check before stage C: a row-12 duplicate with a row-3 hart reports row 12 alone.
TEST(VclusterManifest, StageBEndsBeforeStageC) {
    EXPECT_EQ(
        rows(edit(
            {{"fabric_chip: 2", "fabric_chip: 1"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 4, role: worker}"}})),
        Rows{12});
}

// --- Load errors: the loader fails and reports no row ------------------------------------------------------------
//
// Each case is one edit, so the loader must report exactly one error, and it must name the key or value edited.

TEST(VclusterManifest, LoadErrorUnreadableFile) {
    EXPECT_THROW(load_manifest(std::filesystem::path("/nonexistent/vcluster/line3.yaml")), ManifestLoadError);
}

TEST(VclusterManifest, LoadErrorNotYaml) { EXPECT_EQ(load_errors("version: 1\ndevices: [\n").size(), 1u); }

TEST(VclusterManifest, LoadErrorVersion) {
    expect_load_error(edit({{"version: 1\n", "version: 2\n"}}), "version");
    expect_load_error(edit({{"version: 1\n", ""}}), "version");
}

// § Checks: "An unknown or missing `version` is reported alone and ends the load": an unknown key beside it is not
// reported.
TEST(VclusterManifest, LoadErrorVersionAlone) {
    expect_load_error(edit({{"version: 1\n", "version: 2\ntopology: line\n"}}), "version");
    expect_load_error(edit({{"version: 1\n", "topology: line\n"}}), "version");
}

TEST(VclusterManifest, LoadErrorMissingKey) {
    for (const auto& [from, to, needle] :
         std::initializer_list<std::tuple<std::string_view, std::string_view, std::string_view>>{
             {"  - name: A\n    fabric_chip: 0\n", "  - fabric_chip: 0\n", "name"},
             {"    fabric_chip: 1\n", "", "fabric_chip"},
             {"{tile: 0, cce: 0, hart: 1, role: worker}", "{cce: 0, hart: 1, role: worker}", "tile"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, role: worker}", "hart"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1}", "role"},
             {"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", "{tile: 0, cce: 0, base: 0x0}", "size"},
             {"b: {device: B, port: 0}, backend: ethernet", "b: {device: B, port: 0}", "backend"},
             {"b: {device: B, port: 0}, backend: ethernet", "backend: ethernet", "links[1].b"},
             {"b: {device: B, port: 0}", "b: {device: B}", "port"},
         }) {
        SCOPED_TRACE(to);
        expect_load_error(edit({{from, to}}), needle);
    }
}

// § Checks: `links`, and each device's `slots` and `regions`, are required; an empty list is well formed.
TEST(VclusterManifest, LoadErrorMissingList) {
    expect_load_error(
        edit(
            {{"links:\n  - {a: {device: A, port: 0}, b: {device: M, port: 0}, backend: noc}\n"
              "  - {a: {device: M, port: 1}, b: {device: B, port: 0}, backend: ethernet}\n",
              ""}}),
        "links");
    expect_load_error(
        edit(
            {{"    slots:\n      - {tile: 3, cce: 0, hart: 0, role: router, port: 0}\n"
              "      - {tile: 3, cce: 0, hart: 1, role: worker}\n",
              ""}}),
        "slots");
    expect_load_error(edit({{"    regions:\n      - {tile: 3, cce: 0, base: 0x0, size: 0x400000}\n", ""}}), "regions");
}

// § Checks: an integer is decimal, or hexadecimal after a lowercase `0x`; `0` is accepted.
TEST(VclusterManifest, IntegersDecimalOrHex) {
    const Manifest m = load(edit(
        {{"{tile: 3, cce: 0, base: 0x0, size: 0x400000}", "{tile: 3, cce: 0, base: 0, size: 4194304}"},
         {"fabric_chip: 2", "fabric_chip: 10"},
         {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 0x3, cce: 0, hart: 0x1, role: worker}"}}));
    const Device& b = m.devices[2];
    EXPECT_EQ(b.fabric_chip, 10u);
    EXPECT_EQ(b.regions[0].base, 0u);
    EXPECT_EQ(b.regions[0].size, kSram);
    EXPECT_EQ(b.slots[1].tile, 3u);
    EXPECT_EQ(b.slots[1].hart, 1u);
}

// § Checks: a value too large for its field is a load error, 32 bits for every number but a region's `base` and
// `size`, which are 64. The largest value of each width loads.
TEST(VclusterManifest, IntegerWidths) {
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 4294967296, cce: 0, hart: 1, role: worker}"}}),
        "tile");
    expect_load_error(edit({{"fabric_chip: 2", "fabric_chip: 0x100000000"}}), "fabric_chip");
    expect_load_error(
        edit(
            {{"{tile: 3, cce: 0, base: 0x0, size: 0x400000}",
              "{tile: 3, cce: 0, base: 18446744073709551616, size: 0x400000}"}}),
        "base");
    expect_load_error(
        edit(
            {{"{tile: 3, cce: 0, base: 0x0, size: 0x400000}",
              "{tile: 3, cce: 0, base: 0x0, size: 0x10000000000000000}"}}),
        "size");
    const Manifest m = load(edit(
        {{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 4294967295, cce: 0, hart: 1, role: worker}"},
         {"{tile: 3, cce: 0, base: 0x0, size: 0x400000}",
          "{tile: 3, cce: 0, base: 0xffffffffffffffff, size: 18446744073709551615}"}}));
    EXPECT_EQ(m.devices[2].slots[1].tile, 4294967295u);
    EXPECT_EQ(m.devices[2].regions[0].base, UINT64_MAX);
    EXPECT_EQ(m.devices[2].regions[0].size, UINT64_MAX);
}

// Rows 1 and 4 compare `base + size` with the SRAM without overflow: a region whose sum wraps past 2^64 ends beyond
// the SRAM (row 4), not inside it (row 1, which a wrapped sum would give).
TEST(VclusterManifest, Row4RegionNearUint64Max) {
    for (const std::string_view region : {
             "{tile: 0, cce: 0, base: 0xffffffffffffffff, size: 0x1}",
             "{tile: 0, cce: 0, base: 0x1, size: 0xffffffffffffffff}",
             "{tile: 0, cce: 0, base: 0xffffffffffffffff, size: 0xffffffffffffffff}",
             "{tile: 0, cce: 0, base: 0xfffffffffffffff0, size: 0x400010}",
         }) {
        SCOPED_TRACE(region);
        EXPECT_EQ(rows(edit({{"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", region}})), Rows{4});
    }
}

// § Checks: these are load errors, not octal, hex, signed or string numbers.
TEST(VclusterManifest, LoadErrorIntegerSyntax) {
    for (const auto& [from, to, needle] :
         std::initializer_list<std::tuple<std::string_view, std::string_view, std::string_view>>{
             {"fabric_chip: 2", "fabric_chip: 010", "fabric_chip"},  // a leading zero
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 03, cce: 0, hart: 1, role: worker}", "tile"},
             {"{tile: 3, cce: 0, base: 0x0, size: 0x400000}", "{tile: 3, cce: 0, base: 00, size: 0x400000}", "base"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 0X3, cce: 0, hart: 1, role: worker}", "tile"},
             {"{tile: 3, cce: 0, base: 0x0, size: 0x400000}", "{tile: 3, cce: 0, base: 0x0, size: 0X400000}", "size"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: +1, role: worker}", "hart"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: -1, role: worker}", "hart"},
             {"fabric_chip: 2", "fabric_chip: -2", "fabric_chip"},
             {"b: {device: B, port: 0}", "b: {device: B, port: -1}", "port"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: \"3\", cce: 0, hart: 1, role: worker}", "tile"},
             {"fabric_chip: 2", "fabric_chip: '2'", "fabric_chip"},
         }) {
        SCOPED_TRACE(to);
        expect_load_error(edit({{from, to}}), needle);
    }
}

TEST(VclusterManifest, LoadErrorRouterWithoutPort) {
    expect_load_error(
        edit({{"{tile: 2, cce: 0, hart: 0, role: router, port: 1}", "{tile: 2, cce: 0, hart: 0, role: router}"}}),
        "port");
}

TEST(VclusterManifest, LoadErrorWorkerWithPort) {
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1, role: worker, port: 1}"}}),
        "port");
}

TEST(VclusterManifest, LoadErrorUnknownKey) {
    expect_load_error(edit({{"version: 1\n", "version: 1\ntopology: line\n"}}), "topology");
    expect_load_error(edit({{"    fabric_chip: 1\n", "    fabric_chip: 1\n    color: red\n"}}), "color");
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1, role: worker, cpu: 1}"}}),
        "cpu");
    expect_load_error(
        edit(
            {{"{tile: 3, cce: 0, base: 0x0, size: 0x400000}",
              "{tile: 3, cce: 0, base: 0x0, size: 0x400000, end: 0x400000}"}}),
        "end");
    expect_load_error(edit({{"backend: noc}", "backend: noc, vc: 2}"}}), "vc");
}

TEST(VclusterManifest, LoadErrorWrongType) {
    for (const auto& [from, to, needle] :
         std::initializer_list<std::tuple<std::string_view, std::string_view, std::string_view>>{
             {"version: 1\n", "version: one\n", "version"},
             {"fabric_chip: 2", "fabric_chip: [2]", "fabric_chip"},
             {"name: A", "name: {first: A}", "name"},
             {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: three, cce: 0, hart: 1, role: worker}", "tile"},
             {"{tile: 0, cce: 0, base: 0x0, size: 0x400000}", "{tile: 0, cce: 0, base: 0x0, size: big}", "size"},
             {"    slots:\n      - {tile: 3, cce: 0, hart: 0, role: router, port: 0}\n      - {tile: 3, cce: 0, hart: "
              "1, "
              "role: worker}\n",
              "    slots: 3\n",
              "slots"},
             {"b: {device: B, port: 0}", "b: B", "links[1].b"},
         }) {
        SCOPED_TRACE(to);
        expect_load_error(edit({{from, to}}), needle);
    }
    EXPECT_EQ(load_errors("- version: 1\n").size(), 1u);  // the root is not a map
}

TEST(VclusterManifest, LoadErrorUnknownRoleOrBackend) {
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1, role: master}"}}), "master");
    expect_load_error(edit({{"backend: ethernet", "backend: pcie"}}), "pcie");
}

// § Checks: "A slot with an unknown `role` is a load error whatever else it holds", a `port` included. Whether the
// port is a second error is not stated, so only the role's is required.
TEST(VclusterManifest, LoadErrorUnknownRoleWithPort) {
    const auto errors = load_errors(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1, role: master, port: 1}"}}));
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
        return e.find("master") != std::string::npos;
    })) << joined(errors);
}

TEST(VclusterManifest, LoadErrorNoDevices) { expect_load_error("version: 1\ndevices: []\nlinks: []\n", "devices"); }

// An empty `capture` is a load error.
TEST(VclusterManifest, LoadErrorEmptyCapture) {
    expect_load_error(edit({{"version: 1\n", "version: 1\ncapture: \"\"\n"}}), "capture");
}

// § Checks: a key repeated in any map, a file of more than one YAML document, and a number with an explicit tag are
// load errors.
TEST(VclusterManifest, LoadErrorRepeatedKey) {
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, tile: 3, cce: 0, hart: 1, role: worker}"}}),
        "tile");
    expect_load_error(edit({{"    fabric_chip: 1\n", "    fabric_chip: 1\n    fabric_chip: 1\n"}}), "fabric_chip");
}

TEST(VclusterManifest, LoadErrorMultipleDocuments) {
    EXPECT_EQ(load_errors(std::string(kLine) + "---\n" + std::string(kLine)).size(), 1u);
}

TEST(VclusterManifest, LoadErrorTaggedScalar) {
    expect_load_error(
        edit({{"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: !!int 3, cce: 0, hart: 1, role: worker}"}}), "tile");
}

// § Checks: the loader lists every load error it found. Two independent edits in different items give two errors.
TEST(VclusterManifest, LoadErrorListsEvery) {
    const auto errors = load_errors(edit(
        {{"    fabric_chip: 1\n", "    fabric_chip: 1\n    color: red\n"},
         {"{tile: 3, cce: 0, hart: 1, role: worker}", "{tile: 3, cce: 0, hart: 1, role: master}"}}));
    ASSERT_EQ(errors.size(), 2u) << joined(errors);
    const auto names = [&](std::string_view needle) {
        return std::count_if(
            errors.begin(), errors.end(), [&](const std::string& e) { return e.find(needle) != std::string::npos; });
    };
    EXPECT_EQ(names("color"), 1) << joined(errors);
    EXPECT_EQ(names("master"), 1) << joined(errors);
}

// § Checks: "The checker asserts that its package facts are usable: a hart count and an SRAM size above 0, and at
// least one core. Unusable facts are a caller's bug, not a row." The spec names no exception type.
TEST(VclusterManifest, UnusableFactsThrow) {
    const Manifest m = load(kLine);
    EXPECT_ANY_THROW(check_manifest(m, PackageFacts{}));
    PackageFacts no_harts = mmk();
    no_harts.harts_per_cce = 0;
    EXPECT_ANY_THROW(check_manifest(m, no_harts));
    PackageFacts no_sram = mmk();
    no_sram.sram_bytes_per_cce = 0;
    EXPECT_ANY_THROW(check_manifest(m, no_sram));
    PackageFacts no_cores = mmk();
    no_cores.cores.clear();
    EXPECT_ANY_THROW(check_manifest(m, no_cores));
    EXPECT_NO_THROW(check_manifest(m, mmk()));
}

}  // namespace
}  // namespace tt::tt_fabric::vcluster
