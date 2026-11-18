// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The virtual-cluster manifest (SPEC-0011 § Manifest) and the checks that need only the manifest and facts about
// the package (SPEC-0011 § Checks before the device is touched). No device access.

#pragma once

#include <compare>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tt::tt_fabric::vcluster {

// The row of SPEC-0011's check table a finding belongs to; the value is the row number. Rows 9 and 11 and the
// capture half of row 10 compare against the capture and are not checked here.
enum class ManifestCheck : uint32_t {
    Format = 1,                  // unknown version, key or value, a missing or mistyped key, a Phase 3 value, a
                                 // duplicated device name, fabric_chip or router port
    CceNotInPackage = 2,         // a slot's or region's (tile, cce) is not in PackageFacts::cores
    Slot = 3,                    // two slots on one (tile, cce, hart), or a hart not below the CCE's hart count
    Region = 4,                  // two regions overlap, or a region leaves its CCE's SRAM
    SlotWithoutRegion = 5,       // a slot on a CCE its device owns no region of
    WorkerWithoutRouter = 6,     // a CCE with a worker and no router on hart 0
    Link = 7,                    // a link end names no device or no router port, a port is in two links, or one device
                                 // is at both ends
    EthernetTilesNotWired = 8,   // an ethernet link's two router tiles are not in PackageFacts::ethernet_tile_pairs
    RouterPortWithoutLink = 10,  // a router's port is in no link
};

// One failing instance of a check. A manifest can produce several findings for one row.
struct ManifestFinding {
    ManifestCheck check;
    std::string detail;  // names the device, slot, region or link; for people, not parsed
};

enum class SlotRole { Router, Worker };

// `local` is parsed so that check_manifest() can refuse it as a Phase 3 value.
enum class LinkBackend { Ethernet, Noc, Local };

// A CCE of the package: its HSIO tile and its index in the tile.
struct CceId {
    uint32_t tile = 0;
    uint32_t cce = 0;
    auto operator<=>(const CceId&) const = default;
};

struct Slot {
    uint32_t tile = 0;
    uint32_t cce = 0;
    uint32_t hart = 0;
    SlotRole role = SlotRole::Worker;
    std::optional<uint32_t> port;  // a router's Ethernet channel in the capture; empty for a worker, or when missing
};

// SRAM of one CCE a device owns, [base, base + size) in the CCE's SRAM offsets.
struct Region {
    uint32_t tile = 0;
    uint32_t cce = 0;
    uint64_t base = 0;
    uint64_t size = 0;
};

struct Device {
    std::string name;
    uint32_t fabric_chip = 0;
    std::vector<Slot> slots;
    std::vector<Region> regions;
};

struct LinkEnd {
    std::string device;
    uint32_t port = 0;
};

struct Link {
    LinkEnd a;
    LinkEnd b;
    LinkBackend backend = LinkBackend::Ethernet;
};

struct Manifest {
    uint32_t version = 0;
    std::string capture;  // as written: a path relative to the manifest's directory
    std::vector<Device> devices;
    std::vector<Link> links;
};

// What parse_manifest() read, and the row-1 findings it met while reading. An item missing a key it cannot do
// without (a slot's tile, a link end's device, ...) or carrying a mistyped or unknown value is reported and left
// out of `manifest`; a router slot missing only its port is kept with `port` empty.
struct ManifestParse {
    Manifest manifest;
    std::vector<ManifestFinding> findings;
};

// Facts about the open package the checks need. The caller fills them, from the open SoC descriptor and the
// session's bring-up; nothing here reads a device or a descriptor.
struct PackageFacts {
    std::vector<CceId> cores;         // the SoC descriptor's keraunos.cores (row 2)
    uint32_t harts_per_cce = 0;       // harts of every CCE (row 3)
    uint64_t sram_bytes_per_cce = 0;  // SRAM of every CCE; a Phase 2 region is exactly this (rows 1 and 4)
    // HSIO tile pairs the session's bring-up wired with Ethernet, in either order (row 8). On mmk 26ww45 with
    // chippy bringup_mmk.py KER_ETH_LINKS: {0, 1} and {2, 3}.
    std::vector<std::pair<uint32_t, uint32_t>> ethernet_tile_pairs;
};

// Reads a manifest from YAML text. Format problems (row 1) are findings, not exceptions: a root that is not a map
// or an unknown or missing `version` stops the read with one finding and an empty manifest; otherwise every unknown
// key, missing required key, mistyped value and unknown enum string is reported and reading goes on. Throws
// std::runtime_error only when the text is not YAML (a syntax error).
ManifestParse parse_manifest(std::string_view yaml_text);

// parse_manifest() on the file at `path`. Throws std::runtime_error naming the path when the file cannot be opened
// or read, and as parse_manifest() otherwise.
ManifestParse load_manifest(const std::filesystem::path& path);

// Runs row 1's value rules (Phase 3 values: `cce` 1, a router on a hart other than 0, a region other than its whole
// CCE, backend `local`; a device name or fabric_chip used twice; two routers of one device on one port), rows 2 to
// 8, and the manifest half of row 10 against `manifest` and `facts`. Returns every failing
// instance of every row, in row order, not only the first; empty means the manifest passes. Never throws; reads
// no device.
std::vector<ManifestFinding> check_manifest(const Manifest& manifest, const PackageFacts& facts);

}  // namespace tt::tt_fabric::vcluster
