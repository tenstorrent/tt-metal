// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The virtual-cluster manifest and the checks that need only the manifest and facts about the package (stages A to
// C). No device access.

#pragma once

#include <compare>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tt::tt_fabric::vcluster {

// The check a finding belongs to; the value is its row number. Rows 9 to 11 compare
// against the capture (stage D) and are not checked here.
enum class ManifestCheck : uint32_t {
    ReservedValue = 1,          // `cce` 1, a router on a hart other than 0, a region inside its CCE's SRAM that is
                                // not the whole of it (size 0 included), backend `local`
    CceNotInPackage = 2,        // a slot's or region's (tile, cce) is not in PackageFacts::cores
    Slot = 3,                   // two slots on one (tile, cce, hart), or a hart not below the CCE's hart count
    Region = 4,                 // two regions overlap, or a region's base + size is beyond its CCE's SRAM
    SlotWithoutRegion = 5,      // a slot on a CCE its device owns no region of
    WorkerPlacement = 6,        // a worker on hart 0, or a CCE with a worker and no router on hart 0
    Link = 7,                   // a link end names no device or no router port, a port is in two link ends, one
                                // device is at both ends, or a router port is in no link
    EthernetTilesNotWired = 8,  // an ethernet link's two router tiles are not in PackageFacts::ethernet_tile_pairs
    Duplicate = 12,             // two devices share a name or a fabric_chip, or two routers of a device a port
    DeviceWithoutRouter = 13,   // a device has no router slot
};

// One failing instance of a check row. A manifest can produce several findings for one row.
struct ManifestFinding {
    ManifestCheck check;
    std::string detail;  // names the device, slot, region or link; for people, not parsed
};

// Idle: an Ethernet core of the device with no link, on which Metal launches no router.
enum class SlotRole { Router, Worker, Idle };

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
    std::optional<uint32_t> port;  // set exactly when role is Router or Idle: the core's Ethernet channel
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

// A device port. Its tile is the tile of the device's router slot with this port.
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
    // The capture file, when the manifest names one. parse_manifest() keeps it as written; load_manifest() resolves
    // a relative path against the manifest's directory.
    std::optional<std::filesystem::path> capture;
    std::vector<Device> devices;  // never empty
    std::vector<Link> links;
};

// A file that cannot be turned into a manifest. errors() lists every load error found, one line each; what() joins
// them.
class ManifestLoadError : public std::runtime_error {
public:
    explicit ManifestLoadError(std::vector<std::string> errors);
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::vector<std::string> errors_;
};

// Facts about the open package the checks need. The caller fills them, from the open SoC descriptor and the
// session's bring-up; nothing here reads a device or a descriptor.
struct PackageFacts {
    std::vector<CceId> cores;         // the SoC descriptor's keraunos.cores (row 2)
    uint32_t harts_per_cce = 0;       // harts of every CCE (row 3)
    uint64_t sram_bytes_per_cce = 0;  // SRAM of every CCE; a region is exactly this (rows 1 and 4)
    // HSIO tile pairs the session's bring-up wired with Ethernet, in either order (row 8). On the emulated mmk
    // package: {0, 1} and {2, 3}.
    std::vector<std::pair<uint32_t, uint32_t>> ethernet_tile_pairs;
};

// Reads a manifest from YAML text. Throws ManifestLoadError when the text does not parse as YAML, is not exactly one
// YAML document, or its root is not a map, and when `version` is missing or not 1; each of these is reported alone,
// since the rest of the text cannot be judged. Otherwise throws ManifestLoadError listing every load error found: a
// required key missing; a key unknown (`port` on a worker included) or repeated in one map; a value of the wrong type
// or negative; an integer that is not a plain untagged scalar in decimal without a leading 0 (other than "0") or in
// hexadecimal after "0x"; a `role` or `backend` that is not a known name; a router or idle slot with no `port`; an
// empty `capture`; an empty `devices`. Required keys: version, devices, links; a device's name, fabric_chip, slots,
// regions; every field of a slot (port for a router only), a region, a link and a link end. `links`, `slots` and
// `regions` may be empty sequences.
Manifest parse_manifest(std::string_view yaml_text);

// parse_manifest() on the file at `path`, then a relative `capture` resolved against the file's directory. Throws
// ManifestLoadError naming the path when the file cannot be opened or read, and as parse_manifest() otherwise.
Manifest load_manifest(const std::filesystem::path& path);

// The device of `manifest` named `name`, or null.
const Device* find_device(const Manifest& manifest, const std::string& name);
// The router slot `end` names: the slot of that device with role router and that port. Null when the device or the
// port does not exist.
const Slot* find_router(const Manifest& manifest, const LinkEnd& end);

// Runs the check rows in stages against `manifest` and `facts`: stage A (row 1); if it found nothing,
// stage B (rows 2 and 12); if that found nothing, stage C (rows 3 to 8 and 13). Returns every failing instance of
// every row the last stage run checked, in row order; empty means the manifest passes stages A to C. Stage D (rows
// 9 to 11, against the capture) is not run. Reads no device. Throws std::runtime_error (TT_FATAL) only on a caller
// bug, not a manifest fault: `facts` with no cores, 0 harts or 0 SRAM bytes, or a manifest with no devices.
std::vector<ManifestFinding> check_manifest(const Manifest& manifest, const PackageFacts& facts);

}  // namespace tt::tt_fabric::vcluster
