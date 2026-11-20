// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "manifest.hpp"

#include <algorithm>
#include <charconv>
#include <type_traits>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt_stl/assert.hpp>
#include <yaml-cpp/yaml.h>

namespace tt::tt_fabric::vcluster {

ManifestLoadError::ManifestLoadError(std::vector<std::string> errors) :
    std::runtime_error(fmt::format("manifest load error: {}", fmt::join(errors, "; "))), errors_(std::move(errors)) {}

namespace {

constexpr uint32_t kManifestVersion = 1;
// Phase 2 values (SPEC-0011 § Manifest): routers on hart 0 of CCE0. CCE1 takes idle slots only: a router is built for
// CCE0, and a worker slot is a hart of a router's CCE.
constexpr uint32_t kRouterHart = 0;
constexpr uint32_t kIdleOnlyCce = 1;

// Load errors, collected so that one ManifestLoadError lists them all.
using Errors = std::vector<std::string>;
using Findings = std::vector<ManifestFinding>;

// Reports every key of the map `node` that is not in `known`, and every key that appears more than once: yaml-cpp
// keeps both entries of a repeated key and node[key] returns one of them.
void check_keys(
    const YAML::Node& node, std::initializer_list<std::string_view> known, const std::string& where, Errors& errors) {
    std::set<std::string> seen;
    for (const auto& entry : node) {
        const std::string key = entry.first.IsScalar() ? entry.first.Scalar() : std::string("<not a scalar>");
        if (std::find(known.begin(), known.end(), key) == known.end()) {
            errors.push_back(fmt::format("{}: unknown key '{}'", where, key));
        }
        if (!seen.insert(key).second) {
            errors.push_back(fmt::format("{}: duplicate key '{}'", where, key));
        }
    }
}

// `text` as an unsigned T: decimal with no leading zero other than "0" itself, or hexadecimal after "0x" (SPEC-0011
// § Checks). Empty for anything else, a sign included, or a value T cannot hold.
template <typename T>
std::optional<T> parse_integer(std::string_view text) {
    static_assert(std::is_unsigned_v<T>);
    constexpr std::string_view kHexPrefix = "0x";
    constexpr int kHex = 16;
    constexpr int kDecimal = 10;
    int base = kDecimal;
    if (text.starts_with(kHexPrefix)) {
        text.remove_prefix(kHexPrefix.size());
        base = kHex;
    } else if (text.size() > 1 && text.front() == '0') {
        return std::nullopt;
    }
    T out{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out, base);
    if (text.empty() || ec != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return out;
}

// A scalar as a T: a string as written; an integer by parse_integer(), from a plain scalar with no tag only.
template <typename T>
std::optional<T> decode_scalar(const YAML::Node& value) {
    if (!value.IsScalar()) {
        return std::nullopt;
    }
    if constexpr (std::is_same_v<T, std::string>) {
        return value.Scalar();
    } else {
        // yaml-cpp tags a plain untagged scalar "?"; a quoted one gets "!", and an explicit tag (!!str, !!int, !foo)
        // its own.
        if (value.Tag() != "?") {
            return std::nullopt;
        }
        return parse_integer<T>(value.Scalar());
    }
}

// The value of `key` in the map `node` as a T, or empty with an error when it is missing or does not decode.
template <typename T>
std::optional<T> read_value(const YAML::Node& node, const char* key, const std::string& where, Errors& errors) {
    const YAML::Node value = node[key];
    if (!value) {
        errors.push_back(fmt::format("{}: missing key '{}'", where, key));
        return std::nullopt;
    }
    const std::optional<T> out = decode_scalar<T>(value);
    if (!out) {
        errors.push_back(fmt::format(
            "{}: '{}' is not {} ({})",
            where,
            key,
            std::is_same_v<T, std::string> ? "a string" : "a decimal or 0x integer in range, without sign or leading 0",
            value.IsScalar() ? value.Scalar() : std::string("not a scalar")));
        return std::nullopt;
    }
    return out;
}

// The sequence at `key` in `node`, or an empty node with an error when it is missing or not a sequence.
YAML::Node read_sequence(const YAML::Node& node, const char* key, const std::string& where, Errors& errors) {
    const YAML::Node value = node[key];
    if (!value) {
        errors.push_back(fmt::format("{}: missing key '{}'", where, key));
        return YAML::Node();
    }
    if (!value.IsSequence()) {
        errors.push_back(fmt::format("{}: '{}' is not a sequence", where, key));
        return YAML::Node();
    }
    return value;
}

bool is_map(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!node.IsMap()) {
        errors.push_back(fmt::format("{}: not a map", where));
        return false;
    }
    return true;
}

std::optional<SlotRole> parse_role(const std::string& text) {
    if (text == "router") {
        return SlotRole::Router;
    }
    if (text == "worker") {
        return SlotRole::Worker;
    }
    if (text == "idle") {
        return SlotRole::Idle;
    }
    return std::nullopt;
}

std::optional<LinkBackend> parse_backend(const std::string& text) {
    if (text == "ethernet") {
        return LinkBackend::Ethernet;
    }
    if (text == "noc") {
        return LinkBackend::Noc;
    }
    if (text == "local") {
        return LinkBackend::Local;
    }
    return std::nullopt;
}

std::optional<Slot> parse_slot(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!is_map(node, where, errors)) {
        return std::nullopt;
    }
    const auto tile = read_value<uint32_t>(node, "tile", where, errors);
    const auto cce = read_value<uint32_t>(node, "cce", where, errors);
    const auto hart = read_value<uint32_t>(node, "hart", where, errors);
    const auto role_text = read_value<std::string>(node, "role", where, errors);
    std::optional<SlotRole> role;
    if (role_text) {
        role = parse_role(*role_text);
        if (!role) {
            errors.push_back(fmt::format("{}: unknown role '{}'", where, *role_text));
        }
    }
    std::optional<uint32_t> port;
    if (role == SlotRole::Worker) {
        check_keys(node, {"tile", "cce", "hart", "role"}, where, errors);
    } else {
        check_keys(node, {"tile", "cce", "hart", "role", "port"}, where, errors);
    }
    const bool has_port = role && *role != SlotRole::Worker;
    if (has_port) {
        port = read_value<uint32_t>(node, "port", where, errors);
    }
    if (!tile || !cce || !hart || !role || (has_port && !port)) {
        return std::nullopt;
    }
    return Slot{.tile = *tile, .cce = *cce, .hart = *hart, .role = *role, .port = port};
}

std::optional<Region> parse_region(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!is_map(node, where, errors)) {
        return std::nullopt;
    }
    check_keys(node, {"tile", "cce", "base", "size"}, where, errors);
    const auto tile = read_value<uint32_t>(node, "tile", where, errors);
    const auto cce = read_value<uint32_t>(node, "cce", where, errors);
    const auto base = read_value<uint64_t>(node, "base", where, errors);
    const auto size = read_value<uint64_t>(node, "size", where, errors);
    if (!tile || !cce || !base || !size) {
        return std::nullopt;
    }
    return Region{.tile = *tile, .cce = *cce, .base = *base, .size = *size};
}

std::optional<Device> parse_device(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!is_map(node, where, errors)) {
        return std::nullopt;
    }
    check_keys(node, {"name", "fabric_chip", "slots", "regions"}, where, errors);
    const auto name = read_value<std::string>(node, "name", where, errors);
    const auto fabric_chip = read_value<uint32_t>(node, "fabric_chip", where, errors);
    const std::string label = name ? fmt::format("{} ({})", where, *name) : where;
    Device device;
    const YAML::Node slots = read_sequence(node, "slots", label, errors);
    for (std::size_t i = 0; i < slots.size(); i++) {
        if (auto slot = parse_slot(slots[i], fmt::format("{}.slots[{}]", label, i), errors)) {
            device.slots.push_back(*slot);
        }
    }
    const YAML::Node regions = read_sequence(node, "regions", label, errors);
    for (std::size_t i = 0; i < regions.size(); i++) {
        if (auto region = parse_region(regions[i], fmt::format("{}.regions[{}]", label, i), errors)) {
            device.regions.push_back(*region);
        }
    }
    if (!name || !fabric_chip) {
        return std::nullopt;
    }
    device.name = *name;
    device.fabric_chip = *fabric_chip;
    return device;
}

std::optional<LinkEnd> parse_link_end(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!node) {
        errors.push_back(fmt::format("{}: missing", where));
        return std::nullopt;
    }
    if (!is_map(node, where, errors)) {
        return std::nullopt;
    }
    check_keys(node, {"device", "port"}, where, errors);
    const auto device = read_value<std::string>(node, "device", where, errors);
    const auto port = read_value<uint32_t>(node, "port", where, errors);
    if (!device || !port) {
        return std::nullopt;
    }
    return LinkEnd{.device = *device, .port = *port};
}

std::optional<Link> parse_link(const YAML::Node& node, const std::string& where, Errors& errors) {
    if (!is_map(node, where, errors)) {
        return std::nullopt;
    }
    check_keys(node, {"a", "b", "backend"}, where, errors);
    const auto a = parse_link_end(node["a"], where + ".a", errors);
    const auto b = parse_link_end(node["b"], where + ".b", errors);
    const auto backend_text = read_value<std::string>(node, "backend", where, errors);
    std::optional<LinkBackend> backend;
    if (backend_text) {
        backend = parse_backend(*backend_text);
        if (!backend) {
            errors.push_back(fmt::format("{}: unknown backend '{}'", where, *backend_text));
        }
    }
    if (!a || !b || !backend) {
        return std::nullopt;
    }
    return Link{.a = *a, .b = *b, .backend = *backend};
}

std::string cce_label(uint32_t tile, uint32_t cce) { return fmt::format("(tile {}, cce {})", tile, cce); }

std::string slot_label(const Device& device, std::size_t index) {
    const Slot& slot = device.slots[index];
    return fmt::format(
        "device {} slot {} (tile {}, cce {}, hart {})", device.name, index, slot.tile, slot.cce, slot.hart);
}

std::string region_label(const Device& device, std::size_t index) {
    const Region& region = device.regions[index];
    return fmt::format(
        "device {} region {} (tile {}, cce {}, base {:#x}, size {:#x})",
        device.name,
        index,
        region.tile,
        region.cce,
        region.base,
        region.size);
}

std::string end_label(std::size_t link, char side, const LinkEnd& end) {
    return fmt::format("links[{}].{} (device {}, port {})", link, side, end.device, end.port);
}

// True when [base, base + size) ends beyond `sram` bytes, computed without forming base + size.
bool beyond_sram(const Region& region, uint64_t sram) { return region.base > sram || region.size > sram - region.base; }

const Device* find_device(const Manifest& manifest, const std::string& name) {
    const auto it =
        std::find_if(manifest.devices.begin(), manifest.devices.end(), [&](const Device& d) { return d.name == name; });
    return it == manifest.devices.end() ? nullptr : &*it;
}

// The router slot `end` names: the slot of that device with role router and that port. Null when the device or the
// port does not exist.
const Slot* find_router(const Manifest& manifest, const LinkEnd& end) {
    const Device* device = find_device(manifest, end.device);
    if (device == nullptr) {
        return nullptr;
    }
    const auto it = std::find_if(device->slots.begin(), device->slots.end(), [&](const Slot& s) {
        return s.role == SlotRole::Router && s.port == end.port;
    });
    return it == device->slots.end() ? nullptr : &*it;
}

// Row 1.
void check_reserved_values(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (slot.cce == kIdleOnlyCce && slot.role != SlotRole::Idle) {
                findings.push_back(
                    {ManifestCheck::ReservedValue,
                     fmt::format("{}: cce {} takes idle slots only", slot_label(device, i), slot.cce)});
            }
            if (slot.role != SlotRole::Worker && slot.hart != kRouterHart) {
                findings.push_back(
                    {ManifestCheck::ReservedValue,
                     fmt::format(
                         "{}: a router or idle slot on a hart other than {} is reserved for Phase 3",
                         slot_label(device, i),
                         kRouterHart)});
            }
        }
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            const bool whole = region.base == 0 && region.size == facts.sram_bytes_per_cce;
            if (!whole && !beyond_sram(region, facts.sram_bytes_per_cce)) {
                findings.push_back(
                    {ManifestCheck::ReservedValue,
                     fmt::format(
                         "{}: a part of a CCE (the whole is base 0, size {:#x}) is reserved for Phase 3",
                         region_label(device, i),
                         facts.sram_bytes_per_cce)});
            }
        }
    }
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        if (manifest.links[i].backend == LinkBackend::Local) {
            findings.push_back(
                {ManifestCheck::ReservedValue, fmt::format("links[{}]: backend local is reserved for Phase 3", i)});
        }
    }
}

// Row 2.
void check_cces_in_package(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    const std::set<CceId> cores(facts.cores.begin(), facts.cores.end());
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (!cores.contains(CceId{slot.tile, slot.cce})) {
                findings.push_back(
                    {ManifestCheck::CceNotInPackage,
                     fmt::format(
                         "{}: {} is not a Keraunos core of the package",
                         slot_label(device, i),
                         cce_label(slot.tile, slot.cce))});
            }
        }
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            if (!cores.contains(CceId{region.tile, region.cce})) {
                findings.push_back(
                    {ManifestCheck::CceNotInPackage,
                     fmt::format(
                         "{}: {} is not a Keraunos core of the package",
                         region_label(device, i),
                         cce_label(region.tile, region.cce))});
            }
        }
    }
}

// Row 12.
void check_duplicates(const Manifest& manifest, Findings& findings) {
    std::set<std::string> names;
    std::set<uint32_t> chips;
    for (const Device& device : manifest.devices) {
        if (!names.insert(device.name).second) {
            findings.push_back(
                {ManifestCheck::Duplicate, fmt::format("device name {} is used more than once", device.name)});
        }
        if (!chips.insert(device.fabric_chip).second) {
            findings.push_back(
                {ManifestCheck::Duplicate, fmt::format("fabric_chip {} is used more than once", device.fabric_chip)});
        }
        std::set<uint32_t> ports;
        for (const Slot& slot : device.slots) {
            if (slot.port && !ports.insert(*slot.port).second) {
                findings.push_back(
                    {ManifestCheck::Duplicate,
                     fmt::format("device {}: two slots on port {}", device.name, *slot.port)});
            }
        }
    }
}

// Row 3.
void check_slots(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    std::set<std::tuple<uint32_t, uint32_t, uint32_t>> harts;
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (!harts.insert({slot.tile, slot.cce, slot.hart}).second) {
                findings.push_back(
                    {ManifestCheck::Slot, fmt::format("{}: another slot is on this hart", slot_label(device, i))});
            }
            if (slot.hart >= facts.harts_per_cce) {
                findings.push_back(
                    {ManifestCheck::Slot,
                     fmt::format(
                         "{}: hart {} is not below the CCE's {} harts",
                         slot_label(device, i),
                         slot.hart,
                         facts.harts_per_cce)});
            }
        }
    }
}

// Row 4.
void check_regions(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    struct Placed {
        const Device* device;
        std::size_t index;
    };
    std::map<CceId, std::vector<Placed>> by_cce;
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            if (beyond_sram(region, facts.sram_bytes_per_cce)) {
                findings.push_back(
                    {ManifestCheck::Region,
                     fmt::format(
                         "{}: ends beyond the CCE's {:#x} bytes of SRAM",
                         region_label(device, i),
                         facts.sram_bytes_per_cce)});
            }
            by_cce[CceId{region.tile, region.cce}].push_back(Placed{&device, i});
        }
    }
    for (const auto& [cce, placed] : by_cce) {
        for (std::size_t i = 0; i < placed.size(); i++) {
            for (std::size_t j = i + 1; j < placed.size(); j++) {
                const Region& x = placed[i].device->regions[placed[i].index];
                const Region& y = placed[j].device->regions[placed[j].index];
                // [base, base + size) intervals, compared without forming base + size.
                const bool overlap = x.size != 0 && y.size != 0 &&
                                     (x.base < y.base ? y.base - x.base < x.size : x.base - y.base < y.size);
                if (overlap) {
                    findings.push_back(
                        {ManifestCheck::Region,
                         fmt::format(
                             "{} overlaps {}",
                             region_label(*placed[i].device, placed[i].index),
                             region_label(*placed[j].device, placed[j].index))});
                }
            }
        }
    }
}

// Row 5.
void check_slots_have_regions(const Manifest& manifest, Findings& findings) {
    for (const Device& device : manifest.devices) {
        std::set<CceId> owned;
        for (const Region& region : device.regions) {
            owned.insert(CceId{region.tile, region.cce});
        }
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (!owned.contains(CceId{slot.tile, slot.cce})) {
                findings.push_back(
                    {ManifestCheck::SlotWithoutRegion,
                     fmt::format(
                         "{}: device {} owns no region of {}",
                         slot_label(device, i),
                         device.name,
                         cce_label(slot.tile, slot.cce))});
            }
        }
    }
}

// Row 6.
void check_worker_placement(const Manifest& manifest, Findings& findings) {
    struct Roles {
        bool worker = false;
        bool router_on_hart0 = false;
    };
    std::map<CceId, Roles> by_cce;
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (slot.role == SlotRole::Worker && slot.hart == kRouterHart) {
                findings.push_back(
                    {ManifestCheck::WorkerPlacement,
                     fmt::format("{}: a worker is on hart {}", slot_label(device, i), kRouterHart)});
            }
            Roles& roles = by_cce[CceId{slot.tile, slot.cce}];
            roles.worker |= slot.role == SlotRole::Worker;
            roles.router_on_hart0 |= slot.role == SlotRole::Router && slot.hart == kRouterHart;
        }
    }
    for (const auto& [cce, roles] : by_cce) {
        if (roles.worker && !roles.router_on_hart0) {
            findings.push_back(
                {ManifestCheck::WorkerPlacement,
                 fmt::format("{} has a worker and no router on hart {}", cce_label(cce.tile, cce.cce), kRouterHart)});
        }
    }
}

// Row 7.
void check_links(const Manifest& manifest, Findings& findings) {
    std::set<std::pair<std::string, uint32_t>> uses;
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        const Link& link = manifest.links[i];
        const auto check_end = [&](char side, const LinkEnd& end) {
            if (find_device(manifest, end.device) == nullptr) {
                findings.push_back({ManifestCheck::Link, fmt::format("{}: no such device", end_label(i, side, end))});
                return;
            }
            if (find_router(manifest, end) == nullptr) {
                findings.push_back(
                    {ManifestCheck::Link,
                     fmt::format("{}: the device has no router on that port", end_label(i, side, end))});
            }
            if (!uses.insert({end.device, end.port}).second) {
                findings.push_back(
                    {ManifestCheck::Link,
                     fmt::format("{}: the port is in another link end too", end_label(i, side, end))});
            }
        };
        check_end('a', link.a);
        check_end('b', link.b);
        if (link.a.device == link.b.device) {
            findings.push_back(
                {ManifestCheck::Link, fmt::format("links[{}]: device {} is at both ends", i, link.a.device)});
        }
    }
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (slot.role == SlotRole::Router && slot.port && !uses.contains({device.name, *slot.port})) {
                findings.push_back(
                    {ManifestCheck::Link,
                     fmt::format("{}: router port {} is in no link", slot_label(device, i), *slot.port)});
            }
        }
    }
}

// Row 8.
void check_ethernet_pairs(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        const Link& link = manifest.links[i];
        if (link.backend != LinkBackend::Ethernet) {
            continue;
        }
        const Slot* a = find_router(manifest, link.a);
        const Slot* b = find_router(manifest, link.b);
        if (a == nullptr || b == nullptr) {
            continue;  // row 7 reports the link
        }
        const bool wired =
            std::any_of(facts.ethernet_tile_pairs.begin(), facts.ethernet_tile_pairs.end(), [&](const auto& pair) {
                return (pair.first == a->tile && pair.second == b->tile) ||
                       (pair.first == b->tile && pair.second == a->tile);
            });
        if (!wired) {
            findings.push_back(
                {ManifestCheck::EthernetTilesNotWired,
                 fmt::format(
                     "links[{}]: tiles {} and {} are not an Ethernet pair the bring-up wired", i, a->tile, b->tile)});
        }
    }
}

// Row 13.
void check_devices_have_router(const Manifest& manifest, Findings& findings) {
    for (const Device& device : manifest.devices) {
        if (std::none_of(
                device.slots.begin(), device.slots.end(), [](const Slot& s) { return s.role == SlotRole::Router; })) {
            findings.push_back(
                {ManifestCheck::DeviceWithoutRouter, fmt::format("device {} has no router slot", device.name)});
        }
    }
}

// Sorts a stage's findings into row order, keeping each row's findings in the order they were found.
Findings in_row_order(Findings findings) {
    std::stable_sort(findings.begin(), findings.end(), [](const ManifestFinding& x, const ManifestFinding& y) {
        return x.check < y.check;
    });
    return findings;
}

}  // namespace

Manifest parse_manifest(std::string_view yaml_text) {
    std::vector<YAML::Node> documents;
    try {
        documents = YAML::LoadAll(std::string(yaml_text));
    } catch (const YAML::Exception& e) {
        throw ManifestLoadError({fmt::format("not YAML: {}", e.what())});
    }
    if (documents.size() != 1) {
        throw ManifestLoadError({fmt::format("a manifest is one YAML document; this one has {}", documents.size())});
    }
    const YAML::Node& root = documents.front();
    if (!root.IsMap()) {
        throw ManifestLoadError({"the root is not a map"});
    }
    // An unknown version says nothing about the rest of the file, so it is the only error reported.
    const YAML::Node version = root["version"];
    const std::optional<uint32_t> version_value = version ? decode_scalar<uint32_t>(version) : std::nullopt;
    if (version_value != kManifestVersion) {
        throw ManifestLoadError({fmt::format(
            "version {} is not {}", version && version.IsScalar() ? version.Scalar() : "<missing>", kManifestVersion)});
    }
    Errors errors;
    Manifest manifest;
    manifest.version = *version_value;
    check_keys(root, {"version", "capture", "devices", "links"}, "manifest", errors);
    if (root["capture"]) {
        if (const auto capture = read_value<std::string>(root, "capture", "manifest", errors)) {
            if (capture->empty()) {
                errors.push_back("manifest: 'capture' is empty");
            } else {
                manifest.capture = *capture;
            }
        }
    }
    const YAML::Node devices = read_sequence(root, "devices", "manifest", errors);
    if (devices.IsSequence() && devices.size() == 0) {
        errors.push_back("manifest: 'devices' is empty");
    }
    for (std::size_t i = 0; i < devices.size(); i++) {
        if (auto device = parse_device(devices[i], fmt::format("devices[{}]", i), errors)) {
            manifest.devices.push_back(std::move(*device));
        }
    }
    const YAML::Node links = read_sequence(root, "links", "manifest", errors);
    for (std::size_t i = 0; i < links.size(); i++) {
        if (auto link = parse_link(links[i], fmt::format("links[{}]", i), errors)) {
            manifest.links.push_back(std::move(*link));
        }
    }
    if (!errors.empty()) {
        throw ManifestLoadError(std::move(errors));
    }
    return manifest;
}

Manifest load_manifest(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw ManifestLoadError({fmt::format("cannot open {}", path.string())});
    }
    std::stringstream text;
    text << file.rdbuf();
    if (file.bad()) {
        throw ManifestLoadError({fmt::format("cannot read {}", path.string())});
    }
    Manifest manifest = parse_manifest(text.str());
    if (manifest.capture && manifest.capture->is_relative()) {
        manifest.capture = path.parent_path() / *manifest.capture;
    }
    return manifest;
}

std::vector<ManifestFinding> check_manifest(const Manifest& manifest, const PackageFacts& facts) {
    TT_FATAL(
        facts.harts_per_cce > 0, "PackageFacts::harts_per_cce is {}; the caller must fill it", facts.harts_per_cce);
    TT_FATAL(
        facts.sram_bytes_per_cce > 0,
        "PackageFacts::sram_bytes_per_cce is {}; the caller must fill it",
        facts.sram_bytes_per_cce);
    TT_FATAL(!facts.cores.empty(), "PackageFacts::cores is empty; the caller must fill it");
    TT_FATAL(!manifest.devices.empty(), "Manifest::devices is empty; parse_manifest() never returns one");
    Findings findings;
    // Stage A.
    check_reserved_values(manifest, facts, findings);
    if (!findings.empty()) {
        return in_row_order(std::move(findings));
    }
    // Stage B.
    check_cces_in_package(manifest, facts, findings);
    check_duplicates(manifest, findings);
    if (!findings.empty()) {
        return in_row_order(std::move(findings));
    }
    // Stage C.
    check_slots(manifest, facts, findings);
    check_regions(manifest, facts, findings);
    check_slots_have_regions(manifest, findings);
    check_worker_placement(manifest, findings);
    check_links(manifest, findings);
    check_ethernet_pairs(manifest, facts, findings);
    check_devices_have_router(manifest, findings);
    return in_row_order(std::move(findings));
}

}  // namespace tt::tt_fabric::vcluster
