// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "manifest.hpp"

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

namespace tt::tt_fabric::vcluster {

namespace {

constexpr uint32_t kManifestVersion = 1;
// Phase 2 values (SPEC-0011 § Manifest): routers on hart 0 of CCE0; CCE1 is reserved for Phase 3.
constexpr uint32_t kRouterHart = 0;
constexpr uint32_t kReservedCce = 1;

using Findings = std::vector<ManifestFinding>;

void add(Findings& findings, ManifestCheck check, std::string detail) {
    findings.push_back(ManifestFinding{check, std::move(detail)});
}

void format_finding(Findings& findings, std::string detail) { add(findings, ManifestCheck::Format, std::move(detail)); }

// Reports every key of the map `node` that is not in `known`.
void check_keys(
    const YAML::Node& node,
    std::initializer_list<std::string_view> known,
    const std::string& where,
    Findings& findings) {
    for (const auto& entry : node) {
        const std::string key = entry.first.IsScalar() ? entry.first.Scalar() : std::string("<not a scalar>");
        if (std::find(known.begin(), known.end(), key) == known.end()) {
            format_finding(findings, fmt::format("{}: unknown key '{}'", where, key));
        }
    }
}

// The value of `key` in the map `node` as a T, or empty with a finding when it is missing or does not convert.
template <typename T>
std::optional<T> read_value(const YAML::Node& node, const char* key, const std::string& where, Findings& findings) {
    const YAML::Node value = node[key];
    if (!value) {
        format_finding(findings, fmt::format("{}: missing key '{}'", where, key));
        return std::nullopt;
    }
    T out{};
    if (!value.IsScalar() || !YAML::convert<T>::decode(value, out)) {
        format_finding(
            findings,
            fmt::format("{}: '{}' is not a valid {}", where, key, value.IsScalar() ? value.Scalar() : "scalar"));
        return std::nullopt;
    }
    return out;
}

// The sequence at `key` in `node`, or an empty node with a finding when it is missing or not a sequence.
YAML::Node read_sequence(const YAML::Node& node, const char* key, const std::string& where, Findings& findings) {
    const YAML::Node value = node[key];
    if (!value) {
        format_finding(findings, fmt::format("{}: missing key '{}'", where, key));
        return YAML::Node();
    }
    if (!value.IsSequence()) {
        format_finding(findings, fmt::format("{}: '{}' is not a sequence", where, key));
        return YAML::Node();
    }
    return value;
}

bool is_map(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!node.IsMap()) {
        format_finding(findings, fmt::format("{}: not a map", where));
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

std::optional<Slot> parse_slot(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!is_map(node, where, findings)) {
        return std::nullopt;
    }
    check_keys(node, {"tile", "cce", "hart", "role", "port"}, where, findings);
    const auto tile = read_value<uint32_t>(node, "tile", where, findings);
    const auto cce = read_value<uint32_t>(node, "cce", where, findings);
    const auto hart = read_value<uint32_t>(node, "hart", where, findings);
    const auto role_text = read_value<std::string>(node, "role", where, findings);
    std::optional<SlotRole> role;
    if (role_text) {
        role = parse_role(*role_text);
        if (!role) {
            format_finding(findings, fmt::format("{}: unknown role '{}'", where, *role_text));
        }
    }
    std::optional<uint32_t> port;
    if (node["port"]) {
        port = read_value<uint32_t>(node, "port", where, findings);
    }
    if (!tile || !cce || !hart || !role) {
        return std::nullopt;
    }
    if (*role == SlotRole::Worker && node["port"]) {
        format_finding(findings, fmt::format("{}: a worker has no port", where));
        port.reset();
    }
    if (*role == SlotRole::Router && !node["port"]) {
        format_finding(findings, fmt::format("{}: missing key 'port' (required for a router)", where));
    }
    return Slot{.tile = *tile, .cce = *cce, .hart = *hart, .role = *role, .port = port};
}

std::optional<Region> parse_region(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!is_map(node, where, findings)) {
        return std::nullopt;
    }
    check_keys(node, {"tile", "cce", "base", "size"}, where, findings);
    const auto tile = read_value<uint32_t>(node, "tile", where, findings);
    const auto cce = read_value<uint32_t>(node, "cce", where, findings);
    const auto base = read_value<uint64_t>(node, "base", where, findings);
    const auto size = read_value<uint64_t>(node, "size", where, findings);
    if (!tile || !cce || !base || !size) {
        return std::nullopt;
    }
    return Region{.tile = *tile, .cce = *cce, .base = *base, .size = *size};
}

std::optional<Device> parse_device(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!is_map(node, where, findings)) {
        return std::nullopt;
    }
    check_keys(node, {"name", "fabric_chip", "slots", "regions"}, where, findings);
    const auto name = read_value<std::string>(node, "name", where, findings);
    const auto fabric_chip = read_value<uint32_t>(node, "fabric_chip", where, findings);
    const std::string label = name ? fmt::format("{} ({})", where, *name) : where;
    Device device;
    const YAML::Node slots = read_sequence(node, "slots", label, findings);
    for (std::size_t i = 0; i < slots.size(); i++) {
        if (auto slot = parse_slot(slots[i], fmt::format("{}.slots[{}]", label, i), findings)) {
            device.slots.push_back(*slot);
        }
    }
    const YAML::Node regions = read_sequence(node, "regions", label, findings);
    for (std::size_t i = 0; i < regions.size(); i++) {
        if (auto region = parse_region(regions[i], fmt::format("{}.regions[{}]", label, i), findings)) {
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

std::optional<LinkEnd> parse_link_end(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!node) {
        format_finding(findings, fmt::format("{}: missing", where));
        return std::nullopt;
    }
    if (!is_map(node, where, findings)) {
        return std::nullopt;
    }
    check_keys(node, {"device", "port"}, where, findings);
    const auto device = read_value<std::string>(node, "device", where, findings);
    const auto port = read_value<uint32_t>(node, "port", where, findings);
    if (!device || !port) {
        return std::nullopt;
    }
    return LinkEnd{.device = *device, .port = *port};
}

std::optional<Link> parse_link(const YAML::Node& node, const std::string& where, Findings& findings) {
    if (!is_map(node, where, findings)) {
        return std::nullopt;
    }
    check_keys(node, {"a", "b", "backend"}, where, findings);
    const auto a = parse_link_end(node["a"], where + ".a", findings);
    const auto b = parse_link_end(node["b"], where + ".b", findings);
    const auto backend_text = read_value<std::string>(node, "backend", where, findings);
    std::optional<LinkBackend> backend;
    if (backend_text) {
        backend = parse_backend(*backend_text);
        if (!backend) {
            format_finding(findings, fmt::format("{}: unknown backend '{}'", where, *backend_text));
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

const Device* find_device(const Manifest& manifest, const std::string& name) {
    const auto it =
        std::find_if(manifest.devices.begin(), manifest.devices.end(), [&](const Device& d) { return d.name == name; });
    return it == manifest.devices.end() ? nullptr : &*it;
}

// The router slot of `end`, or null when its device or router port does not exist.
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

// Row 1, the value rules.
void check_values(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    std::map<std::string, std::size_t> names;
    std::map<uint32_t, std::size_t> chips;
    for (const Device& device : manifest.devices) {
        if (names[device.name]++ > 0) {
            format_finding(findings, fmt::format("device name {} is used more than once", device.name));
        }
        if (chips[device.fabric_chip]++ > 0) {
            format_finding(findings, fmt::format("fabric_chip {} is used more than once", device.fabric_chip));
        }
        std::map<uint32_t, std::size_t> ports;
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (slot.cce == kReservedCce) {
                format_finding(
                    findings, fmt::format("{}: cce {} is reserved for Phase 3", slot_label(device, i), slot.cce));
            }
            if (slot.role == SlotRole::Router && slot.hart != kRouterHart) {
                format_finding(
                    findings,
                    fmt::format(
                        "{}: a router on a hart other than {} is reserved for Phase 3",
                        slot_label(device, i),
                        kRouterHart));
            }
            if (slot.role == SlotRole::Router && slot.port && ports[*slot.port]++ > 0) {
                format_finding(findings, fmt::format("device {}: two routers on port {}", device.name, *slot.port));
            }
        }
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            if (region.base != 0 || region.size != facts.sram_bytes_per_cce) {
                format_finding(
                    findings,
                    fmt::format(
                        "{}: a region other than the whole CCE (base 0, size {:#x}) is reserved for Phase 3",
                        region_label(device, i),
                        facts.sram_bytes_per_cce));
            }
        }
    }
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        if (manifest.links[i].backend == LinkBackend::Local) {
            format_finding(findings, fmt::format("links[{}]: backend local is reserved for Phase 3", i));
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
                add(findings,
                    ManifestCheck::CceNotInPackage,
                    fmt::format(
                        "{}: {} is not a Keraunos core of the package",
                        slot_label(device, i),
                        cce_label(slot.tile, slot.cce)));
            }
        }
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            if (!cores.contains(CceId{region.tile, region.cce})) {
                add(findings,
                    ManifestCheck::CceNotInPackage,
                    fmt::format(
                        "{}: {} is not a Keraunos core of the package",
                        region_label(device, i),
                        cce_label(region.tile, region.cce)));
            }
        }
    }
}

// Row 3.
void check_slots(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::size_t> harts;
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (harts[{slot.tile, slot.cce, slot.hart}]++ > 0) {
                add(findings,
                    ManifestCheck::Slot,
                    fmt::format(
                        "two slots on (tile {}, cce {}, hart {}), the second in device {}",
                        slot.tile,
                        slot.cce,
                        slot.hart,
                        device.name));
            }
            if (slot.hart >= facts.harts_per_cce) {
                add(findings,
                    ManifestCheck::Slot,
                    fmt::format(
                        "{}: hart {} is not below the CCE's {} harts",
                        slot_label(device, i),
                        slot.hart,
                        facts.harts_per_cce));
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
            if (region.base > facts.sram_bytes_per_cce || region.size > facts.sram_bytes_per_cce - region.base) {
                add(findings,
                    ManifestCheck::Region,
                    fmt::format(
                        "{}: leaves the CCE's {:#x} bytes of SRAM", region_label(device, i), facts.sram_bytes_per_cce));
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
                    add(findings,
                        ManifestCheck::Region,
                        fmt::format(
                            "{} overlaps {}",
                            region_label(*placed[i].device, placed[i].index),
                            region_label(*placed[j].device, placed[j].index)));
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
                add(findings,
                    ManifestCheck::SlotWithoutRegion,
                    fmt::format(
                        "{}: device {} owns no region of {}",
                        slot_label(device, i),
                        device.name,
                        cce_label(slot.tile, slot.cce)));
            }
        }
    }
}

// Row 6.
void check_workers_have_router(const Manifest& manifest, Findings& findings) {
    struct Roles {
        bool worker = false;
        bool router_on_hart0 = false;
    };
    std::map<CceId, Roles> by_cce;
    for (const Device& device : manifest.devices) {
        for (const Slot& slot : device.slots) {
            Roles& roles = by_cce[CceId{slot.tile, slot.cce}];
            roles.worker |= slot.role == SlotRole::Worker;
            roles.router_on_hart0 |= slot.role == SlotRole::Router && slot.hart == kRouterHart;
        }
    }
    for (const auto& [cce, roles] : by_cce) {
        if (roles.worker && !roles.router_on_hart0) {
            add(findings,
                ManifestCheck::WorkerWithoutRouter,
                fmt::format("{} has a worker and no router on hart {}", cce_label(cce.tile, cce.cce), kRouterHart));
        }
    }
}

// Row 7.
void check_links(const Manifest& manifest, Findings& findings) {
    std::map<std::pair<std::string, uint32_t>, std::size_t> uses;
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        const Link& link = manifest.links[i];
        const auto check_end = [&](char side, const LinkEnd& end) {
            if (find_device(manifest, end.device) == nullptr) {
                add(findings, ManifestCheck::Link, fmt::format("{}: no such device", end_label(i, side, end)));
            } else if (find_router(manifest, end) == nullptr) {
                add(findings,
                    ManifestCheck::Link,
                    fmt::format("{}: the device has no router on that port", end_label(i, side, end)));
            }
            if (uses[{end.device, end.port}]++ > 0) {
                add(findings,
                    ManifestCheck::Link,
                    fmt::format("{}: the port is in another link end too", end_label(i, side, end)));
            }
        };
        check_end('a', link.a);
        check_end('b', link.b);
        if (link.a.device == link.b.device) {
            add(findings, ManifestCheck::Link, fmt::format("links[{}]: device {} is at both ends", i, link.a.device));
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
            continue;  // row 7 reports it
        }
        const bool wired =
            std::any_of(facts.ethernet_tile_pairs.begin(), facts.ethernet_tile_pairs.end(), [&](const auto& p) {
                return (p.first == a->tile && p.second == b->tile) || (p.first == b->tile && p.second == a->tile);
            });
        if (!wired) {
            add(findings,
                ManifestCheck::EthernetTilesNotWired,
                fmt::format(
                    "links[{}]: tiles {} and {} are not an Ethernet pair the bring-up wired", i, a->tile, b->tile));
        }
    }
}

// Row 10, the manifest half.
void check_router_ports_linked(const Manifest& manifest, Findings& findings) {
    std::set<std::pair<std::string, uint32_t>> linked;
    for (const Link& link : manifest.links) {
        linked.insert({link.a.device, link.a.port});
        linked.insert({link.b.device, link.b.port});
    }
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.slots.size(); i++) {
            const Slot& slot = device.slots[i];
            if (slot.role == SlotRole::Router && slot.port && !linked.contains({device.name, *slot.port})) {
                add(findings,
                    ManifestCheck::RouterPortWithoutLink,
                    fmt::format("{}: port {} is in no link", slot_label(device, i), *slot.port));
            }
        }
    }
}

}  // namespace

ManifestParse parse_manifest(std::string_view yaml_text) {
    YAML::Node root;
    try {
        root = YAML::Load(std::string(yaml_text));
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("manifest is not YAML: {}", e.what()));
    }
    ManifestParse parse;
    Findings& findings = parse.findings;
    if (!root.IsMap()) {
        format_finding(findings, "manifest: the root is not a map");
        return parse;
    }
    const YAML::Node version = root["version"];
    uint32_t version_value = 0;
    if (!version || !version.IsScalar() || !YAML::convert<uint32_t>::decode(version, version_value) ||
        version_value != kManifestVersion) {
        format_finding(
            findings,
            fmt::format(
                "manifest: version {} is not {}",
                version && version.IsScalar() ? version.Scalar() : "<missing>",
                kManifestVersion));
        return parse;
    }
    Manifest& manifest = parse.manifest;
    manifest.version = version_value;
    check_keys(root, {"version", "capture", "devices", "links"}, "manifest", findings);
    if (const auto capture = read_value<std::string>(root, "capture", "manifest", findings)) {
        manifest.capture = *capture;
    }
    const YAML::Node devices = read_sequence(root, "devices", "manifest", findings);
    for (std::size_t i = 0; i < devices.size(); i++) {
        if (auto device = parse_device(devices[i], fmt::format("devices[{}]", i), findings)) {
            manifest.devices.push_back(std::move(*device));
        }
    }
    const YAML::Node links = read_sequence(root, "links", "manifest", findings);
    for (std::size_t i = 0; i < links.size(); i++) {
        if (auto link = parse_link(links[i], fmt::format("links[{}]", i), findings)) {
            manifest.links.push_back(std::move(*link));
        }
    }
    return parse;
}

ManifestParse load_manifest(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(fmt::format("cannot open manifest {}", path.string()));
    }
    std::stringstream text;
    text << file.rdbuf();
    if (file.bad()) {
        throw std::runtime_error(fmt::format("cannot read manifest {}", path.string()));
    }
    return parse_manifest(text.str());
}

std::vector<ManifestFinding> check_manifest(const Manifest& manifest, const PackageFacts& facts) {
    Findings findings;
    check_values(manifest, facts, findings);
    check_cces_in_package(manifest, facts, findings);
    check_slots(manifest, facts, findings);
    check_regions(manifest, facts, findings);
    check_slots_have_regions(manifest, findings);
    check_workers_have_router(manifest, findings);
    check_links(manifest, findings);
    check_ethernet_pairs(manifest, facts, findings);
    check_router_ports_linked(manifest, findings);
    return findings;
}

}  // namespace tt::tt_fabric::vcluster
