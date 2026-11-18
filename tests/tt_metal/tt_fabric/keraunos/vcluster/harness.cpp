// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "harness.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::vcluster {

namespace {

using Findings = std::vector<HarnessFinding>;

constexpr uint64_t kMaxWord = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kUnreachable = std::numeric_limits<uint64_t>::max();

void add(Findings& findings, HarnessCheck check, std::string detail) {
    findings.push_back(HarnessFinding{check, std::move(detail)});
}

const Device& device_named(const Manifest& manifest, const std::string& name) {
    const auto it =
        std::find_if(manifest.devices.begin(), manifest.devices.end(), [&](const Device& d) { return d.name == name; });
    TT_FATAL(it != manifest.devices.end(), "A link names device {}, which check_manifest() refuses (row 7)", name);
    return *it;
}

std::size_t index_of(const Manifest& manifest, const std::string& name) {
    return &device_named(manifest, name) - manifest.devices.data();
}

// The device's router slot. check_manifest() row 13 guarantees one.
const Slot& router_slot(const Device& device) {
    const auto it = std::find_if(
        device.slots.begin(), device.slots.end(), [](const Slot& s) { return s.role == SlotRole::Router; });
    TT_FATAL(it != device.slots.end(), "Device {} has no router slot, which check_manifest() refuses (row 13)", device.name);
    return *it;
}

// The port numbers the link ends naming each device name, by device index, in link order.
std::vector<std::vector<uint32_t>> ports_named(const Manifest& manifest) {
    std::vector<std::vector<uint32_t>> ports(manifest.devices.size());
    for (const Link& link : manifest.links) {
        ports[index_of(manifest, link.a.device)].push_back(link.a.port);
        ports[index_of(manifest, link.b.device)].push_back(link.b.port);
    }
    return ports;
}

// a * b, or the 64-bit maximum when that overflows.
uint64_t saturating_mul(uint64_t a, uint64_t b) {
    uint64_t out = 0;
    return __builtin_mul_overflow(a, b, &out) ? std::numeric_limits<uint64_t>::max() : out;
}

uint64_t ring_slots_for(const RunShape& shape, uint64_t num_devices) {
    return saturating_mul(saturating_mul(shape.packets_per_flow, num_devices), num_devices - 1);
}

// H1.
void check_one_slot(const Manifest& manifest, Findings& findings) {
    for (const Device& device : manifest.devices) {
        if (device.slots.size() != 1) {
            add(findings,
                HarnessCheck::SlotNotOne,
                fmt::format("device {} has {} slots; a harness device has one", device.name, device.slots.size()));
        } else if (device.slots[0].port.value_or(0) != 0) {
            add(findings,
                HarnessCheck::SlotNotOne,
                fmt::format("device {}: the slot's port is {}, not 0", device.name, *device.slots[0].port));
        }
    }
}

// H2.
void check_one_region(const Manifest& manifest, Findings& findings) {
    for (const Device& device : manifest.devices) {
        if (device.regions.size() != 1) {
            add(findings,
                HarnessCheck::RegionNotOne,
                fmt::format("device {} has {} regions; a harness device has one", device.name, device.regions.size()));
        }
    }
}

// H3.
void check_region_placement(const Manifest& manifest, const PackageFacts& facts, Findings& findings) {
    for (const Device& device : manifest.devices) {
        for (std::size_t i = 0; i < device.regions.size(); i++) {
            const Region& region = device.regions[i];
            if (region.base < facts.unreserved_base) {
                add(findings,
                    HarnessCheck::RegionPlacement,
                    fmt::format(
                        "device {} region {}: base {:#x} is below the unreserved base {:#x}",
                        device.name,
                        i,
                        region.base,
                        facts.unreserved_base));
            }
            if (region.base % vc_node::kAlign != 0 || region.size % vc_node::kAlign != 0) {
                add(findings,
                    HarnessCheck::RegionPlacement,
                    fmt::format(
                        "device {} region {}: base {:#x} or size {:#x} is not a multiple of {}",
                        device.name,
                        i,
                        region.base,
                        region.size,
                        vc_node::kAlign));
            }
        }
    }
}

// H4.
void check_no_ethernet(const Manifest& manifest, Findings& findings) {
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        if (manifest.links[i].backend == LinkBackend::Ethernet) {
            add(findings, HarnessCheck::EthernetLink, fmt::format("links[{}]: backend ethernet", i));
        }
    }
}

// H5.
void check_link_placement(const Manifest& manifest, Findings& findings) {
    for (std::size_t i = 0; i < manifest.links.size(); i++) {
        const Link& link = manifest.links[i];
        const Slot& a = router_slot(device_named(manifest, link.a.device));
        const Slot& b = router_slot(device_named(manifest, link.b.device));
        if (link.backend == LinkBackend::Local && (a.tile != b.tile || a.cce != b.cce)) {
            add(findings,
                HarnessCheck::LinkPlacement,
                fmt::format(
                    "links[{}]: local, but {} is on (tile {}, cce {}) and {} on (tile {}, cce {})",
                    i,
                    link.a.device,
                    a.tile,
                    a.cce,
                    link.b.device,
                    b.tile,
                    b.cce));
        }
        if (link.backend == LinkBackend::Noc && a.tile == b.tile) {
            add(findings,
                HarnessCheck::LinkPlacement,
                fmt::format("links[{}]: noc, but {} and {} are both on tile {}", i, link.a.device, link.b.device, a.tile));
        }
    }
}

// H6.
void check_ports(const Manifest& manifest, Findings& findings) {
    const auto ports = ports_named(manifest);
    const uint32_t num_devices = manifest.devices.size();
    std::map<CceId, uint64_t> words_by_cce;
    for (std::size_t d = 0; d < manifest.devices.size(); d++) {
        const Device& device = manifest.devices[d];
        std::vector<uint32_t> sorted = ports[d];
        std::sort(sorted.begin(), sorted.end());
        bool dense = true;
        for (std::size_t p = 0; p < sorted.size(); p++) {
            dense &= sorted[p] == p;
        }
        if (!dense) {
            add(findings,
                HarnessCheck::Ports,
                fmt::format(
                    "device {}: its link ends name ports [{}]; they must be 0 to k-1, each once",
                    device.name,
                    fmt::join(sorted, ", ")));
        }
        if (sorted.size() > vc_node::kMaxPorts) {
            add(findings,
                HarnessCheck::Ports,
                fmt::format("device {}: {} ports; at most {}", device.name, sorted.size(), vc_node::kMaxPorts));
        }
        const Slot& slot = router_slot(device);
        words_by_cce[CceId{slot.tile, slot.cce}] +=
            vc_node::kArgNextPortBase + uint64_t{num_devices} + uint64_t{vc_node::kArgsPerPort} * sorted.size();
    }
    for (const auto& [cce, words] : words_by_cce) {
        if (words > kMaxArgWordsPerCce) {
            add(findings,
                HarnessCheck::Ports,
                fmt::format(
                    "(tile {}, cce {}): its nodes need {} runtime-argument words; at most {}",
                    cce.tile,
                    cce.cce,
                    words,
                    kMaxArgWordsPerCce));
        }
    }
}

// H7.
void check_connected(const Manifest& manifest, Findings& findings) {
    const std::size_t n = manifest.devices.size();
    std::vector<std::vector<std::size_t>> adjacent(n);
    for (const Link& link : manifest.links) {
        const std::size_t a = index_of(manifest, link.a.device);
        const std::size_t b = index_of(manifest, link.b.device);
        adjacent[a].push_back(b);
        adjacent[b].push_back(a);
    }
    std::vector<bool> seen(n, false);
    std::deque<std::size_t> queue{0};
    seen[0] = true;
    while (!queue.empty()) {
        const std::size_t u = queue.front();
        queue.pop_front();
        for (std::size_t v : adjacent[u]) {
            if (!seen[v]) {
                seen[v] = true;
                queue.push_back(v);
            }
        }
    }
    for (std::size_t d = 0; d < n; d++) {
        if (!seen[d]) {
            add(findings,
                HarnessCheck::Disconnected,
                fmt::format(
                    "device {} cannot be reached from device {}", manifest.devices[d].name, manifest.devices[0].name));
        }
    }
}

// H8.
void check_region_size(const Manifest& manifest, const RunShape& shape, Findings& findings) {
    const auto ports = ports_named(manifest);
    const uint64_t ring_slots = ring_slots_for(shape, manifest.devices.size());
    const uint64_t slot_bytes = vc_node::layout(0, 0, shape.payload_bytes).slot_bytes;
    for (std::size_t d = 0; d < manifest.devices.size(); d++) {
        const Device& device = manifest.devices[d];
        const uint64_t num_ports = ports[d].size();
        if (device.regions.size() != 1 || num_ports > vc_node::kMaxPorts) {
            continue;  // H2 or H6 reports the device
        }
        const Region& region = device.regions[0];
        // Bounds ring_slots * slot_bytes by the region before layout() forms it.
        if (ring_slots > kMaxWord || ring_slots > region.size / slot_bytes) {
            add(findings,
                HarnessCheck::RegionTooSmall,
                fmt::format(
                    "device {}: {} ring slots of {} bytes do not fit its region of {:#x} bytes",
                    device.name,
                    ring_slots,
                    slot_bytes,
                    region.size));
            continue;
        }
        const uint64_t total =
            vc_node::layout(num_ports, static_cast<uint32_t>(ring_slots), shape.payload_bytes).total;
        if (total > region.size) {
            add(findings,
                HarnessCheck::RegionTooSmall,
                fmt::format(
                    "device {}: the node's layout needs {:#x} bytes; its region is {:#x}", device.name, total, region.size));
        }
    }
}

// The asserts SPEC-0012 § Manifest names: what check_manifest() has already refused.
void assert_checked(const Manifest& manifest) {
    for (const Device& device : manifest.devices) {
        router_slot(device);
        for (const Slot& slot : device.slots) {
            const bool has_region = std::any_of(device.regions.begin(), device.regions.end(), [&](const Region& r) {
                return r.tile == slot.tile && r.cce == slot.cce;
            });
            TT_FATAL(
                has_region,
                "Device {}: a slot on (tile {}, cce {}) has no region there, which check_manifest() refuses (row 5)",
                device.name,
                slot.tile,
                slot.cce);
        }
    }
    for (const Link& link : manifest.links) {
        device_named(manifest, link.a.device);
        device_named(manifest, link.b.device);
    }
}

}  // namespace

std::vector<HarnessFinding> check_harness_manifest(
    const Manifest& manifest, const PackageFacts& facts, const RunShape& shape) {
    TT_FATAL(manifest.router == RouterKind::Harness, "check_harness_manifest() takes a manifest with router: harness");
    TT_FATAL(!manifest.devices.empty(), "Manifest::devices is empty; parse_manifest() never returns one");
    TT_FATAL(
        facts.harts_per_cce > 0, "PackageFacts::harts_per_cce is {}; the caller must fill it", facts.harts_per_cce);
    TT_FATAL(
        facts.sram_bytes_per_cce > 0 && facts.sram_bytes_per_cce <= kMaxWord + 1,
        "PackageFacts::sram_bytes_per_cce is {:#x}; a node takes 32-bit SRAM offsets",
        facts.sram_bytes_per_cce);
    TT_FATAL(!facts.cores.empty(), "PackageFacts::cores is empty; the caller must fill it");
    TT_FATAL(facts.unreserved_base > 0, "PackageFacts::unreserved_base is 0; the caller must fill it");
    TT_FATAL(shape.packets_per_flow > 0, "RunShape::packets_per_flow is 0");
    TT_FATAL(
        shape.payload_bytes % vc_node::kPayloadGranule == 0,
        "RunShape::payload_bytes {} is not a multiple of {}",
        shape.payload_bytes,
        vc_node::kPayloadGranule);
    assert_checked(manifest);

    Findings findings;
    check_one_slot(manifest, findings);
    check_one_region(manifest, findings);
    check_region_placement(manifest, facts, findings);
    check_no_ethernet(manifest, findings);
    check_link_placement(manifest, findings);
    check_ports(manifest, findings);
    check_connected(manifest, findings);
    check_region_size(manifest, shape, findings);
    return findings;
}

HarnessPlan plan_harness(const Manifest& manifest, const RunShape& shape) {
    const uint32_t n = manifest.devices.size();
    HarnessPlan plan;
    plan.shape = shape;
    const uint64_t ring_slots = ring_slots_for(shape, n);
    TT_FATAL(ring_slots <= kMaxWord, "{} ring slots do not fit a runtime-argument word; H8 refuses it", ring_slots);
    plan.ring_slots = static_cast<uint32_t>(ring_slots);
    plan.nodes.resize(n);

    std::vector<bool> placed(n, false);
    const auto ports = ports_named(manifest);
    for (std::size_t d = 0; d < n; d++) {
        const Device& device = manifest.devices[d];
        TT_FATAL(
            device.fabric_chip < n && !placed[device.fabric_chip],
            "Device {}: fabric_chip {} is not one of 0 to {} or is taken; the node indexes routes by it",
            device.name,
            device.fabric_chip,
            n - 1);
        placed[device.fabric_chip] = true;
        TT_FATAL(
            device.slots.size() == 1 && device.regions.size() == 1 && ports[d].size() <= vc_node::kMaxPorts,
            "Device {}: H1, H2 or H6 refuses it",
            device.name);
        PlannedNode& node = plan.nodes[device.fabric_chip];
        node.name = device.name;
        node.device = device.fabric_chip;
        node.tile = device.slots[0].tile;
        node.hart = device.slots[0].hart;
        node.region_base = device.regions[0].base;
        node.ports.resize(ports[d].size());
        node.predicted_sent.assign(ports[d].size(), 0);
        node.predicted_received.assign(ports[d].size(), 0);
        node.layout = vc_node::layout(ports[d].size(), plan.ring_slots, shape.payload_bytes);
    }
    for (const Link& link : manifest.links) {
        TT_FATAL(link.backend != LinkBackend::Ethernet, "An ethernet link; H4 refuses it");
        const uint32_t a = device_named(manifest, link.a.device).fabric_chip;
        const uint32_t b = device_named(manifest, link.b.device).fabric_chip;
        plan.nodes[a].ports.at(link.a.port) = PlannedPort{link.backend, b, link.b.port};
        plan.nodes[b].ports.at(link.b.port) = PlannedPort{link.backend, a, link.a.port};
    }

    // next_port[d] at u: the lowest port of u whose peer is one hop nearer d.
    for (uint32_t d = 0; d < n; d++) {
        std::vector<uint64_t> distance(n, kUnreachable);
        std::deque<uint32_t> queue{d};
        distance[d] = 0;
        while (!queue.empty()) {
            const uint32_t u = queue.front();
            queue.pop_front();
            for (const PlannedPort& port : plan.nodes[u].ports) {
                if (distance[port.peer_device] == kUnreachable) {
                    distance[port.peer_device] = distance[u] + 1;
                    queue.push_back(port.peer_device);
                }
            }
        }
        for (uint32_t u = 0; u < n; u++) {
            PlannedNode& node = plan.nodes[u];
            node.next_port.resize(n, vc_node::kNoPort);
            if (u == d) {
                continue;
            }
            TT_FATAL(distance[u] != kUnreachable, "Device {} cannot reach {}; H7 refuses it", node.name, plan.nodes[d].name);
            for (uint32_t p = 0; p < node.ports.size(); p++) {
                if (distance[node.ports[p].peer_device] + 1 == distance[u]) {
                    node.next_port[d] = p;
                    break;
                }
            }
        }
    }

    for (uint32_t s = 0; s < n; s++) {
        for (uint32_t d = 0; d < n; d++) {
            for (uint32_t u = s; u != d;) {
                PlannedNode& node = plan.nodes[u];
                const uint32_t p = node.next_port[d];
                const PlannedPort& port = node.ports[p];
                node.predicted_sent[p] += shape.packets_per_flow;
                plan.nodes[port.peer_device].predicted_received[port.peer_port] += shape.packets_per_flow;
                u = port.peer_device;
            }
        }
    }
    return plan;
}

std::vector<uint32_t> node_runtime_args(const HarnessPlan& plan, uint32_t device, uint32_t nonce) {
    TT_FATAL(device < plan.nodes.size(), "Device {} is not a node of the plan's {}", device, plan.nodes.size());
    const PlannedNode& node = plan.nodes[device];
    const auto word = [](uint64_t value, const char* what) {
        TT_FATAL(value <= kMaxWord, "{} {:#x} does not fit a runtime-argument word", what, value);
        return static_cast<uint32_t>(value);
    };
    std::vector<uint32_t> args = {
        node.device,
        static_cast<uint32_t>(plan.nodes.size()),
        word(node.region_base, "region_base"),
        static_cast<uint32_t>(node.ports.size()),
        plan.shape.packets_per_flow,
        plan.shape.payload_bytes,
        plan.ring_slots,
        nonce};
    args.insert(args.end(), node.next_port.begin(), node.next_port.end());
    for (const PlannedPort& port : node.ports) {
        const PlannedNode& peer = plan.nodes[port.peer_device];
        args.push_back(port.backend == LinkBackend::Local ? vc_node::kPortKindLocal : vc_node::kPortKindNoc);
        args.push_back(peer.tile);
        args.push_back(word(peer.region_base + peer.layout.port_block(port.peer_port), "peer_port_base"));
    }
    return args;
}

}  // namespace tt::tt_fabric::vcluster
