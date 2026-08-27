// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_live_ranges.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal::emule {

namespace {

uint64_t pack(uint32_t start, uint32_t end) {
    return (static_cast<uint64_t>(start) << 32) | static_cast<uint64_t>(end);
}

// Each entry keeps the registering buffer's unique_id so removal erases exactly
// that registration — never an address match (see the header comment).
struct OwnedRange {
    uint64_t packed;
    uint64_t owner;
};

struct OwnedRangeRegistry {
    std::mutex mu;
    std::unordered_map<int, std::vector<OwnedRange>> per_device;
};

// add never de-dups and remove is a linear find/erase, so each op is O(n).
// Intentional: a device's live-range set stays small (a handful of buffers) and
// this is a debug-only build, so the simplicity beats an index.
void add_to(OwnedRangeRegistry& reg, int device_id, uint32_t start, uint32_t end, uint64_t owner) {
    std::lock_guard<std::mutex> g(reg.mu);
    reg.per_device[device_id].push_back(OwnedRange{pack(start, end), owner});
}

void remove_from(OwnedRangeRegistry& reg, int device_id, uint64_t owner) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return;
    }
    auto& v = it->second;
    auto match = std::find_if(v.begin(), v.end(), [owner](const OwnedRange& r) { return r.owner == owner; });
    if (match != v.end()) {
        v.erase(match);
    }
}

std::vector<uint64_t> snapshot_of(OwnedRangeRegistry& reg, int device_id) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return {};
    }
    std::vector<uint64_t> out;
    out.reserve(it->second.size());
    for (const auto& r : it->second) {
        out.push_back(r.packed);
    }
    return out;
}

// Host-poke ranges have no owning Buffer: add-only with dedup, since the same
// scratch address is re-poked every iteration of a benchmark loop and these
// ranges are never removed.
struct PlainRangeRegistry {
    std::mutex mu;
    std::unordered_map<int, std::vector<uint64_t>> per_device;
};

void add_dedup_to(PlainRangeRegistry& reg, int device_id, uint32_t start, uint32_t end) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto& v = reg.per_device[device_id];
    uint64_t packed = pack(start, end);
    if (std::find(v.begin(), v.end(), packed) == v.end()) {
        v.push_back(packed);
    }
}

std::vector<uint64_t> snapshot_of(PlainRangeRegistry& reg, int device_id) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return {};
    }
    return it->second;
}

OwnedRangeRegistry& l1_registry() {
    static OwnedRangeRegistry r;
    return r;
}

OwnedRangeRegistry& dram_registry() {
    static OwnedRangeRegistry r;
    return r;
}

PlainRangeRegistry& l1_host_poke_registry() {
    static PlainRangeRegistry r;
    return r;
}

}  // namespace

void LiveL1Ranges::add(int device_id, uint32_t start, uint32_t end, uint64_t owner) {
    add_to(l1_registry(), device_id, start, end, owner);
}

void LiveL1Ranges::remove(int device_id, uint64_t owner) { remove_from(l1_registry(), device_id, owner); }

std::vector<uint64_t> LiveL1Ranges::snapshot(int device_id) {
    return snapshot_of(l1_registry(), device_id);
}

void LiveDramRanges::add(int device_id, uint32_t start, uint32_t end, uint64_t owner) {
    add_to(dram_registry(), device_id, start, end, owner);
}

void LiveDramRanges::remove(int device_id, uint64_t owner) { remove_from(dram_registry(), device_id, owner); }

std::vector<uint64_t> LiveDramRanges::snapshot(int device_id) {
    return snapshot_of(dram_registry(), device_id);
}

void LiveL1HostPokeRanges::add(int device_id, uint32_t start, uint32_t end) {
    add_dedup_to(l1_host_poke_registry(), device_id, start, end);
}

std::vector<uint64_t> LiveL1HostPokeRanges::snapshot(int device_id) {
    return snapshot_of(l1_host_poke_registry(), device_id);
}

namespace {

// `start` is the lookup key for clear(); the kernel-side check only consumes
// (logical_end, physical_end).
struct PaddingEntry {
    uint32_t start;
    uint32_t logical_end;
    uint32_t physical_end;
};

struct PaddingRegistry {
    std::mutex mu;
    std::unordered_map<int, std::vector<PaddingEntry>> per_device;
};

PaddingRegistry& padding_registry() {
    static PaddingRegistry r;
    return r;
}

}  // namespace

void LiveL1PaddingRanges::set(int device_id, uint32_t start, uint32_t logical_end, uint32_t physical_end) {
    auto& reg = padding_registry();
    std::lock_guard<std::mutex> g(reg.mu);
    auto& v = reg.per_device[device_id];
    auto match = std::find_if(
        v.begin(), v.end(), [start](const PaddingEntry& e) { return e.start == start; });
    if (match != v.end()) {
        match->logical_end = logical_end;
        match->physical_end = physical_end;
    } else {
        v.push_back(PaddingEntry{start, logical_end, physical_end});
    }
}

void LiveL1PaddingRanges::clear(int device_id, uint32_t start) {
    auto& reg = padding_registry();
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return;
    }
    auto& v = it->second;
    auto match = std::find_if(
        v.begin(), v.end(), [start](const PaddingEntry& e) { return e.start == start; });
    if (match != v.end()) {
        v.erase(match);
    }
}

std::vector<uint64_t> LiveL1PaddingRanges::snapshot(int device_id) {
    auto& reg = padding_registry();
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return {};
    }
    std::vector<uint64_t> out;
    out.reserve(it->second.size());
    for (const auto& e : it->second) {
        out.push_back(pack(e.logical_end, e.physical_end));
    }
    return out;
}

}  // namespace tt::tt_metal::emule
