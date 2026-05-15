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

struct RangeRegistry {
    std::mutex mu;
    std::unordered_map<int, std::vector<uint64_t>> per_device;
};

uint64_t pack(uint32_t start, uint32_t end) {
    return (static_cast<uint64_t>(start) << 32) | static_cast<uint64_t>(end);
}

uint32_t unpack_start(uint64_t r) { return static_cast<uint32_t>(r >> 32); }

void add_to(RangeRegistry& reg, int device_id, uint32_t start, uint32_t end) {
    std::lock_guard<std::mutex> g(reg.mu);
    reg.per_device[device_id].push_back(pack(start, end));
}

void remove_from(RangeRegistry& reg, int device_id, uint32_t start) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return;
    }
    auto& v = it->second;
    auto match = std::find_if(
        v.begin(), v.end(), [start](uint64_t r) { return unpack_start(r) == start; });
    if (match != v.end()) {
        v.erase(match);
    }
}

std::vector<uint64_t> snapshot_of(RangeRegistry& reg, int device_id) {
    std::lock_guard<std::mutex> g(reg.mu);
    auto it = reg.per_device.find(device_id);
    if (it == reg.per_device.end()) {
        return {};
    }
    return it->second;
}

RangeRegistry& l1_registry() {
    static RangeRegistry r;
    return r;
}

RangeRegistry& dram_registry() {
    static RangeRegistry r;
    return r;
}

}  // namespace

void LiveL1Ranges::add(int device_id, uint32_t start, uint32_t end) {
    add_to(l1_registry(), device_id, start, end);
}

void LiveL1Ranges::remove(int device_id, uint32_t start) {
    remove_from(l1_registry(), device_id, start);
}

std::vector<uint64_t> LiveL1Ranges::snapshot(int device_id) {
    return snapshot_of(l1_registry(), device_id);
}

void LiveDramRanges::add(int device_id, uint32_t start, uint32_t end) {
    add_to(dram_registry(), device_id, start, end);
}

void LiveDramRanges::remove(int device_id, uint32_t start) {
    remove_from(dram_registry(), device_id, start);
}

std::vector<uint64_t> LiveDramRanges::snapshot(int device_id) {
    return snapshot_of(dram_registry(), device_id);
}

}  // namespace tt::tt_metal::emule
