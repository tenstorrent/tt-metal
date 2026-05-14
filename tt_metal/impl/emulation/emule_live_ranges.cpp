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

struct State {
    std::mutex mu;
    std::unordered_map<int, std::vector<uint64_t>> per_device;
};

State& state() {
    static State s;
    return s;
}

uint64_t pack(uint32_t start, uint32_t end) {
    return (static_cast<uint64_t>(start) << 32) | static_cast<uint64_t>(end);
}

uint32_t unpack_start(uint64_t r) { return static_cast<uint32_t>(r >> 32); }

}  // namespace

void LiveL1Ranges::add(int device_id, uint32_t start, uint32_t end) {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    s.per_device[device_id].push_back(pack(start, end));
}

void LiveL1Ranges::remove(int device_id, uint32_t start) {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    auto it = s.per_device.find(device_id);
    if (it == s.per_device.end()) {
        return;
    }
    auto& v = it->second;
    auto match = std::find_if(
        v.begin(), v.end(), [start](uint64_t r) { return unpack_start(r) == start; });
    if (match != v.end()) {
        v.erase(match);
    }
}

std::vector<uint64_t> LiveL1Ranges::snapshot(int device_id) {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    auto it = s.per_device.find(device_id);
    if (it == s.per_device.end()) {
        return {};
    }
    return it->second;
}

}  // namespace tt::tt_metal::emule
