// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_consumer.hpp"

#include <array>
#include <atomic>

#include <algorithm>
#include <mutex>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdevcommon/profiler_zone_id.h"
#include "llrt/zone_meta.hpp"
#include "tools/profiler/perf_debug_receiver.hpp"

namespace tt::tt_metal::perf_debug {

namespace {
constexpr uint32_t kMaxCorrChip = 64;
std::array<std::atomic<int64_t>, kMaxCorrChip>& correction_table() {
    static std::array<std::atomic<int64_t>, kMaxCorrChip> t{};
    return t;
}
std::atomic<uint64_t>& correction_epoch() {
    static std::atomic<uint64_t> e{0};
    return e;
}
}  // namespace

int64_t get_zone_ts_correction(uint32_t chip_id) {
    return chip_id < kMaxCorrChip ? correction_table()[chip_id].load(std::memory_order_relaxed) : 0;
}

uint64_t alignment_epoch() { return correction_epoch().load(std::memory_order_relaxed); }

void set_zone_ts_correction(uint32_t chip_id, int64_t cycles) {
    if (chip_id >= kMaxCorrChip) {
        return;
    }
    auto& slot = correction_table()[chip_id];
    int64_t cur = slot.load(std::memory_order_relaxed);
    while (cycles > cur && !slot.compare_exchange_weak(cur, cycles, std::memory_order_relaxed)) {
    }
    if (cycles > cur) {
        correction_epoch().fetch_add(1, std::memory_order_relaxed);
    }
}


void ZoneNameMirror::refresh() {
    std::vector<llrt::ZoneMetaEntry> delta;
    cursor_ = llrt::ZoneMetaRegistry::instance().additions_since(cursor_, delta);
    for (auto& e : delta) {
        names_.emplace(e.zone_id, std::move(e.name));
    }
}

void log_unnamed_ids(std::string_view consumer_name, const ZoneNameMirror& mirror) {
    if (mirror.unnamed() == 0) {
        return;
    }
    std::string ids;
    for (uint32_t id : mirror.unnamed_ids()) {
        ids +=
            fmt::format("{}{} (tu {} local {})", ids.empty() ? "" : ", ", id, TT_ZONE_TU_OF(id), TT_ZONE_LOCAL_OF(id));
    }
    log_warning(
        tt::LogMetal,
        "[perf-debug receiver] consumer \"{}\": {} unnamed marker rows [MUST be 0 -- a binary loaded without "
        ".tt_zone_meta, or two TUs share a tu_id]; ids (up to {} distinct): {}",
        consumer_name,
        mirror.unnamed(),
        ZoneNameMirror::kMaxUnnamedIds,
        ids);
}

namespace {

struct Registry {
    struct Entry {
        PerfDebugConsumerHandle handle = 0;
        std::string name;
        PerfDebugRecordCallback cb;
        PerfDebugConsumerHandle live = 0;  // receiver-side handle while a capture is active
    };
    std::mutex mu;
    std::vector<Entry> entries;
    PerfDebugConsumerHandle next_handle = 1;
    PerfDebugReceiver* receiver = nullptr;
};

Registry& registry() {
    static ttsl::Indestructible<Registry> r;
    return r.get();
}

}  // namespace

PerfDebugConsumerHandle register_consumer(std::string name, PerfDebugRecordCallback cb) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    Registry::Entry e{r.next_handle++, std::move(name), std::move(cb)};
    if (r.receiver != nullptr) {
        e.live = r.receiver->add_consumer(e.name, e.cb);
    }
    const PerfDebugConsumerHandle handle = e.handle;
    r.entries.push_back(std::move(e));
    return handle;
}

void unregister_consumer(PerfDebugConsumerHandle handle) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = std::find_if(r.entries.begin(), r.entries.end(), [&](const auto& e) { return e.handle == handle; });
    TT_FATAL(it != r.entries.end(), "unknown perf-debug consumer handle {}", handle);
    if (it->live != 0 && r.receiver != nullptr) {
        r.receiver->remove_consumer(it->live);
    }
    r.entries.erase(it);
}

void attach_registered_consumers(PerfDebugReceiver& receiver) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    r.receiver = &receiver;
    for (auto& e : r.entries) {
        e.live = receiver.add_consumer(e.name, e.cb);
    }
}

void detach_registered_consumers() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    r.receiver = nullptr;
    for (auto& e : r.entries) {
        e.live = 0;
    }
}

}  // namespace tt::tt_metal::perf_debug
