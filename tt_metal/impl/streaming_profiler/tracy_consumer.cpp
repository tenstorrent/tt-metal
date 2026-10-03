// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/tracy_consumer.hpp"

#if defined(TRACY_ENABLE)

#include <algorithm>

#include <fmt/format.h>
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

constexpr size_t kSrclocTableInitial = 1024;

// By physical coordinate: an eth core's logical coordinate can equal a Tensix core's.
uint64_t lane_key(const api::Core& core) {
    return (static_cast<uint64_t>(core.chip_id) << 32) | ((static_cast<uint64_t>(core.physical.x) & 0xFFFu) << 20) |
           ((static_cast<uint64_t>(core.physical.y) & 0xFFFu) << 8) | (static_cast<uint64_t>(core.processor) & 0xFFu);
}

uint64_t srcloc_key(std::string_view name, uint32_t processor) {
    return (static_cast<uint64_t>(name.size()) << 32) | processor;
}

size_t srcloc_hash(const char* name, uint64_t key) {
    return (((reinterpret_cast<uintptr_t>(name) >> 3) ^ key) * 0x9E3779B97F4A7C15ull) >> 32;
}

// Tracy's RiscType has one ERISC. ERISC1 sets the top bit of the 6-bit RISC field, which no RiscType uses, so its row
// gets its own thread id.
constexpr auto kTracyErisc1 = static_cast<tracy::RiscType>((1u << 5) | static_cast<uint8_t>(tracy::RiscType::ERISC));
struct ProcessorRow {
    const char* name;
    tracy::RiscType risc;
    uint32_t color;
};
constexpr ProcessorRow kProcessorRows[kProcessorCount] = {
    {"BRISC", tracy::RiscType::BRISC, tracy::Color::Orange2},
    {"NCRISC", tracy::RiscType::NCRISC, tracy::Color::SeaGreen3},
    {"TRISC_0", tracy::RiscType::TRISC_0, tracy::Color::SkyBlue3},
    {"TRISC_1", tracy::RiscType::TRISC_1, tracy::Color::Turquoise2},
    {"TRISC_2", tracy::RiscType::TRISC_2, tracy::Color::CadetBlue1},
    {"ERISC_0", tracy::RiscType::ERISC, tracy::Color::Yellow3},
    {"ERISC_1", kTracyErisc1, tracy::Color::Yellow2}};

tracy::TTDeviceMarker device_marker(const api::Core& core) {
    tracy::TTDeviceMarker marker;
    marker.chip_id = core.chip_id;
    marker.core_x = core.logical.x;
    marker.core_y = core.logical.y;
    marker.risc = kProcessorRows[static_cast<uint32_t>(core.processor)].risc;
    return marker;
}

}  // namespace

TracyConsumer::TracyConsumer() : anchor_tracy_(tracy::Profiler::GetTime()), srcloc_table_(kSrclocTableInitial) {}

void TracyConsumer::operator()(const Batch& batch) {
    for (const api::Zone& zone : batch.zones()) {
        push_zone(zone.core(), zone.site().name, zone.start_tsc(), zone.end_tsc());
    }
    for (const api::TimestampedData& data : batch.timestamped_data()) {
        push_marker(data.core(), data.site().name, data.tsc(), data.runtime_id(), data.payload());
    }
    for (const api::Event& event : batch.events()) {
        push_marker(event.core(), event.site().name, event.tsc(), event.runtime_id(), {});
    }
}

// A record's TSC comes from host_sync's tsc_now(), the same counter Tracy's GetTime() reads (rdtsc on x86-64,
// CLOCK_MONOTONIC_RAW elsewhere), so timeline_ns can subtract a GetTime() anchor from it.
int64_t TracyConsumer::timeline_ns(int64_t tsc) const {
    return static_cast<int64_t>(static_cast<double>(tsc - anchor_tracy_) * TracyGetTimerMul());
}

TracyConsumer::Lane TracyConsumer::lane(const Core& core) {
    const uint64_t key = lane_key(core);
    if (key == lane_key_) {
        return lane_hit_;
    }
    Lane found;
    found.processor = static_cast<uint32_t>(core.processor);
    auto [it, fresh] = cores_.try_emplace(key & ~uint64_t{0xFF});
    CoreEntry& entry = it->second;
    if (fresh) {
        entry.ctx = TracyTTContext();
        // Everything the sink emits goes through this thread's lock-free queue, and its FIFO order keeps the context
        // ahead of the zones that refer to it.
        TracyTTContextPopulateCalibratedLockfree(entry.ctx, anchor_tracy_, 0.0, 1.0);
        const std::string name = fmt::format(
            "Device: {}, {}Logical ({},{}) Physical ({},{})",
            core.chip_id,
            core.processor >= api::Processor::ERISC0 ? "Ethernet " : "",
            core.logical.x,
            core.logical.y,
            core.physical.x,
            core.physical.y);
        TracyTTContextNameLockfree(entry.ctx, name.c_str(), name.size());
    }
    if ((entry.named & (1u << found.processor)) == 0) {
        // Use the marker path's thread id (TTDeviceMarker::get_thread_id) so zones and markers share each RISC's row.
        entry.thread[found.processor] = device_marker(core).get_thread_id();
        tracy::SetThreadName(entry.thread[found.processor], kProcessorRows[found.processor].name);
        entry.named |= static_cast<uint8_t>(1u << found.processor);
    }
    found.ctx = entry.ctx;
    found.thread = entry.thread[found.processor];
    lane_key_ = key;
    lane_hit_ = found;
    return found;
}

const tracy::SourceLocationData* TracyConsumer::srcloc(std::string_view name, uint32_t processor) {
    const uint64_t key = srcloc_key(name, processor);
    const size_t mask = srcloc_table_.size() - 1;
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    for (size_t i = srcloc_hash(name_ptr, key) & mask;; i = (i + 1) & mask) {
        const SrclocEntry& e = srcloc_table_[i];
        if (e.name == name_ptr && e.key == key) {
            return e.srcloc;
        }
        if (e.name == nullptr) {
            return srcloc_slow(name, processor);
        }
    }
}

const tracy::SourceLocationData* TracyConsumer::srcloc_slow(std::string_view name, uint32_t processor) {
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    const uint64_t key = srcloc_key(name, processor);
    // Same colors as getMarkerColor (TracyTTDevice.hpp): Tomato3 for PROFILER-keyword names, then the per-RISC palette.
    const uint32_t color =
        name.find("PROFILER") != std::string_view::npos ? tracy::Color::Tomato3 : kProcessorRows[processor].color;
    const std::string skey = fmt::format("{}#{:08x}", name, color);
    auto it = srclocs_.find(skey);
    if (it == srclocs_.end()) {
        const std::string& owned_name = srcloc_names_.emplace_back(name);
        const tracy::SourceLocationData& data = srcloc_data_.emplace_back(
            tracy::SourceLocationData{owned_name.c_str(), "kernel_profiler", "kernel_profiler", 0, color});
        it = srclocs_.emplace(skey, &data).first;
    }
    auto insert = [](std::vector<SrclocEntry>& table, const SrclocEntry& e) {
        const size_t mask = table.size() - 1;
        size_t i = srcloc_hash(e.name, e.key) & mask;
        while (table[i].name != nullptr) {
            i = (i + 1) & mask;
        }
        table[i] = e;
    };
    if ((srcloc_count_ + 1) * 2 > srcloc_table_.size()) {
        std::vector<SrclocEntry> old = std::move(srcloc_table_);
        srcloc_table_.assign(old.size() * 2, {});
        for (const SrclocEntry& e : old) {
            if (e.name != nullptr) {
                insert(srcloc_table_, e);
            }
        }
    }
    insert(srcloc_table_, {name_ptr, key, it->second});
    srcloc_count_++;
    return it->second;
}

void TracyConsumer::push_zone(const Core& core, std::string_view name, int64_t start_tsc, int64_t end_tsc) {
    const int64_t start_ns = timeline_ns(start_tsc);
    const int64_t end_ns = std::max(timeline_ns(end_tsc), start_ns);
    const Lane zone_lane = lane(core);
    TracyTTPushZone(
        zone_lane.ctx,
        srcloc(name, zone_lane.processor),
        zone_lane.thread,
        static_cast<uint64_t>(start_ns),
        static_cast<uint64_t>(end_ns));
}

void TracyConsumer::push_marker(
    const Core& core, std::string_view name, int64_t tsc, uint32_t runtime_id, std::span<const uint64_t> values) {
    const int64_t timestamp_ns = timeline_ns(tsc);
    TracyTTCtx ctx = lane(core).ctx;
    tracy::TTDeviceMarker marker = device_marker(core);
    // Tracy's fallback color asserts on a RiscType it doesn't know.
    marker.color = kProcessorRows[static_cast<uint32_t>(core.processor)].color;
    marker.timestamp = static_cast<uint64_t>(timestamp_ns);
    marker.runtime_host_id = runtime_id;
    marker.marker_type = values.empty() ? tracy::TTDeviceMarkerType::EVENT : tracy::TTDeviceMarkerType::DATA;
    marker.marker_name = std::string(name);
    marker.file = "kernel_profiler";
    marker.line = 0;
    if (!values.empty()) {
        marker.data = values[0];
    }
    if (values.size() > 1) {
        marker.data_high = values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (size_t i = 2; i < values.size(); i++) {
        marker.meta_data[fmt::format("value{}", i)] = values[i];
    }
#endif
    TracyTTPushMarkerLockfree(ctx, marker);
}

}  // namespace tt::tt_metal::streaming_profiler

#endif
