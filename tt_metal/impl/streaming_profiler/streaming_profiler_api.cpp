// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "llrt/zone_meta.hpp"

namespace tt::tt_metal::experimental::streaming_profiler {

namespace {

namespace internal = tt::tt_metal::streaming_profiler;

Core core_of(const internal::LaneInfo& li) {
    return Core{
        .logical = CoreCoord(li.logical_x, li.logical_y),
        .physical = CoreCoord(li.physical_x, li.physical_y),
        .chip_id = li.chip_id,
        .risc = static_cast<Risc>(li.risc)};
}

// Per-subscription mirror of the process-wide per-ELF zone-name registry; a subscription runs on its own thread, so
// lookups need no lock. refresh() once per batch, lookup() per record. Names register at load, strictly before a
// binary can emit, so a miss is a binary without .tt_zone_meta or a tu_id collision. The strings never move or die
// while the subscription lives; the Tracy sink keys on their addresses.
class ZoneNameMirror {
public:
    struct Entry {
        std::string name;
        std::string file;
        uint32_t line = 0;
    };

    void refresh() {
        std::vector<tt::llrt::ZoneMetaEntry> delta;
        cursor_ = tt::llrt::ZoneMetaRegistry::instance().additions_since(cursor_, delta);
        for (auto& e : delta) {
            sites_.emplace(e.zone_id, Entry{std::move(e.name), std::move(e.file), e.line});
        }
    }
    Site lookup(uint32_t id) const {
        const auto it = sites_.find(id);
        if (it == sites_.end()) {
            return {};
        }
        return Site{.name = it->second.name, .location = {.file = it->second.file, .line = it->second.line}};
    }

private:
    std::unordered_map<uint32_t, Entry> sites_;
    uint32_t cursor_ = 0;
};

}  // namespace

// One per subscription, driven from that subscription's consumer thread, so no locking. Unsubscribed channels are
// skipped before any decoding.
class Adapter {
public:
    Adapter(Channel channels, std::function<void(const Batch<Channel::All>&)> cb) :
        channels_(channels), cb_(std::move(cb)) {}

    void operator()(const internal::RecordBatch& batch) {
        using T = internal::RecType;
        names_.refresh();
        zones_.clear();
        events_.clear();
        data_.clear();
        arena_.clear();
        // Every Ext and Cont record adds at most one payload element, so the arena never reallocates under the
        // spans handed out below.
        arena_.reserve(batch.records.size());
        const internal::CaptureContext& ctx = *batch.context;
        const std::span<const internal::Rec> recs = batch.records;
        for (size_t i = 0; i < recs.size(); i++) {
            const internal::Rec& r = recs[i];
            const internal::LaneInfo& lane = ctx.devices[r.meta.dev].lanes[r.meta.lane];
            const Clock& clock = batch.clocks[r.meta.dev];
            switch (r.meta.type) {
                case T::Zone:
                    if (detail::has(channels_, Channel::Zones)) {
                        const bool is_stall = r.id == profiler::kSpscStallZoneId;
                        zones_.push_back(Zone{
                            .site = is_stall ? Site{.name = kStallZoneName} : names_.lookup(r.id),
                            .core = core_of(lane),
                            .clock = std::cref(clock),
                            .start_timestamp = r.data.zone.start,
                            .end_timestamp = r.data.zone.start + r.data.zone.duration,
                            .runtime_id = r.prog,
                            .stall = is_stall});
                    }
                    break;
                case T::Event:
                    if (detail::has(channels_, Channel::Events)) {
                        events_.push_back(Event{
                            .site = names_.lookup(r.id),
                            .core = core_of(lane),
                            .clock = std::cref(clock),
                            .timestamp = r.data.ts,
                            .runtime_id = r.prog});
                    }
                    break;
                case T::Data: {
                    // The head's Ext (payload words 0-1 and the word count) and Conts follow it directly: the decoder
                    // writes the group in one call and batches split only between frames.
                    const size_t first = arena_.size();
                    size_t j = i + 1;
                    if (j < recs.size() && recs[j].meta.type == T::Ext) {
                        if (recs[j].id != 0) {
                            arena_.push_back(recs[j].data.ext);
                        }
                        for (j++; j < recs.size() && recs[j].meta.type == T::Cont; j++) {
                            arena_.push_back(recs[j].data.payload);
                        }
                    }
                    if (detail::has(channels_, Channel::TimestampedData)) {
                        data_.push_back(TimestampedData{
                            .site = names_.lookup(r.id),
                            .core = core_of(lane),
                            .payload = std::span<const uint64_t>(arena_.data() + first, arena_.size() - first),
                            .clock = std::cref(clock),
                            .timestamp = r.data.ts,
                            .runtime_id = r.prog});
                    }
                    i = j - 1;
                    break;
                }
                case T::Ext:
                case T::Cont: break;  // consumed with their Data head
            }
        }
        if (zones_.empty() && events_.empty() && data_.empty() && batch.dropped_delta == 0 && batch.stall_delta == 0) {
            return;
        }
        Batch<Channel::All> full;
        full.zones = zones_;
        full.timestamped_data = data_;
        full.events = events_;
        full.clocks = batch.clocks;
        full.dropped = batch.dropped_delta;
        full.stall_count = batch.stall_delta;
        cb_(full);
    }

private:
    const Channel channels_;
    const std::function<void(const Batch<Channel::All>&)> cb_;
    ZoneNameMirror names_;
    std::vector<Zone> zones_;
    std::vector<Event> events_;
    std::vector<TimestampedData> data_;
    std::vector<uint64_t> arena_;  // the batch's payloads, which data_'s spans point into
};

}  // namespace tt::tt_metal::experimental::streaming_profiler

namespace tt::tt_metal::streaming_profiler {

RecordCallback make_public_adapter(
    experimental::streaming_profiler::Channel channels,
    std::function<void(const experimental::streaming_profiler::Batch<experimental::streaming_profiler::Channel::All>&)>
        callback) {
    auto adapter = std::make_shared<experimental::streaming_profiler::Adapter>(channels, std::move(callback));
    return [adapter](const RecordBatch& b) { (*adapter)(b); };
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {

SubscriptionHandle detail::subscribe(
    std::string_view name, Channel channels, std::function<void(const Batch<Channel::All>&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    const std::string label = name.empty() ? "subscription-" + std::to_string(++anonymous) : std::string(name);
    return internal::service().add_consumer(label, internal::make_public_adapter(channels, std::move(callback)));
}

void Unsubscribe(SubscriptionHandle handle) { internal::service().remove_consumer(handle); }

bool IsActive() { return internal::service().is_active(); }

}  // namespace tt::tt_metal::experimental::streaming_profiler
