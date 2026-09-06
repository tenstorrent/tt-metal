// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <functional>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tools/profiler/streaming_profiler_consumer.hpp"
#include "tools/profiler/streaming_profiler_service.hpp"

namespace tt::tt_metal::experimental::streaming_profiler {

namespace {

namespace internal = tt::tt_metal::streaming_profiler;

Core core_of(const internal::LaneInfo& li) {
    return Core{
        .coord = CoreCoord(li.logical_x, li.logical_y), .chip_id = li.chip_id, .risc = static_cast<Risc>(li.risc)};
}

Site site_of(const internal::ZoneNameMirror::Site* s) {
    if (s == nullptr) {
        return {};
    }
    return Site{.name = s->name, .location = {.file = s->file, .line = s->line}};
}

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
        stalls_.clear();
        staged_.clear();
        arena_.clear();
        const internal::CaptureContext& ctx = *batch.context;
        clocks_ = batch.clocks;
        for (const internal::Rec& r : batch.records) {
            const internal::LaneInfo& lane = ctx.devices[r.meta.dev].lanes[r.meta.lane];
            switch (r.meta.type) {
                case T::Zone:
                    if (stall(r.id)) {
                        if (detail::has(channels_, Channel::Stalls)) {
                            stalls_.push_back(Stall{
                                .core = core_of(lane),
                                .clock = std::cref(batch.clocks[r.meta.dev]),
                                .start_timestamp = r.data.zone.start,
                                .end_timestamp = r.data.zone.start + r.data.zone.duration,
                                .runtime_id = r.prog});
                        }
                    } else if (detail::has(channels_, Channel::Zones)) {
                        zones_.push_back(Zone{
                            .site = site_of(names_.lookup_site(r.id)),
                            .core = core_of(lane),
                            .clock = std::cref(batch.clocks[r.meta.dev]),
                            .start_timestamp = r.data.zone.start,
                            .end_timestamp = r.data.zone.start + r.data.zone.duration,
                            .runtime_id = r.prog});
                    }
                    break;
                case T::Event:
                    if (detail::has(channels_, Channel::Events)) {
                        events_.push_back(Event{
                            .site = site_of(names_.lookup_site(r.id)),
                            .core = core_of(lane),
                            .clock = std::cref(batch.clocks[r.meta.dev]),
                            .timestamp = r.data.ts,
                            .runtime_id = r.prog});
                    }
                    break;
                case T::Data:
                case T::Ext:
                case T::Cont:
                    if (detail::has(channels_, Channel::TimestampedData)) {
                        assemble(lane, r);
                    }
                    break;
                default: break;
            }
        }
        const bool any = !zones_.empty() || !events_.empty() || !stalls_.empty() || !staged_.empty();
        if (!any && batch.dropped_delta == 0 && batch.stall_delta == 0) {
            return;
        }
        data_.clear();
        for (const Staged& s : staged_) {
            data_.push_back(s.record);
            data_.back().clock = std::cref(clocks_[s.dev]);  // a head from an earlier batch takes this batch's clock
            data_.back().payload = std::span<const uint64_t>(arena_.data() + s.payload_offset, s.payload_len);
        }
        Batch<Channel::All> full;
        full.zones = zones_;
        full.timestamped_data = data_;
        full.events = events_;
        full.stalls = stalls_;
        full.clocks = batch.clocks;
        full.dropped = batch.dropped_delta;
        full.stall_count = batch.stall_delta;
        cb_(full);
    }

private:
    // A Data head, its Ext and its Conts can straddle batches, so one assembly per lane persists across calls.
    // Payload spans point into arena_, which only grows until the batch is delivered.
    struct Pending {
        bool active = false;
        uint32_t dev = 0;
        std::optional<TimestampedData> head;
        uint32_t want = 0;
        std::vector<uint64_t> payload;
    };
    struct Staged {
        TimestampedData record;
        uint32_t dev = 0;
        size_t payload_offset = 0;
        size_t payload_len = 0;
    };

    bool stall(uint32_t id) {
        if (auto it = is_stall_.find(id); it != is_stall_.end()) {
            return it->second;
        }
        const bool s = names_.lookup(id) == "PROFILER-STALL";
        is_stall_.emplace(id, s);
        return s;
    }

    void complete(Pending& p) {
        p.active = false;
        staged_.push_back(Staged{*p.head, p.dev, arena_.size(), p.payload.size()});
        arena_.insert(arena_.end(), p.payload.begin(), p.payload.end());
    }

    void assemble(const internal::LaneInfo& lane, const internal::Rec& r) {
        using T = internal::RecType;
        Pending& p = pending_[(r.meta.dev << 10) | r.meta.lane];
        if (r.meta.type == T::Data) {
            if (p.active) {
                complete(p);  // a new head ends a truncated predecessor
            }
            p.active = true;
            p.dev = r.meta.dev;
            p.want = 0;
            p.payload.clear();
            p.head = TimestampedData{
                .site = site_of(names_.lookup_site(r.id)),
                .core = core_of(lane),
                .clock = std::cref(clocks_[r.meta.dev]),
                .timestamp = r.data.ts,
                .runtime_id = r.prog};
            return;
        }
        if (!p.active) {
            return;
        }
        if (r.meta.type == T::Ext) {
            p.want = (r.id + 1) / 2;  // payload words, two per element
            if (p.want != 0) {
                p.payload.push_back(r.data.ext);
            }
        } else {
            p.payload.push_back(r.data.payload);
        }
        if (p.payload.size() >= p.want) {
            complete(p);
        }
    }

    const Channel channels_;
    const std::function<void(const Batch<Channel::All>&)> cb_;
    std::span<const Clock> clocks_;  // this batch's
    internal::ZoneNameMirror names_;
    std::unordered_map<uint32_t, bool> is_stall_;    // the stall zone is a Stall, never a Zone
    std::unordered_map<uint32_t, Pending> pending_;  // keyed (dev << 10) | lane
    std::vector<Zone> zones_;
    std::vector<Event> events_;
    std::vector<Stall> stalls_;
    std::vector<Staged> staged_;
    std::vector<uint64_t> arena_;
    std::vector<TimestampedData> data_;
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
