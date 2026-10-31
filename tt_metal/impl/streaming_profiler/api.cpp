// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/observability_event_id.h"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "llrt/observability_event_meta.hpp"

namespace api = tt::tt_metal::experimental::streaming_profiler;

namespace tt::tt_metal::experimental::streaming_profiler::detail {
static_assert(OBSERVABILITY_EVENT_ID_BITS == TT_OBSERVABILITY_EVENT_ID_BITS);
static_assert(OBSERVABILITY_EVENT_LOCAL_BITS == TT_OBSERVABILITY_EVENT_LOCAL_BITS);
static_assert(OBSERVABILITY_EVENT_TU_COUNT == TT_OBSERVABILITY_EVENT_TU_COUNT);
std::atomic<const ObservabilityEventTu*> ObservabilityEventRegistry::tus[OBSERVABILITY_EVENT_TU_COUNT];
}  // namespace tt::tt_metal::experimental::streaming_profiler::detail

namespace tt::tt_metal::streaming_profiler {

namespace {

static_assert(TT_ZONE_STALL_ID == (TT_OBSERVABILITY_EVENT_RESERVED_TU << TT_OBSERVABILITY_EVENT_LOCAL_BITS));

constexpr api::ObservabilityEventInfo kStallObservabilityEvent{.name = api::STALL_ZONE_NAME};
constexpr const api::ObservabilityEventInfo* kStallObservabilityEvents[1] = {&kStallObservabilityEvent};
constexpr api::detail::ObservabilityEventTu kStallTu{kStallObservabilityEvents};

// Builds the tables behind api::detail::observability_event_of from the event-name registry as ELFs load. Nothing is
// ever freed: a record may hold an observability event's address for the life of the process, and a reader may still be
// walking a replaced table.
class ObservabilityEventTables {
public:
    void add(std::span<const tt::llrt::ObservabilityEventMetaEntry* const> entries) {
        std::lock_guard<std::mutex> lk(mu_);
        std::map<uint32_t, std::vector<const api::ObservabilityEventInfo*>> grown;
        for (const tt::llrt::ObservabilityEventMetaEntry* e : entries) {
            const uint32_t tu = TT_OBSERVABILITY_EVENT_TU_OF(e->event_id),
                           local = TT_OBSERVABILITY_EVENT_LOCAL_OF(e->event_id);
            auto [it, fresh] = grown.try_emplace(tu);
            if (fresh) {
                if (const api::detail::ObservabilityEventTu* cur =
                        api::detail::ObservabilityEventRegistry::tus[tu].load(std::memory_order_relaxed)) {
                    it->second.assign(cur->observability_events.begin(), cur->observability_events.end());
                }
            }
            if (local >= it->second.size()) {
                it->second.resize(local + 1, nullptr);
            }
            if (it->second[local] == nullptr) {
                it->second[local] = &observability_events_.emplace_back(
                    api::ObservabilityEventInfo{.name = e->name, .location = {.file = e->file, .line = e->line}});
            }
        }
        for (auto& [tu, v] : grown) {
            auto arr = std::make_unique<const api::ObservabilityEventInfo*[]>(v.size());
            std::copy(v.begin(), v.end(), arr.get());
            auto t = std::make_unique<api::detail::ObservabilityEventTu>(api::detail::ObservabilityEventTu{
                std::span<const api::ObservabilityEventInfo* const>(arr.get(), v.size())});
            api::detail::ObservabilityEventRegistry::tus[tu].store(t.get(), std::memory_order_release);
            arrays_.push_back(std::move(arr));
            tus_.push_back(std::move(t));
        }
    }

private:
    std::mutex mu_;
    std::deque<api::ObservabilityEventInfo> observability_events_;
    std::vector<std::unique_ptr<const api::ObservabilityEventInfo*[]>> arrays_;
    std::vector<std::unique_ptr<api::detail::ObservabilityEventTu>> tus_;
};

}  // namespace

void init_observability_event_registry() {
    static std::once_flag once;
    std::call_once(once, [] {
        api::detail::ObservabilityEventRegistry::tus[TT_OBSERVABILITY_EVENT_RESERVED_TU].store(
            &kStallTu, std::memory_order_release);
        static ttsl::Indestructible<ObservabilityEventTables> tables;
        tt::llrt::ObservabilityEventMetaRegistry::instance().set_listener(
            [](std::span<const tt::llrt::ObservabilityEventMetaEntry* const> entries) { tables.get().add(entries); });
    });
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {

namespace internal = tt::tt_metal::streaming_profiler;

Callback detail::register_callback(std::string name, uint32_t types, std::function<void(const BatchData&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    if (name.empty()) {
        name = "callback-" + std::to_string(++anonymous);
    }
    return Callback(internal::service().add_consumer(std::move(name), types, std::move(callback)));
}

Callback::Callback(Callback&& other) noexcept : id_(std::exchange(other.id_, detail::CallbackId{})) {}

Callback& Callback::operator=(Callback&& other) noexcept {
    if (this != &other) {
        reset();
        id_ = std::exchange(other.id_, detail::CallbackId{});
    }
    return *this;
}

Callback::~Callback() { reset(); }

void Callback::reset() noexcept {
    if (id_ != detail::CallbackId{}) {
        internal::service().remove_consumer(std::exchange(id_, detail::CallbackId{}));
    }
}

TimestampedData::TimestampedData(const TimestampedData& other) : PointRecord(other), value_count_(other.value_count_) {
    uint64_t* values = new uint64_t[value_count_];
    std::ranges::copy(other.payload(), values);
    values_ = values;
}

TimestampedData::TimestampedData(TimestampedData&& other) noexcept :
    PointRecord(other),
    value_count_(std::exchange(other.value_count_, 0)),
    values_(std::exchange(other.values_, nullptr)) {}

TimestampedData& TimestampedData::operator=(TimestampedData other) noexcept {
    PointRecord::operator=(other);
    std::swap(value_count_, other.value_count_);
    std::swap(values_, other.values_);
    return *this;
}

TimestampedData::~TimestampedData() { delete[] values_; }

bool IsActive() { return internal::service().is_active(); }

constinit thread_local Record::TscToSteadyLine Record::tsc_to_steady_line_{};

std::chrono::steady_clock::time_point Record::refill_tsc_to_steady_line(int64_t tsc) {
    const internal::SteadyClock& steady = internal::service().steady();
    const std::optional<internal::Segment<int64_t>> segment = steady.segment_at(tsc);
    TT_FATAL(
        segment, "streaming profiler: a TSC tick maps to steady_clock only after a capture's first host sync burst");
    tsc_to_steady_line_ = {
        .from = segment->from,
        .to = segment->to,
        .origin = segment->origin,
        .base_ns = steady.base_ns(),
        .value = segment->value,
        .slope = segment->slope};
    // Past the newest SteadyClock sample, segment_at() returns an empty segment that only extrapolates, so calling
    // tsc_to_steady() here would just refill again.
    return tsc_to_steady_line_.at(tsc);
}

}  // namespace tt::tt_metal::experimental::streaming_profiler
