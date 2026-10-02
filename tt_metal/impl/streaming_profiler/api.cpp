// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/profiler_zone_id.h"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "llrt/zone_meta.hpp"

namespace api = tt::tt_metal::experimental::streaming_profiler;

namespace tt::tt_metal::experimental::streaming_profiler::detail {
static_assert(ZONE_ID_BITS == TT_ZONE_ID_BITS);
static_assert(ZONE_ID_COUNT == TT_ZONE_ID_COUNT);
std::atomic<const MarkerSite*> SiteRegistry::sites[ZONE_ID_COUNT];
}  // namespace tt::tt_metal::experimental::streaming_profiler::detail

namespace tt::tt_metal::streaming_profiler {

namespace {

static_assert(TT_ZONE_STALL_ID < api::detail::ZONE_ID_COUNT);

constexpr api::MarkerSite kStallSite{.name = api::STALL_ZONE_NAME};

// Fills the table behind api::detail::site_of from the zone-name registry as ELFs load. Nothing is ever freed: a
// record may hold a site's address for the life of the process. Ids are unique per process by construction
// (each image owns a block), so a slot is written once; a repeat is a registry fault and the first writer wins.
class SiteTables {
public:
    void add(std::span<const tt::llrt::ZoneMetaEntry* const> entries) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const tt::llrt::ZoneMetaEntry* e : entries) {
            if (e->zone_id >= api::detail::ZONE_ID_COUNT) {
                continue;
            }
            auto& slot = api::detail::SiteRegistry::sites[e->zone_id];
            if (slot.load(std::memory_order_relaxed) == nullptr) {
                slot.store(
                    &sites_.emplace_back(
                        api::MarkerSite{.name = e->name, .location = {.file = e->file, .line = e->line}}),
                    std::memory_order_release);
            }
        }
    }

private:
    std::mutex mu_;
    std::deque<api::MarkerSite> sites_;
};

}  // namespace

void init_site_registry() {
    static std::once_flag once;
    std::call_once(once, [] {
        api::detail::SiteRegistry::sites[TT_ZONE_STALL_ID].store(&kStallSite, std::memory_order_release);
        static ttsl::Indestructible<SiteTables> tables;
        tt::llrt::ZoneMetaRegistry::instance().set_listener(
            [](std::span<const tt::llrt::ZoneMetaEntry* const> entries) { tables.get().add(entries); });
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
