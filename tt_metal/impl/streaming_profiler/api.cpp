// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <utility>

#include <tt_stl/indestructible.hpp>

#include "hostdev/profiler_zone_id.h"
#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/service.hpp"
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

CallbackHandle detail::register_callback(
    std::string name, std::function<void(const Batch<RecordType::All>&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    if (name.empty()) {
        name = "callback-" + std::to_string(++anonymous);
    }
    return static_cast<CallbackHandle>(internal::service().add_consumer(
        std::move(name), [cb = std::move(callback)](const Batch<RecordType::All>& b, uint64_t) { cb(b); }));
}

void UnregisterCallback(CallbackHandle handle) {
    internal::service().remove_consumer(static_cast<internal::ConsumerHandle>(handle));
}

bool IsActive() { return internal::service().is_active(); }

}  // namespace tt::tt_metal::experimental::streaming_profiler
