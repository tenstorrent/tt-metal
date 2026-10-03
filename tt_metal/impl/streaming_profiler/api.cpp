// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/profiler_zone_id.h"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/host_sync.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "llrt/zone_meta.hpp"

namespace api = tt::tt_metal::experimental::streaming_profiler;

namespace tt::tt_metal::experimental::streaming_profiler::detail {
static_assert(ZONE_ID_BITS == TT_ZONE_ID_BITS);
static_assert(ZONE_LOCAL_BITS == TT_ZONE_LOCAL_BITS);
static_assert(ZONE_TU_COUNT == TT_ZONE_TU_COUNT);
std::atomic<const SiteTu*> site_tus[ZONE_TU_COUNT];
}  // namespace tt::tt_metal::experimental::streaming_profiler::detail

namespace tt::tt_metal::streaming_profiler {

namespace {

static_assert(TT_ZONE_STALL_ID == (TT_ZONE_RESERVED_TU << TT_ZONE_LOCAL_BITS));

constexpr api::MarkerSite kStallSite{.name = api::STALL_ZONE_NAME};
constexpr const api::MarkerSite* kStallSites[1] = {&kStallSite};
constexpr api::detail::SiteTu kStallTu{kStallSites};

// Nothing is ever freed: a record can hold a site's address for the life of the process, and a reader may still be
// walking a table that's been replaced.
class SiteTables {
public:
    void add(std::span<const tt::llrt::ZoneMetaEntry* const> entries) {
        std::lock_guard<std::mutex> lk(mu_);
        std::map<uint32_t, std::vector<const api::MarkerSite*>> grown;
        for (const tt::llrt::ZoneMetaEntry* e : entries) {
            const uint32_t tu = TT_ZONE_TU_OF(e->zone_id), local = TT_ZONE_LOCAL_OF(e->zone_id);
            auto [it, fresh] = grown.try_emplace(tu);
            if (fresh) {
                if (const api::detail::SiteTu* cur = api::detail::site_tus[tu].load(std::memory_order_relaxed)) {
                    it->second.assign(cur->sites.begin(), cur->sites.end());
                }
            }
            if (local >= it->second.size()) {
                it->second.resize(local + 1, nullptr);
            }
            if (it->second[local] == nullptr) {
                it->second[local] = &sites_.emplace_back(
                    api::MarkerSite{.name = e->name, .location = {.file = e->file, .line = e->line}});
            }
        }
        for (auto& [tu, v] : grown) {
            Tu& table = tus_.emplace_back(Tu{.sites = std::move(v)});
            table.view.sites = table.sites;
            api::detail::site_tus[tu].store(&table.view, std::memory_order_release);
        }
    }

private:
    struct Tu {
        std::vector<const api::MarkerSite*> sites;
        api::detail::SiteTu view;
    };
    std::mutex mu_;
    std::deque<api::MarkerSite> sites_;
    std::deque<Tu> tus_;
};

}  // namespace

void init_site_registry() {
    api::detail::site_tus[TT_ZONE_RESERVED_TU].store(&kStallTu, std::memory_order_release);
    static ttsl::Indestructible<SiteTables> tables;
    tt::llrt::ZoneMetaRegistry::instance().set_listener(
        [](std::span<const tt::llrt::ZoneMetaEntry* const> entries) { tables.get().add(entries); });
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {

namespace internal = tt::tt_metal::streaming_profiler;

detail::CallbackId detail::register_callback(std::string name, std::function<void(const BatchData&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    if (name.empty()) {
        name = "callback-" + std::to_string(++anonymous);
    }
    return internal::service().add_consumer(std::move(name), std::move(callback));
}

void Callback::reset() noexcept {
    if (id_ != detail::CallbackId{}) {
        internal::service().remove_consumer(std::exchange(id_, detail::CallbackId{}));
    }
}

bool IsActive() { return internal::service().is_active(); }

double NsPerTscTick() noexcept { return 1.0 / internal::tsc_ticks_per_ns(); }

namespace detail {
std::chrono::steady_clock::time_point tsc_to_steady(int64_t tsc) {
    const std::optional<int64_t> ns = internal::service().steady().ns(tsc);
    TT_FATAL(ns, "streaming profiler: no steady_clock pair yet, before any capture");
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(*ns));
}
}  // namespace detail

}  // namespace tt::tt_metal::experimental::streaming_profiler
