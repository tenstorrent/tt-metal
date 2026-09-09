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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/indestructible.hpp>

#include "hostdevcommon/profiler_zone_id.h"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "llrt/zone_meta.hpp"

namespace api = tt::tt_metal::experimental::streaming_profiler;

namespace tt::tt_metal::experimental::streaming_profiler::detail {
std::atomic<const SiteTu*> g_site_tus[kZoneTuCount];
}

namespace tt::tt_metal::streaming_profiler {

namespace {

static_assert(api::detail::kZoneLocalBits == TT_ZONE_LOCAL_BITS);
static_assert(api::detail::kZoneTuCount == TT_ZONE_TU_COUNT);
static_assert(TT_ZONE_STALL_ID == (TT_ZONE_RESERVED_TU << TT_ZONE_LOCAL_BITS));

constexpr api::Site kStallSite{.name = api::kStallZoneName};
constexpr const api::Site* kStallSites[1] = {&kStallSite};
constexpr api::detail::SiteTu kStallTu{kStallSites};

// Builds the tables behind api::detail::site_of from the zone-name registry as ELFs load. Nothing is ever freed: a
// record may hold a Site's address for the life of the process, and a reader may still be walking a replaced table.
class SiteTables {
public:
    void add(std::span<const tt::llrt::ZoneMetaEntry* const> entries) {
        std::lock_guard<std::mutex> lk(mu_);
        std::map<uint32_t, std::vector<const api::Site*>> grown;
        for (const tt::llrt::ZoneMetaEntry* e : entries) {
            const uint32_t tu = TT_ZONE_TU_OF(e->zone_id), local = TT_ZONE_LOCAL_OF(e->zone_id);
            auto [it, fresh] = grown.try_emplace(tu);
            if (fresh) {
                if (const api::detail::SiteTu* cur = api::detail::g_site_tus[tu].load(std::memory_order_relaxed)) {
                    it->second.assign(cur->sites.begin(), cur->sites.end());
                }
            }
            if (local >= it->second.size()) {
                it->second.resize(local + 1, nullptr);
            }
            if (it->second[local] == nullptr) {
                it->second[local] =
                    &sites_.emplace_back(api::Site{.name = e->name, .location = {.file = e->file, .line = e->line}});
            }
        }
        for (auto& [tu, v] : grown) {
            auto arr = std::make_unique<const api::Site*[]>(v.size());
            std::copy(v.begin(), v.end(), arr.get());
            auto t = std::make_unique<api::detail::SiteTu>(
                api::detail::SiteTu{std::span<const api::Site* const>(arr.get(), v.size())});
            api::detail::g_site_tus[tu].store(t.get(), std::memory_order_release);
            arrays_.push_back(std::move(arr));
            tus_.push_back(std::move(t));
        }
    }

private:
    std::mutex mu_;
    std::deque<api::Site> sites_;
    std::vector<std::unique_ptr<const api::Site*[]>> arrays_;
    std::vector<std::unique_ptr<api::detail::SiteTu>> tus_;
};

}  // namespace

void init_site_registry() {
    static std::once_flag once;
    std::call_once(once, [] {
        api::detail::g_site_tus[TT_ZONE_RESERVED_TU].store(&kStallTu, std::memory_order_release);
        static ttsl::Indestructible<SiteTables> tables;
        tt::llrt::ZoneMetaRegistry::instance().set_listener(
            [](std::span<const tt::llrt::ZoneMetaEntry* const> entries) { tables.get().add(entries); });
    });
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {

namespace internal = tt::tt_metal::streaming_profiler;

CallbackHandle detail::register_callback(
    std::string_view name, std::function<void(const Batch<RecordType::All>&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    const std::string label = name.empty() ? "callback-" + std::to_string(++anonymous) : std::string(name);
    return internal::service().add_consumer(
        label, [cb = std::move(callback)](const Batch<RecordType::All>& b, uint64_t) { cb(b); });
}

void UnregisterCallback(CallbackHandle handle) { internal::service().remove_consumer(handle); }

bool IsActive() { return internal::service().is_active(); }

}  // namespace tt::tt_metal::experimental::streaming_profiler
