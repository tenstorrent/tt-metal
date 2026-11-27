// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/service.hpp"

#include <array>
#include <memory>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/indestructible.hpp>

#include "impl/streaming_profiler/ops_csv.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"
#include "impl/streaming_profiler/zone_csv.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::attach_capture(
    const std::vector<ChipId>& chip_ids, const tt::Cluster& cluster, const tt::llrt::RunTimeOptions& rtoptions) {
    std::lock_guard<std::mutex> lk(mu_);
    // A chip's cores by translated tile are filled before its first capture starts and never changed after, so that
    // api::detail::core_of() reads them without a lock.
    for (const ChipId chip_id : chip_ids) {
        if (static_cast<size_t>(chip_id) >= api::detail::chip_cores.size() ||
            api::detail::chip_cores[chip_id] != nullptr) {
            continue;
        }
        auto* cores = new std::array<api::Core, 64 * 64>{};
        const metal_SocDescriptor& soc = cluster.get_soc_desc(chip_id);
        std::vector<tt::umd::CoreCoord> tiles = soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
        std::ranges::copy(soc.get_cores(CoreType::ETH, CoordSystem::TRANSLATED), std::back_inserter(tiles));
        for (const tt::umd::CoreCoord& tile : tiles) {
            const tt::umd::CoreCoord logical = soc.translate_coord_to(tile, CoordSystem::LOGICAL);
            const tt::umd::CoreCoord noc0 = soc.translate_coord_to(tile, CoordSystem::NOC0);
            cores->at((tile.y * 64) + tile.x) = {
                .logical = {logical.x, logical.y}, .physical = {noc0.x, noc0.y}, .chip_id = chip_id};
        }
        api::detail::chip_cores[chip_id] = cores;
    }
    if (captures_++ != 0) {
        return;
    }
    if (!std::exchange(sinks_made_, true)) {
        const auto add = [&]<typename S>(const char* name, const std::shared_ptr<S>& sink) {
            Sink& added = sinks_.emplace_back();
            added.subscribe = [sink, name] {
                return api::RegisterCallback([sink](const typename S::Batch& batch) { (*sink)(batch); }, name);
            };
            if constexpr (requires { sink->write_csv(); }) {
                added.write_file = [sink] { sink->write_csv(); };
            }
        };
#if defined(TRACY_ENABLE)
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            add("tracy", std::make_shared<TracyConsumer>());
        }
#endif
        if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
            add("zone-csv", std::make_shared<ZoneCsvConsumer>(path));
        }
        if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
            add("ops-csv", std::make_shared<OpsCsvConsumer>(path));
        }
    }
    for (Sink& sink : sinks_) {
        sink.callback = sink.subscribe();
    }
}

void Service::detach_capture(size_t devices, const std::optional<tt::insight::ClockSyncErrorDistribution>& sync_check) {
    // The errors between the capture's timeline and independent exchanges of time between its devices, over all of
    // them. tests/ttnn/tracy/test_streaming_profiler_sync.py reads this line.
    if (sync_check) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync check on {} device(s): {} readings, error p50 {:.3f} ns, p99.9 {:.3f} ns, max "
            "{:.3f} ns",
            devices,
            sync_check->count(),
            sync_check->quantile(0.5),
            sync_check->quantile(0.999),
            sync_check->max());
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (--captures_ != 0) {
        return;
    }
    // A callback's end delivers it everything published before: a stop alone need not have.
    for (Sink& sink : sinks_) {
        sink.callback.reset();
        if (sink.write_file) {
            sink.write_file();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return captures_ > 0;
}

}  // namespace tt::tt_metal::streaming_profiler

bool tt::tt_metal::experimental::streaming_profiler::IsActive() {
    return tt::tt_metal::streaming_profiler::service().is_active();
}

std::array<const std::array<tt::tt_metal::experimental::streaming_profiler::Core, 64 * 64>*, 64>
    tt::tt_metal::experimental::streaming_profiler::detail::chip_cores{};
