// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_service.hpp"

#include <algorithm>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <string>
#include <utility>

#include <common/TracySystem.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

#if defined(TRACY_ENABLE)
#include "realtime_profiler_tracy_consumer.hpp"
#endif

namespace tt::tt_metal {

namespace {

constexpr uint32_t kWaitSpinIterations = 512;

}  // namespace

RealtimeProfilerService& realtime_profiler_service() {
    // Destroyed at exit rather than kept alive indefinitely: the destructor stops and joins every consumer thread.
    static RealtimeProfilerService service;
    return service;
}

void register_builtin_realtime_profiler_consumers() {
    [[maybe_unused]] static const bool registered = [] {
#if defined(TRACY_ENABLE)
        // Bound to DEFAULT_CONTEXT_ID: slot 0 is reserved for the silicon context and mock contexts start at 1
        // (find_free_context_id_locked), so that is the only context the profiler can run on.
        auto tracy_consumer = std::make_shared<RealtimeProfilerTracyConsumer>(DEFAULT_CONTEXT_ID);
        tracy_consumer->set_handle(experimental::RegisterProgramRealtimeProfilerCallback(
            [tracy_consumer](const experimental::ProgramRealtimeRecordBatch& batch) {
                tracy_consumer->on_records(batch);
            }));
#endif
        return true;
    }();
}

RealtimeProfilerService::~RealtimeProfilerService() {
    {
        std::lock_guard lock(topology_mutex_);
        if (!attached_rings_.empty()) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Service destroyed with {} ring(s) still attached; a MeshDevice outlived the "
                "MetalContext. Close all devices before teardown.",
                attached_rings_.size());
            attached_rings_.clear();
        }
    }

    while (true) {
        ConsumerMap::node_type registration;
        {
            std::lock_guard lock(topology_mutex_);
            if (consumers_.empty()) {
                break;
            }
            registration = consumers_.extract(consumers_.begin());
        }
        destroy_consumer(registration.mapped());
    }
}

experimental::ProgramRealtimeProfilerCallbackHandle RealtimeProfilerService::register_consumer(
    experimental::ProgramRealtimeProfilerCallback callback) {
    TT_FATAL(callback != nullptr, "Cannot register a null real-time profiler callback");
    reap_retired_consumers();

    std::lock_guard lock(topology_mutex_);
    const auto handle = next_consumer_handle_++;
    auto it = consumers_.try_emplace(handle, handle, std::move(callback)).first;

    try {
        auto& registration = it->second;
        for (const auto& [ring, max_batch_records] : attached_rings_) {
            registration.readers.emplace_back(ring, ring->make_reader(), max_batch_records);
        }
        registration.thread =
            std::jthread([this, &registration](std::stop_token stop_token) { run_consumer(stop_token, registration); });
    } catch (...) {
        consumers_.erase(it);
        throw;
    }
    return handle;
}

void RealtimeProfilerService::unregister_consumer(experimental::ProgramRealtimeProfilerCallbackHandle handle) {
    if (current_registration_ != nullptr &&
        current_registration_->handle == handle) {  // a callback unregistering itself
        current_registration_->retired.store(true, std::memory_order_release);
        current_registration_->control_pending.store(true, std::memory_order_release);
        return;
    }

    ConsumerMap::node_type registration;
    {
        std::lock_guard lock(topology_mutex_);
        registration = consumers_.extract(handle);
    }
    if (registration) {
        destroy_consumer(registration.mapped());
    }
    reap_retired_consumers();
}

void RealtimeProfilerService::attach_ring(RealtimeProfilerRecordRing& ring, size_t max_batch_records) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        attached_rings_.emplace(&ring, max_batch_records);

        try {
            for (auto& registration : consumers_ | std::views::values) {
                std::lock_guard control_lock(registration.control_mutex);
                if (registration.retired.load(std::memory_order_acquire)) {
                    continue;
                }
                registration.readers_to_add.emplace_back(&ring, ring.make_reader(), max_batch_records);
                registration.control_pending.store(true, std::memory_order_release);
            }
        } catch (...) {
            for (auto& registration : consumers_ | std::views::values) {
                std::lock_guard control_lock(registration.control_mutex);
                std::erase_if(registration.readers_to_add, [&](const RingReader& r) { return r.ring == &ring; });
            }
            attached_rings_.erase(&ring);
            throw;
        }
    }
    wake_consumers();
}

void RealtimeProfilerService::detach_ring(RealtimeProfilerRecordRing& ring) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        attached_rings_.erase(&ring);

        for (auto& registration : consumers_ | std::views::values) {
            std::lock_guard control_lock(registration.control_mutex);
            if (registration.retired.load(std::memory_order_acquire)) {
                continue;
            }
            registration.rings_to_drain.push_back(&ring);
            registration.control_pending.store(true, std::memory_order_release);
        }
    }

    wake_consumers();
    ring.wait_until_no_readers();
}

void RealtimeProfilerService::wake_consumers() noexcept {
    wake_generation_.fetch_add(1, std::memory_order_release);
    wake_generation_.notify_all();
}

bool RealtimeProfilerService::is_active() const {
    std::lock_guard lock(topology_mutex_);
    return !attached_rings_.empty();
}

void RealtimeProfilerService::run_consumer(
    std::stop_token stop_token, RealtimeProfilerService::ConsumerRegistration& registration) {
    current_registration_ = &registration;
    const std::string thread_name = fmt::format("RtProfConsumer{}", registration.handle);
    tracy::SetThreadName(thread_name.c_str());

    std::vector<experimental::ProgramRealtimeRecord> records;
    std::vector<RingReader> readers_to_add;
    std::vector<RealtimeProfilerRecordRing*> rings_to_drain;
    // Losses seen across all readers since the last batch was handed to the callback. Drops can be noticed on a reader
    // that has nothing to deliver, so they are carried until a batch exists to report them on.
    uint64_t pending_dropped = 0;

    auto invoke_callback = [&](std::span<const experimental::ProgramRealtimeRecord> batch, uint64_t dropped) {
        TTZoneScopedDNC(RT_PROFILER, "Callback", 0xF032E6);
        TTZoneValueD(RT_PROFILER, batch.size());
        if (TTZoneIsActiveD(RT_PROFILER) && dropped > 0) {
            const auto dropped_txt = fmt::format("dropped {}", dropped);
            TTZoneTextD(RT_PROFILER, dropped_txt.c_str(), dropped_txt.size());
        }
        const experimental::ProgramRealtimeRecordBatch argument{.records = batch, .dropped = dropped};
        try {
            registration.callback(argument);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[Real-time profiler] Record callback threw an exception: {}", e.what());
        } catch (...) {
            log_warning(tt::LogMetal, "[Real-time profiler] Record callback threw an unknown exception");
        }
    };

    while (!stop_token.stop_requested()) {
        // Snapshot before checking any work condition, so a concurrent publication or control change is not missed.
        const uint32_t wake_token = wake_generation_.load(std::memory_order_acquire);

        if (registration.control_pending.load(std::memory_order_acquire)) {
            bool retired = false;
            {
                std::lock_guard control_lock(registration.control_mutex);
                readers_to_add.swap(registration.readers_to_add);
                rings_to_drain.swap(registration.rings_to_drain);
                retired = registration.retired.load(std::memory_order_acquire);
                registration.control_pending.store(false, std::memory_order_release);
            }

            if (retired) {
                registration.readers.clear();
                break;
            }

            std::ranges::move(readers_to_add, std::back_inserter(registration.readers));
            readers_to_add.clear();

            for (auto* ring : rings_to_drain) {
                auto it = std::ranges::find(registration.readers, ring, &RingReader::ring);
                if (it != registration.readers.end()) {
                    it->draining = true;
                }
            }
            rings_to_drain.clear();
        }

        bool made_progress = false;

        for (auto it = registration.readers.begin(); it != registration.readers.end();) {
            RingReader& ring_reader = *it;

            if (records.size() < ring_reader.max_batch_records) {
                records.resize(ring_reader.max_batch_records);
            }
            const std::span<experimental::ProgramRealtimeRecord> batch =
                ring_reader.reader.read_batch(std::span(records).first(ring_reader.max_batch_records));
            const uint64_t dropped_total = ring_reader.reader.dropped();
            pending_dropped += dropped_total - ring_reader.observed_dropped;
            ring_reader.observed_dropped = dropped_total;

            if (!batch.empty()) {
                invoke_callback(batch, std::exchange(pending_dropped, 0));
                made_progress = true;
            } else if (ring_reader.draining) {
                it = registration.readers.erase(it);
                made_progress = true;
                continue;
            }
            ++it;
        }
        if (stop_token.stop_requested()) {
            break;
        }

        if (made_progress) {
            continue;
        }

        const auto still_idle = [&] {
            return !stop_token.stop_requested() && !registration.control_pending.load(std::memory_order_acquire) &&
                   wake_generation_.load(std::memory_order_acquire) == wake_token;
        };
        for (uint32_t spin = 0; spin < kWaitSpinIterations && still_idle(); ++spin) {
            ttsl::pause();
        }
        if (still_idle()) {
            TTZoneScopedDN(RT_PROFILER, "Wait");
            wake_generation_.wait(wake_token, std::memory_order_acquire);
        }
    }

    current_registration_ = nullptr;
}

void RealtimeProfilerService::reap_retired_consumers() {
    // A consumer thread reaping would join itself; it is never the one that cleans up.
    if (current_registration_ != nullptr) {
        return;
    }
    std::vector<ConsumerMap::node_type> retired;
    {
        std::lock_guard lock(topology_mutex_);
        for (auto it = consumers_.begin(); it != consumers_.end();) {
            const auto next = std::next(it);
            if (it->second.retired.load(std::memory_order_acquire)) {
                retired.push_back(consumers_.extract(it));
            }
            it = next;
        }
    }
    // Outside the lock: destroy_consumer joins.
    for (auto& node : retired) {
        destroy_consumer(node.mapped());
    }
}

void RealtimeProfilerService::destroy_consumer(ConsumerRegistration& registration) {
    registration.thread.request_stop();
    wake_consumers();
    if (registration.thread.joinable()) {
        registration.thread.join();
    }

    uint64_t dropped = 0;
    for (const auto& ring_reader : registration.readers) {
        dropped += ring_reader.reader.dropped();
    }
    for (const auto& ring_reader : registration.readers_to_add) {
        dropped += ring_reader.reader.dropped();
    }
    if (dropped != 0) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Consumer {} dropped {} record(s)", registration.handle, dropped);
    }
}

}  // namespace tt::tt_metal
