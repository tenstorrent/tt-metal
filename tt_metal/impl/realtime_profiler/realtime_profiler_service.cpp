// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_service.hpp"

#include <algorithm>
#include <exception>
#include <span>
#include <string>
#include <utility>

#include <common/TracySystem.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

thread_local bool g_in_realtime_profiler_consumer = false;

constexpr uint32_t kWaitSpinIterations = 2048;

}  // namespace

RealtimeProfilerService::~RealtimeProfilerService() {
    {
        std::lock_guard lock(topology_mutex_);
        TT_FATAL(
            max_batch_records_by_ring_.empty(),
            "RealtimeProfilerService destroyed with {} attached ring(s)",
            max_batch_records_by_ring_.size());
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
        stop_registration(std::move(registration));
    }
}

tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle RealtimeProfilerService::register_consumer(
    std::unique_ptr<RealtimeProfilerConsumer> consumer) {
    TT_FATAL(consumer != nullptr, "Cannot register a null real-time profiler consumer");
    TT_FATAL(
        !g_in_realtime_profiler_consumer,
        "A real-time profiler consumer must not register consumers from its delivery thread");

    std::lock_guard lock(topology_mutex_);
    const auto handle = next_consumer_handle_++;
    auto [it, inserted] = consumers_.try_emplace(handle, handle, std::move(consumer));
    TT_FATAL(inserted, "Duplicate real-time profiler consumer handle {}", handle);
    auto& registration = it->second;

    for (const auto& [ring, max_batch_records] : max_batch_records_by_ring_) {
        auto [reader_it, reader_inserted] =
            registration.readers.try_emplace(ring, ring->make_reader(), max_batch_records);
        TT_FATAL(reader_inserted, "Duplicate real-time profiler ring during consumer registration");
        (void)reader_it;
    }

    try {
        registration.thread =
            std::jthread([this, &registration](std::stop_token stop_token) { run_consumer(stop_token, registration); });
    } catch (...) {
        consumers_.erase(it);
        throw;
    }
    return handle;
}

void RealtimeProfilerService::unregister_consumer(
    tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle) {
    TT_FATAL(
        !g_in_realtime_profiler_consumer,
        "A real-time profiler consumer must not unregister consumers from its delivery thread");

    ConsumerMap::node_type registration;
    {
        std::lock_guard lock(topology_mutex_);
        registration = consumers_.extract(handle);
    }
    stop_registration(std::move(registration));
}

void RealtimeProfilerService::attach_ring(RealtimeProfilerRecordRing& ring, size_t max_batch_records) {
    TT_FATAL(max_batch_records > 0, "Real-time profiler consumer batch size must be nonzero");
    TT_FATAL(
        max_batch_records <= ring.capacity(),
        "Real-time profiler consumer batch size {} exceeds ring capacity {}",
        max_batch_records,
        ring.capacity());

    {
        std::lock_guard topology_lock(topology_mutex_);
        auto [ring_it, inserted] = max_batch_records_by_ring_.emplace(&ring, max_batch_records);
        TT_FATAL(inserted, "Real-time profiler ring is already attached");
        (void)ring_it;

        for (auto& [handle, registration] : consumers_) {
            (void)handle;
            std::lock_guard control_lock(registration.control_mutex);
            auto [reader_it, reader_inserted] =
                registration.readers_to_add.try_emplace(&ring, ring.make_reader(), max_batch_records);
            TT_FATAL(reader_inserted, "Real-time profiler ring reader is already pending attachment");
            (void)reader_it;
            registration.control_pending.store(true, std::memory_order_release);
        }
    }
    wake_consumers();
}

void RealtimeProfilerService::detach_ring(RealtimeProfilerRecordRing& ring) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        const size_t erased = max_batch_records_by_ring_.erase(&ring);
        TT_FATAL(erased == 1, "Cannot detach an unknown real-time profiler ring");

        for (auto& [handle, registration] : consumers_) {
            (void)handle;
            std::lock_guard control_lock(registration.control_mutex);
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
    return !max_batch_records_by_ring_.empty();
}

void RealtimeProfilerService::run_consumer(
    std::stop_token stop_token, RealtimeProfilerService::ConsumerRegistration& registration) {
    g_in_realtime_profiler_consumer = true;
    const std::string thread_name = fmt::format("RtProfConsumer{}", registration.handle);
    tracy::SetThreadName(thread_name.c_str());

    std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord> records;
    std::unordered_map<RealtimeProfilerRecordRing*, RingReader> readers_to_add;
    std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

    auto invoke_records = [&](std::span<const tt::tt_metal::experimental::ProgramRealtimeRecord> batch,
                              uint64_t dropped) {
        TTZoneScopedDNC(RT_PROFILER, "Callback", 0xF032E6);
        TTZoneValueD(RT_PROFILER, batch.size());
        const tt::tt_metal::experimental::ProgramRealtimeRecordBatch argument{batch, dropped};
        try {
            registration.consumer->on_records(argument);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[Real-time profiler] Record consumer threw an exception: {}", e.what());
        } catch (...) {
            log_warning(tt::LogMetal, "[Real-time profiler] Record consumer threw an unknown exception");
        }
    };

    while (!stop_token.stop_requested()) {
        // Snapshot before checking any work condition so a concurrent publication/control update cannot be lost.
        const uint32_t wake_token = wake_generation_.load(std::memory_order_acquire);

        if (registration.control_pending.load(std::memory_order_acquire)) {
            {
                std::lock_guard control_lock(registration.control_mutex);
                readers_to_add.swap(registration.readers_to_add);
                rings_to_drain.swap(registration.rings_to_drain);
                registration.control_pending.store(false, std::memory_order_release);
            }

            while (!readers_to_add.empty()) {
                auto node = readers_to_add.extract(readers_to_add.begin());
                auto result = registration.readers.insert(std::move(node));
                TT_FATAL(result.inserted, "Duplicate real-time profiler reader attached to a consumer");
            }

            for (auto* ring : rings_to_drain) {
                auto it = registration.readers.find(ring);
                TT_FATAL(it != registration.readers.end(), "Missing real-time profiler reader during detach");
                it->second.draining = true;
            }
            rings_to_drain.clear();
        }

        bool made_progress = false;

        auto read_pass = [&](bool draining) {
            for (auto it = registration.readers.begin(); it != registration.readers.end();) {
                RingReader& ring_reader = it->second;
                if (ring_reader.draining != draining) {
                    ++it;
                    continue;
                }

                if (records.size() < ring_reader.max_batch_records) {
                    records.resize(ring_reader.max_batch_records);
                }
                std::span<tt::tt_metal::experimental::ProgramRealtimeRecord> batch;
                {
                    TTZoneScopedDN(RT_PROFILER, "ReadBatch");
                    batch = ring_reader.reader.read_batch(std::span(records).first(ring_reader.max_batch_records));
                }
                const uint64_t dropped_total = ring_reader.reader.dropped();
                registration.pending_dropped += dropped_total - ring_reader.observed_dropped;
                ring_reader.observed_dropped = dropped_total;

                if (!batch.empty()) {
                    invoke_records(batch, std::exchange(registration.pending_dropped, 0));
                    made_progress = true;
                    if (stop_token.stop_requested()) {
                        return;
                    }
                }

                if (ring_reader.draining && !ring_reader.reader.has_data()) {
                    it = registration.readers.erase(it);
                    made_progress = true;
                    continue;
                }
                if (ring_reader.reader.has_data()) {
                    made_progress = true;
                }
                ++it;
            }
        };

        // Closing managers have finite backlogs and should not wait behind unrelated active rings.
        read_pass(true);
        if (stop_token.stop_requested()) {
            break;
        }
        read_pass(false);
        if (stop_token.stop_requested()) {
            break;
        }

        if (made_progress) {
            continue;
        }

        for (uint32_t spin = 0; spin < kWaitSpinIterations; ++spin) {
            if (stop_token.stop_requested() || registration.control_pending.load(std::memory_order_acquire) ||
                wake_generation_.load(std::memory_order_acquire) != wake_token) {
                break;
            }
            ttsl::pause();
        }
        if (!stop_token.stop_requested() && !registration.control_pending.load(std::memory_order_acquire) &&
            wake_generation_.load(std::memory_order_acquire) == wake_token) {
            TTZoneScopedDN(RT_PROFILER, "Wait");
            wake_generation_.wait(wake_token, std::memory_order_acquire);
        }
    }

    g_in_realtime_profiler_consumer = false;
}

void RealtimeProfilerService::stop_registration(ConsumerMap::node_type registration) {
    if (registration.empty()) {
        return;
    }
    auto& value = registration.mapped();
    value.thread.request_stop();
    wake_consumers();
    if (value.thread.joinable()) {
        value.thread.join();
    }

    uint64_t dropped = 0;
    for (const auto& [ring, reader] : value.readers) {
        (void)ring;
        dropped += reader.reader.dropped();
    }
    for (const auto& [ring, reader] : value.readers_to_add) {
        (void)ring;
        dropped += reader.reader.dropped();
    }
    if (dropped != 0) {
        log_warning(tt::LogMetal, "[Real-time profiler] Consumer {} dropped {} record(s)", value.handle, dropped);
    }
}

}  // namespace tt::tt_metal
