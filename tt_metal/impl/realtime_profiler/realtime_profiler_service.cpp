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
#include <tt_stl/indestructible.hpp>

#include "tt_metal/impl/realtime_profiler/realtime_profiler_tracy_consumer.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

RealtimeProfilerService& realtime_profiler_service() {
    static ttsl::Indestructible<RealtimeProfilerService> service;
    return service.get();
}

namespace {

#if defined(TRACY_ENABLE)
constexpr bool kTracyEnabled = true;
#else
constexpr bool kTracyEnabled = false;
#endif
constexpr uint64_t kNoConsumerHandle = ~uint64_t{0};

}  // namespace

// The Tracy consumer, when Tracy is built in, is registered first, so it gets the first handle.
RealtimeProfilerService::RealtimeProfilerService() :
    tracy_handle_(kTracyEnabled ? next_consumer_handle_ : kNoConsumerHandle),
    tracy_(std::make_unique<RealtimeProfilerTracyConsumer>()) {
#if defined(TRACY_ENABLE)
    // The Tracy consumer unregisters itself once Tracy will take no more records.
    register_consumer([this](const experimental::ProgramRealtimeRecordBatch& batch) {
        if (!tracy_->on_records(batch)) {
            unregister_consumer(current_registration_->handle);
        }
    });
#endif
}

uint64_t RealtimeProfilerService::register_consumer(const experimental::ProgramRealtimeProfilerCallback& callback) {
    TT_FATAL(callback != nullptr, "Cannot register a null real-time profiler callback");
    reap_retired_consumers();

    std::lock_guard lock(topology_mutex_);
    const auto handle = next_consumer_handle_++;
    auto& registration = consumers_.try_emplace(handle, handle, callback).first->second;
    for (const AttachedProducer& producer : attached_producers_) {
        registration.readers.emplace_back(producer.ring, producer.ring->make_reader(), producer.max_batch_records);
    }
    registration.thread = std::jthread(
        [this, &registration](const std::stop_token& stop_token) { run_consumer(stop_token, registration); });
    num_live_consumers_.fetch_add(1, std::memory_order_relaxed);
    return handle;
}

void RealtimeProfilerService::unregister_consumer(uint64_t handle) {
    if (current_registration_ != nullptr && current_registration_->handle == handle) {
        // exchange, not store: an external unregister may race this self-retire across the whole
        // callback invocation, and only the one thread that flips retired may decrement.
        if (!current_registration_->retired.exchange(true, std::memory_order_acq_rel)) {
            num_live_consumers_.fetch_sub(1, std::memory_order_relaxed);
        }
        current_registration_->control_pending.store(true, std::memory_order_release);
        return;
    }

    ConsumerMap::node_type registration;
    {
        std::lock_guard lock(topology_mutex_);
        registration = consumers_.extract(handle);
    }
    if (registration) {
        if (!registration.mapped().retired.exchange(true, std::memory_order_acq_rel)) {
            num_live_consumers_.fetch_sub(1, std::memory_order_relaxed);
        }
        destroy_consumer(registration.mapped());
    }
    reap_retired_consumers();
}

void RealtimeProfilerService::attach_producer(RealtimeProfilerRecordRing& ring, size_t max_batch_records) noexcept {
    {
        std::lock_guard topology_lock(topology_mutex_);
        attached_producers_.push_back(AttachedProducer{.ring = &ring, .max_batch_records = max_batch_records});
        num_attached_producers_.store(attached_producers_.size(), std::memory_order_relaxed);

        for (auto& registration : consumers_ | std::views::values) {
            std::lock_guard control_lock(registration.control_mutex);
            if (registration.retired.load(std::memory_order_acquire)) {
                continue;
            }
            registration.readers_to_add.emplace_back(&ring, ring.make_reader(), max_batch_records);
            registration.control_pending.store(true, std::memory_order_release);
        }
    }
    wake_consumers();
}

void RealtimeProfilerService::detach_producer(RealtimeProfilerRecordRing& ring) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        std::erase_if(attached_producers_, [&](const AttachedProducer& p) { return p.ring == &ring; });
        num_attached_producers_.store(attached_producers_.size(), std::memory_order_relaxed);

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

void RealtimeProfilerService::run_consumer(const std::stop_token& stop_token, ConsumerRegistration& registration) {
    current_registration_ = &registration;
    const std::string thread_name = fmt::format("RtProfilerConsumer{}", registration.handle);
    tracy::SetThreadName(thread_name.c_str());

    while (true) {
        // Taken before looking for work, so a publication, control change or stop request that comes after it ends
        // the wait below.
        const uint32_t wake_token = wake_generation_.load(std::memory_order_acquire);
        if (stop_token.stop_requested() || !apply_control(registration)) {
            break;
        }
        if (deliver(registration)) {
            continue;
        }
        if (registration.handle == tracy_handle_) {
            tracy_->on_idle();
        }
        TTZoneScopedDN(RT_PROFILER, "Wait");
        wake_generation_.wait(wake_token, std::memory_order_acquire);
    }

    current_registration_ = nullptr;
}

bool RealtimeProfilerService::apply_control(ConsumerRegistration& registration) {
    if (!registration.control_pending.load(std::memory_order_acquire)) {
        return true;
    }
    std::vector<ProducerReader> readers_to_add;
    std::vector<RealtimeProfilerRecordRing*> rings_to_drain;
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
        return false;
    }
    std::ranges::move(readers_to_add, std::back_inserter(registration.readers));
    for (auto* ring : rings_to_drain) {
        std::ranges::find(registration.readers, ring, &ProducerReader::ring)->draining = true;
    }
    return true;
}

bool RealtimeProfilerService::deliver(ConsumerRegistration& registration) {
    auto& records = registration.records;
    bool made_progress = false;
    for (auto it = registration.readers.begin(); it != registration.readers.end();) {
        // A callback that retired itself must not run again for a later reader in the same pass; unregister
        // promises no invocation after the retiring one ends.
        if (registration.retired.load(std::memory_order_acquire)) {
            break;
        }
        ProducerReader& producer_reader = *it;

        if (records.size() < producer_reader.max_batch_records) {
            records.resize(producer_reader.max_batch_records);
        }
        const std::span<experimental::ProgramRealtimeRecord> batch =
            producer_reader.reader.read_batch(std::span(records).first(producer_reader.max_batch_records));
        const uint64_t dropped_total = producer_reader.reader.dropped();
        registration.pending_dropped += dropped_total - producer_reader.observed_dropped;
        producer_reader.observed_dropped = dropped_total;

        if (!batch.empty()) {
            invoke_callback(registration, batch, std::exchange(registration.pending_dropped, 0));
            made_progress = true;
        } else if (producer_reader.draining) {
            it = registration.readers.erase(it);
            made_progress = true;
            continue;
        }
        ++it;
    }
    return made_progress;
}

void RealtimeProfilerService::invoke_callback(
    ConsumerRegistration& registration, std::span<const experimental::ProgramRealtimeRecord> batch, uint64_t dropped) {
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
        log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an exception: {}", e.what());
    } catch (...) {
        log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an unknown exception");
    }
}

void RealtimeProfilerService::reap_retired_consumers() {
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
    for (auto& node : retired) {
        destroy_consumer(node.mapped());
    }
}

void RealtimeProfilerService::destroy_consumer(ConsumerRegistration& registration) {
    registration.thread.request_stop();
    wake_consumers();
    registration.thread.join();

    uint64_t dropped = 0;
    for (const auto& producer_reader : registration.readers) {
        dropped += producer_reader.reader.dropped();
    }
    if (dropped != 0) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Callback {} dropped {} record(s)", registration.handle, dropped);
    }
}

}  // namespace tt::tt_metal
