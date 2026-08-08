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

constexpr uint32_t kWaitSpinIterations = 256;

}  // namespace

RealtimeProfilerService& realtime_profiler_service() {
    static RealtimeProfilerService service;
    return service;
}

void register_builtin_realtime_profiler_consumers() {
    [[maybe_unused]] static const bool registered = [] {
#if defined(TRACY_ENABLE)
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
        if (!attached_producers_.empty()) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Service destroyed with {} producer(s) still attached; a MeshDevice outlived the "
                "MetalContext.",
                attached_producers_.size());
            attached_producers_.clear();
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
        for (ProgramRecordProducer* producer : attached_producers_) {
            registration.readers.emplace_back(producer, producer->make_reader(), producer->max_batch_records());
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
    if (current_registration_ != nullptr && current_registration_->handle == handle) {
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

void RealtimeProfilerService::attach_producer(ProgramRecordProducer& producer) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        attached_producers_.insert(&producer);

        try {
            for (auto& registration : consumers_ | std::views::values) {
                std::lock_guard control_lock(registration.control_mutex);
                if (registration.retired.load(std::memory_order_acquire)) {
                    continue;
                }
                registration.readers_to_add.emplace_back(
                    &producer, producer.make_reader(), producer.max_batch_records());
                registration.control_pending.store(true, std::memory_order_release);
            }
        } catch (...) {
            for (auto& registration : consumers_ | std::views::values) {
                std::lock_guard control_lock(registration.control_mutex);
                std::erase_if(
                    registration.readers_to_add, [&](const ProducerReader& r) { return r.producer == &producer; });
            }
            attached_producers_.erase(&producer);
            throw;
        }
    }
    wake_consumers();
}

void RealtimeProfilerService::detach_producer(ProgramRecordProducer& producer) {
    {
        std::lock_guard topology_lock(topology_mutex_);
        attached_producers_.erase(&producer);

        for (auto& registration : consumers_ | std::views::values) {
            std::lock_guard control_lock(registration.control_mutex);
            if (registration.retired.load(std::memory_order_acquire)) {
                continue;
            }
            registration.producers_to_drain.push_back(&producer);
            registration.control_pending.store(true, std::memory_order_release);
        }
    }

    wake_consumers();
    producer.wait_until_no_readers();
}

void RealtimeProfilerService::wake_consumers() noexcept {
    wake_generation_.fetch_add(1, std::memory_order_release);
    wake_generation_.notify_all();
}

bool RealtimeProfilerService::is_active() const {
    std::lock_guard lock(topology_mutex_);
    return !attached_producers_.empty();
}

void RealtimeProfilerService::run_consumer(
    std::stop_token stop_token, RealtimeProfilerService::ConsumerRegistration& registration) {
    current_registration_ = &registration;
    const std::string thread_name = fmt::format("RtProfConsumer{}", registration.handle);
    tracy::SetThreadName(thread_name.c_str());

    std::vector<experimental::ProgramRealtimeRecord> records;
    std::vector<ProducerReader> readers_to_add;
    std::vector<ProgramRecordProducer*> producers_to_drain;
    // Drops noticed on a reader with nothing to deliver, carried until a batch exists to report them on.
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
        // we snapshot before checking work, so a concurrent publication or control change isn't missed.
        const uint32_t wake_token = wake_generation_.load(std::memory_order_acquire);

        if (registration.control_pending.load(std::memory_order_acquire)) {
            bool retired = false;
            {
                std::lock_guard control_lock(registration.control_mutex);
                readers_to_add.swap(registration.readers_to_add);
                producers_to_drain.swap(registration.producers_to_drain);
                retired = registration.retired.load(std::memory_order_acquire);
                registration.control_pending.store(false, std::memory_order_release);
            }

            if (retired) {
                registration.readers.clear();
                break;
            }

            std::ranges::move(readers_to_add, std::back_inserter(registration.readers));
            readers_to_add.clear();

            for (auto* producer : producers_to_drain) {
                auto it = std::ranges::find(registration.readers, producer, &ProducerReader::producer);
                if (it != registration.readers.end()) {
                    it->draining = true;
                }
            }
            producers_to_drain.clear();
        }

        bool made_progress = false;

        for (auto it = registration.readers.begin(); it != registration.readers.end();) {
            ProducerReader& producer_reader = *it;

            if (records.size() < producer_reader.max_batch_records) {
                records.resize(producer_reader.max_batch_records);
            }
            const std::span<experimental::ProgramRealtimeRecord> batch =
                producer_reader.reader.read_batch(std::span(records).first(producer_reader.max_batch_records));
            const uint64_t dropped_total = producer_reader.reader.dropped();
            pending_dropped += dropped_total - producer_reader.observed_dropped;
            producer_reader.observed_dropped = dropped_total;

            if (!batch.empty()) {
                invoke_callback(batch, std::exchange(pending_dropped, 0));
                made_progress = true;
            } else if (producer_reader.draining) {
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
    if (registration.thread.joinable()) {
        registration.thread.join();
    }

    uint64_t dropped = 0;
    for (const auto& producer_reader : registration.readers) {
        dropped += producer_reader.reader.dropped();
    }
    for (const auto& producer_reader : registration.readers_to_add) {
        dropped += producer_reader.reader.dropped();
    }
    if (dropped != 0) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Consumer {} dropped {} record(s)", registration.handle, dropped);
    }
}

}  // namespace tt::tt_metal
