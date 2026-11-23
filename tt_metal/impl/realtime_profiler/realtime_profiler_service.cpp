// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <enchantum/enchantum.hpp>
#include <fmt/core.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/math.hpp>
#include <internal/mesh_device.hpp>
#include <tt_metal.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/xy_pair.hpp>

#include <common/TracySystem.hpp>
#include <llrt/tt_cluster.hpp>
#include <tracy/TracyTTDevice.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/hal.hpp"
#include "program/program_impl.hpp"
#include "program/slow_dispatch.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_tracy_consumer.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"                // tt::tt_metal::SyncInfo, DeviceProfiler
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"  // ProfilerStateManager
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal::distributed {

void RealtimeProfilerManager::run_consumer(Consumer& consumer) {
    tracy::SetThreadName(fmt::format("RtProfilerConsumer{}", consumer.handle).c_str());
    std::vector<tt::ProgramRealtimeRecord> records(
        std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * devices_.size()));
    uint64_t reported_dropped = 0;

    auto deliver_batch = [&](std::span<const tt::ProgramRealtimeRecord> batch, uint64_t dropped_total) {
        TTZoneScopedDNC(RT_PROFILER, "Callback", 0xF032E6);
        TTZoneValueD(RT_PROFILER, batch.size());
        const uint64_t dropped_delta = dropped_total - reported_dropped;
        if (TTZoneIsActiveD(RT_PROFILER) && dropped_delta > 0) {
            const auto dropped_txt = fmt::format("dropped {}", dropped_delta);
            TTZoneTextD(RT_PROFILER, dropped_txt.c_str(), dropped_txt.size());
        }
        const tt::ProgramRealtimeRecordBatch arg{batch, dropped_delta};
        reported_dropped = dropped_total;
        try {
            consumer.callback(arg);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an exception: {}", e.what());
        } catch (...) {
            log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an unknown exception");
        }
    };

    while (true) {
        // stop_consumer sets the stop mode then wakes, so waiting on a token sampled only inside wait() could miss that
        // wake and hang
        const auto token = consumer.reader.wait_token();
        std::span<tt::ProgramRealtimeRecord> batch = consumer.reader.read_batch(records);
        const uint64_t dropped_total = consumer.reader.dropped();
        const ConsumerStopMode stop_mode = consumer.stop_mode.load(std::memory_order_acquire);
        if (stop_mode == ConsumerStopMode::StopWithoutDrain) {
            break;
        }
        if (!batch.empty()) {
            deliver_batch(batch, dropped_total);
        } else if (stop_mode == ConsumerStopMode::DrainThenStop) {
            break;
        } else {
            TTZoneScopedDN(RT_PROFILER, "Wait");
            consumer.reader.wait(token);
        }
    }
    consumer.dropped = consumer.reader.dropped();
}

void RealtimeProfilerManager::stop_consumer(Consumer& consumer, ConsumerStopMode stop_mode) {
    consumer.stop_mode.store(stop_mode, std::memory_order_release);
    ring_->writer().wake_readers();
    if (consumer.thread.joinable()) {
        consumer.thread.join();
    }
}

void RealtimeProfilerManager::on_callback_registered(
    tt::ProgramRealtimeProfilerCallbackHandle handle, const tt::ProgramRealtimeProfilerCallback& callback) {
    auto consumer = std::make_unique<Consumer>(ring_->make_reader(), callback, handle);
    Consumer* raw = consumer.get();
    std::lock_guard<std::mutex> lock(consumers_mutex_);
    const auto caller = std::this_thread::get_id();
    const bool from_callback_thread =
        std::ranges::any_of(consumers_, [caller](const auto& kv) { return kv.second->thread.get_id() == caller; });
    TT_FATAL(!from_callback_thread, "A real-time profiler callback must not register callbacks");
    consumers_.emplace(handle, std::move(consumer));
    raw->thread = std::thread([this, raw]() { run_consumer(*raw); });
}

void RealtimeProfilerManager::on_callback_unregistered(tt::ProgramRealtimeProfilerCallbackHandle handle) {
    std::unique_ptr<Consumer> consumer;
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        auto it = consumers_.find(handle);
        if (it == consumers_.end()) {
            return;
        }
        const auto caller = std::this_thread::get_id();
        const bool from_callback_thread =
            std::ranges::any_of(consumers_, [caller](const auto& kv) { return kv.second->thread.get_id() == caller; });
        TT_FATAL(!from_callback_thread, "A real-time profiler callback must not unregister callbacks");
        consumer = std::move(it->second);
        consumers_.erase(it);
    }
    stop_consumer(*consumer, ConsumerStopMode::StopWithoutDrain);
    const uint64_t dropped = consumer->dropped;
    if (dropped > 0) {
        log_warning(tt::LogMetal, "[Real-time profiler] Callback {} dropped {} record(s)", handle, dropped);
    }
}

}  // namespace tt::tt_metal::distributed
