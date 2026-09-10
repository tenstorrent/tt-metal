// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <thread>
#include <utility>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "llrt/rtoptions.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_ops_csv.hpp"
#include "impl/streaming_profiler/streaming_profiler_tracy.hpp"
#include "impl/streaming_profiler/streaming_profiler_zone_csv.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

thread_local bool t_in_consumer = false;

}  // namespace

namespace api = experimental::streaming_profiler;

// A consumer's position in one stream and the decoder that turns its frames into records.
struct Service::AttachedStream {
    profiler::SpanDecodeState state;
    std::vector<profiler::SpscRecConsts> lanes;
    StreamDecoder dec;
    uint64_t cursor = 0;   // bytes, absolute: the next frame this consumer reads
    uint64_t dropped = 0;  // bytes of frames this consumer never saw
    uint64_t stalls_reported = 0;
};
struct Service::Attached {
    Producer* producer = nullptr;
    uint64_t capture = 0;
    std::vector<std::unique_ptr<AttachedStream>> streams;  // stable: dec.st points into the stream
};

// Frames decoded per delivery; bounds the consumer's scratch and the callback's batch.
constexpr uint32_t kBatchFrames = 64;

struct Service::Consumer {
    std::string name;
    BatchCallback cb;
    ConsumerHooks hooks;
    uint64_t captures = 0;  // producers attached so far; the capture number its callback sees
    ConsumerHandle handle = 0;
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> control_pending{false};
    std::mutex control_mu;
    std::vector<std::pair<Producer*, bool>> control;  // (producer, attach), in order
    std::atomic<uint64_t> dropped{0};
};

Service::Service() { init_site_registry(); }

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::post_control(Consumer& c, Producer* producer, bool attach) {
    {
        std::lock_guard<std::mutex> lk(c.control_mu);
        c.control.emplace_back(producer, attach);
    }
    c.control_pending.store(true, std::memory_order_release);
    pending_acks_++;
    wake_consumers();
}

void Service::wait_acks(std::unique_lock<std::mutex>& lk) {
    ack_cv_.wait(lk, [&] { return pending_acks_ == 0; });
}

ConsumerHandle Service::add_consumer(std::string name, BatchCallback cb) {
    return add_consumer(std::move(name), std::move(cb), ConsumerHooks{});
}

ConsumerHandle Service::add_consumer(std::string name, BatchCallback cb, ConsumerHooks hooks) {
    TT_FATAL(!t_in_consumer, "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->cb = std::move(cb);
    c->hooks = std::move(hooks);
    Consumer& ref = *c;
    std::unique_lock<std::mutex> lk(mu_);
    ref.handle = next_handle_++;
    for (Producer* p : producers_) {
        post_control(ref, p, true);
    }
    consumers_.push_back(std::move(c));
    ref.thread = std::thread(&Service::consumer_thread, this, std::ref(ref));
    wait_acks(lk);
    return ref.handle;
}

void Service::remove_consumer(ConsumerHandle handle) {
    TT_FATAL(!t_in_consumer, "streaming profiler: remove_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it =
            std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& c) { return c->handle == handle; });
        TT_FATAL(it != consumers_.end(), "streaming profiler: unknown consumer handle {}", handle);
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->stop.store(true, std::memory_order_release);
    wake_consumers();
    victim->thread.join();
    warn_missed(*victim);
}

void Service::attach_producer(Producer& producer) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    TT_FATAL(
        std::find(producers_.begin(), producers_.end(), &producer) == producers_.end(),
        "streaming profiler: producer attached twice");
    producers_.push_back(&producer);
    for (auto& c : consumers_) {
        post_control(*c, &producer, true);
    }
    wait_acks(lk);
}

void Service::detach_producer(Producer& producer) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        auto it = std::find(producers_.begin(), producers_.end(), &producer);
        TT_FATAL(it != producers_.end(), "streaming profiler: detaching a producer that is not attached");
        for (auto& c : consumers_) {
            post_control(*c, &producer, false);
        }
        wait_acks(lk);
        producers_.erase(it);
        last = producers_.empty();
    }
    if (last) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& c : consumers_) {
                warn_missed(*c);
            }
        }
        for (const auto& write : file_sinks_) {
            write();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !producers_.empty();
}

void Service::warn_missed(const Consumer& c) {
    if (const uint64_t dropped = c.dropped.load(std::memory_order_relaxed); dropped != 0) {
        log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" missed {} bytes of frames", c.name, dropped);
    }
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    std::call_once(builtins_once_, [&] {
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            tracy_ = std::make_unique<TracySink>(*this);
        }
        {
            // Device<->device sync: consumes only the PP_CLOCK samples (idle-eth trackers and link stamps), fits
            // at capture end. Its batch callback is a no-op; the decode pass is what routes the samples to it.
            auto c = std::make_shared<D2dSyncConsumer>();
            add_consumer(
                "d2d-sync",
                [](const api::Batch<api::RecordType::All>&, uint64_t) {},
                ConsumerHooks{
                    .on_attach = [c](const CaptureContext& ctx) { c->on_attach(ctx); },
                    .clock_sink = [c](const ClockSample& cs) { c->on_clock(cs); },
                    .on_capture_end = [c](const CaptureContext& ctx) { c->on_capture_end(ctx); }});
        }
        auto add_public = [&]<typename C>(const char* name, std::shared_ptr<C> c) {
            using B = typename C::Batch;
            add_consumer(name, [c](const api::Batch<api::RecordType::All>& full, uint64_t) {
                (*c)(api::Batch<B::types>(full));
            });
        };
        if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
            auto c = std::make_shared<ZoneCsvConsumer>(path);
            add_public("zone-csv", c);
            file_sinks_.push_back([c] { c->write_csv(); });
        }
        if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
            auto c = std::make_shared<OpsCsvConsumer>(path);
            add_public("ops-csv", c);
            file_sinks_.push_back([c] { c->write_csv(); });
        }
    });
}

void Service::consumer_thread(Consumer& c) {
    const std::string name = "sp-con:" + c.name;
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    t_in_consumer = true;
    std::vector<Attached> attached;
    const size_t rec_bytes =
        (size_t{kBatchFrames} * kFrameRecsReserve + profiler::kSpscSinkSlackRecs) * profiler::kSpscRecBytes;
    const size_t data_bytes = size_t{kBatchFrames} * kFrameDataBytesReserve;
    auto zones_buf = std::make_unique_for_overwrite<uint8_t[]>(rec_bytes);
    auto events_buf = std::make_unique_for_overwrite<uint8_t[]>(rec_bytes);
    auto data_buf = std::make_unique_for_overwrite<uint8_t[]>(data_bytes);
    constexpr size_t kFramesBytes = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords * 4;
    std::array<uint32_t, kBatchFrames> frame_words;
    auto frames_buf = std::make_unique_for_overwrite<std::byte[]>(kFramesBytes);

    auto attach = [&](Producer* p) {
        Attached a;
        a.producer = p;
        const CaptureContext& ctx = p->capture_context();
        for (const ProducerStream& ps : p->streams()) {
            auto s = std::make_unique<AttachedStream>();
            s->cursor = ps.walked->load(std::memory_order_acquire);
            const CaptureContext::Device& dev = ctx.devices[ps.dev];
            s->state.reset(dev.lanes.size() / profiler::kSpscNRiscDecode);
            s->state.core_of_xy.load(dev.core_xy);
            s->lanes.reserve(dev.lanes.size());
            for (const auto& core : dev.lanes) {
                s->lanes.push_back(record_consts(core, p->clock(ps.dev)));
            }
            s->dec.st = &s->state;
            s->dec.lanes = s->lanes.data();
            s->dec.dev = ps.dev;  // stamped on ClockSamples the decoder routes to the clock sink
            if (c.hooks.clock_sink) {
                s->dec.clock_ctx = &c;
                s->dec.clock_fn = [](void* ctx, const ClockSample& cs) {
                    static_cast<Consumer*>(ctx)->hooks.clock_sink(cs);
                };
            }
            a.streams.push_back(std::move(s));
        }
        if (c.hooks.on_attach) {
            c.hooks.on_attach(ctx);
        }
        a.capture = ++c.captures;
        attached.push_back(std::move(a));
    };

    auto deliver = [&](Attached& a, AttachedStream& s, const StreamDecoder::Produced& n, uint64_t dropped) {
        api::Batch<api::RecordType::All> b;
        b.zones_ = std::span<const api::Zone>(reinterpret_cast<const api::Zone*>(zones_buf.get()), n.zones);
        b.events_ = std::span<const api::Event>(reinterpret_cast<const api::Event*>(events_buf.get()), n.events);
        b.timestamped_data_ = std::ranges::subrange(
            api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(data_buf.get())),
            api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(data_buf.get()) + n.data_bytes));
        b.dropped_ = dropped;
        b.stall_count_ = s.dec.stall_zones - s.stalls_reported;
        s.stalls_reported = s.dec.stall_zones;
        try {
            c.cb(b, a.capture);
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", c.name, ex.what());
        }
        s.dec.commit();
    };

    // One delivery of up to kBatchFrames frames from the stream, read in place up to the ingest's walk position.
    // False when the stream had nothing new.
    auto read = [&](Attached& a, AttachedStream& s, uint32_t stream_index) {
        const ProducerStream& ps = a.producer->streams()[stream_index];
        const Walked w = walk_frames(
            ps.fifo,
            s.cursor,
            ps.walked->load(std::memory_order_acquire),
            ps.marks,
            std::span<std::byte>(frames_buf.get(), kFramesBytes),
            frame_words,
            [&] { return a.producer->live_head(stream_index); });
        if (w.cursor == s.cursor) {
            return false;
        }
        s.cursor = w.cursor;
        s.dropped += w.dropped;
        StreamDecoder::Produced n{0, 0, 0};
        const std::byte* p = frames_buf.get();
        for (uint32_t i = 0; i < w.frames; i++) {
            const uint32_t fw = frame_words[i];
            const auto out = s.dec.decode_frame(
                reinterpret_cast<const uint32_t*>(p),
                fw,
                {zones_buf.get() + size_t{n.zones} * profiler::kSpscRecBytes,
                 events_buf.get() + size_t{n.events} * profiler::kSpscRecBytes,
                 data_buf.get() + n.data_bytes});
            n.zones += out.zones;
            n.events += out.events;
            n.data_bytes += out.data_bytes;
            p += size_t{fw} * 4;
        }
        deliver(a, s, n, w.dropped);
        return true;
    };
    // One delivery per stream per round, so no stream's ring laps while an earlier one is drained to empty.
    auto pass = [&](Attached& a) {
        bool any = false;
        for (bool progress = true; progress;) {
            progress = false;
            for (uint32_t i = 0; i < a.streams.size(); i++) {
                progress |= read(a, *a.streams[i], i);
            }
            any |= progress;
        }
        return any;
    };

    auto detach = [&](Producer* p) {
        auto it = std::find_if(attached.begin(), attached.end(), [&](const Attached& a) { return a.producer == p; });
        if (it == attached.end()) {
            return;
        }
        Attached& a = *it;
        while (pass(a)) {
        }
        if (c.hooks.on_capture_end) {
            c.hooks.on_capture_end(p->capture_context());
        }
        for (size_t i = 0; i < a.streams.size(); i++) {
            c.dropped.fetch_add(a.streams[i]->dropped, std::memory_order_relaxed);
            p->finish_stream(static_cast<uint32_t>(i), a.streams[i]->dropped, a.streams[i]->dec.stats);
        }
        attached.erase(it);
    };

    auto handle_control = [&](bool run) {
        std::vector<std::pair<Producer*, bool>> control;
        {
            std::lock_guard<std::mutex> lk(c.control_mu);
            control.swap(c.control);
        }
        if (run) {
            for (const auto& [p, is_attach] : control) {
                is_attach ? attach(p) : detach(p);
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        pending_acks_ -= control.size();
        ack_cv_.notify_all();
    };

    auto pass_all = [&] {
        bool any = false;
        for (Attached& a : attached) {
            any |= pass(a);
        }
        return any;
    };
    IdleBackoff backoff(0);
    for (;;) {
        const uint32_t seen = wake_token();
        if (c.control_pending.load(std::memory_order_acquire)) {
            c.control_pending.store(false, std::memory_order_release);
            handle_control(true);
        }
        if (pass_all()) {
            backoff.reset();
            continue;
        }
        if (c.stop.load(std::memory_order_acquire)) {
            break;
        }
        if (backoff.spin()) {
            continue;
        }
        wait_wake(seen);
    }
    // Stopped mid-capture: the readers go before the queues they read.
    for (const Attached& a : attached) {
        for (const auto& s : a.streams) {
            c.dropped.fetch_add(s->dropped, std::memory_order_relaxed);
        }
    }
    attached.clear();
    handle_control(false);
}

}  // namespace tt::tt_metal::streaming_profiler
