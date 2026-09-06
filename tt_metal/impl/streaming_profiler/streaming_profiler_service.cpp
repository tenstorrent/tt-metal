// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_service.hpp"

#include <algorithm>
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

// One consumer thread's reader, walker and decoder for one of a producer's streams.
struct AttachedStream {
    BroadcastRing<RingLine>::Reader reader;
    FrameWalker walker;
    StreamDecoder dec;
    profiler::SpanDecodeState state;

    AttachedStream(
        BroadcastRing<RingLine>::Reader r, const CaptureContext::Device& dev, uint32_t dev_index, uint8_t* scratch) :
        reader(std::move(r)) {
        state.reset(dev.lanes.size() / profiler::kSpscNRiscDecode);
        state.core_of_xy.load(dev.core_xy);
        dec.st = &state;
        dec.dev = dev_index;
        dec.sink.buf = scratch;
    }
};
struct Attached {
    Producer* producer = nullptr;
    std::vector<AttachedStream> streams;  // never resized after attach: dec.st points into it
};

}  // namespace

struct Service::Consumer {
    std::string name;
    RecordCallback cb;
    ConsumerHandle handle = 0;
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> control_pending{false};
    std::mutex control_mu;
    std::vector<std::pair<Producer*, bool>> control;  // (producer, attach), in order
    std::atomic<uint64_t> dropped{0};
};

Service::Service() = default;

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

Service::~Service() {
    tracy_.reset();  // unsubscribes, so before the consumer list goes
    std::vector<std::unique_ptr<Consumer>> consumers;
    {
        std::lock_guard<std::mutex> lk(mu_);
        consumers.swap(consumers_);
    }
    for (auto& c : consumers) {
        c->stop.store(true, std::memory_order_release);
    }
    wake_consumers();
    for (auto& c : consumers) {
        if (c->thread.joinable()) {
            c->thread.join();
        }
    }
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

ConsumerHandle Service::add_consumer(std::string name, RecordCallback cb) {
    TT_FATAL(!t_in_consumer, "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->cb = std::move(cb);
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
    if (const uint64_t dropped = victim->dropped.load(std::memory_order_relaxed); dropped != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] consumer \"{}\" removed having dropped {} records",
            victim->name,
            dropped);
    }
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
        producers_.erase(std::find(producers_.begin(), producers_.end(), &producer));
        last = producers_.empty();
    }
    if (last) {
        log_consumer_drops();
        for (const auto& write : file_sinks_) {
            write();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !producers_.empty();
}

void Service::log_consumer_drops() const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& c : consumers_) {
        if (const uint64_t dropped = c->dropped.load(std::memory_order_relaxed); dropped != 0) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" dropped {} records", c->name, dropped);
        }
    }
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (builtins_registered_) {
            return;
        }
        builtins_registered_ = true;
    }
    namespace api = experimental::streaming_profiler;
    std::vector<std::function<void()>> sinks;
    if (rtoptions.get_streaming_profiler_tracy_enabled()) {
        tracy_ = std::make_unique<TracySink>(*this);
    }
    auto add_public = [&]<typename C>(const char* name, std::shared_ptr<C> c) {
        using B = typename C::Batch;
        add_consumer(name, make_public_adapter(B::channels, [c](const api::Batch<api::Channel::All>& full) {
                         (*c)(api::detail::narrow<B::channels>(full));
                     }));
    };
    if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
        auto c = std::make_shared<ZoneCsvConsumer>();
        add_public("zone-csv", c);
        sinks.push_back([c, path] { c->write_csv(path); });
    }
    if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
        auto c = std::make_shared<OpsCsvConsumer>();
        add_public("ops-csv", c);
        sinks.push_back([c, path] { c->write_csv(path); });
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& s : sinks) {
        file_sinks_.push_back(std::move(s));
    }
}

void Service::consumer_thread(Consumer& c) {
    const std::string name = "sp-con:" + c.name;
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    t_in_consumer = true;
    std::vector<Rec> scratch(kConsumerScratchRecs + profiler::kSpscSinkSlackRecs);  // slack, not capacity
    std::vector<Attached> attached;

    auto attach = [&](Producer* p) {
        Attached a;
        a.producer = p;
        const CaptureContext& ctx = p->capture_context();
        const auto streams = p->streams();
        a.streams.reserve(streams.size());
        for (const ProducerStream& ps : streams) {
            a.streams.emplace_back(
                ps.ring->make_reader(), ctx.devices[ps.dev], ps.dev, reinterpret_cast<uint8_t*>(scratch.data()));
        }
        attached.push_back(std::move(a));
    };

    auto deliver = [&](Attached& a, StreamDecoder& dec, uint64_t dd) {
        const auto end = dec.end_batch();
        if (end.records == 0 && dd == 0 && end.stalls == 0) {
            return;
        }
        try {
            c.cb(RecordBatch{
                .records = std::span<const Rec>(scratch.data(), end.records),
                .dropped_delta = dd,
                .context = &a.producer->capture_context(),
                .clocks = a.producer->clocks(),
                .stall_delta = end.stalls});
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", c.name, e.what());
        }
    };

    auto pass = [&](Attached& a) {
        bool any = false;
        for (AttachedStream& s : a.streams) {
            any |= s.walker.pass(s.reader, s.dec, [&](uint64_t dd) { deliver(a, s.dec, dd); });
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
        for (size_t i = 0; i < a.streams.size(); i++) {
            const AttachedStream& s = a.streams[i];
            StreamStats stats = s.dec.stats;
            stats.dropped = s.reader.dropped();
            stats.heads.reserve(s.state.lanes.size());
            for (const profiler::SpscLane& lane : s.state.lanes) {
                stats.heads.push_back(lane.seeded != 0 ? lane.head : 0u);
            }
            c.dropped.fetch_add(stats.dropped, std::memory_order_relaxed);
            p->finish_stream(static_cast<uint32_t>(i), stats);
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
        if (c.control_pending.exchange(false, std::memory_order_acq_rel)) {
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
    // Stopped mid-capture: the readers go before the rings they read, and any control still queued is acknowledged
    // without work.
    for (const Attached& a : attached) {
        for (const AttachedStream& s : a.streams) {
            c.dropped.fetch_add(s.reader.dropped(), std::memory_order_relaxed);
        }
    }
    attached.clear();
    handle_control(false);
}

}  // namespace tt::tt_metal::streaming_profiler
