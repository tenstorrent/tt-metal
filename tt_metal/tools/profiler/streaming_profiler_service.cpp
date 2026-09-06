// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler_service.hpp"

#include <algorithm>
#include <thread>
#include <utility>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "llrt/rtoptions.hpp"
#include "tools/profiler/streaming_profiler_decode.hpp"
#include "tools/profiler/streaming_profiler_ops_csv.hpp"
#include "tools/profiler/streaming_profiler_tracy.hpp"
#include "tools/profiler/streaming_profiler_zone_csv.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

thread_local bool t_in_consumer = false;

// Everything one consumer thread holds for one producer.
struct Attached {
    Producer* producer = nullptr;
    std::vector<BroadcastRing<RingLine>::Reader> readers;
    std::vector<FrameWalker> walkers;
    std::vector<StreamDecoder<profiler::SpscRecSink>> decs;
    std::vector<profiler::SpanDecodeState> states;
    std::vector<std::vector<uint64_t>> last_ts;
    std::vector<std::vector<uint64_t>> last_rec;
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
    std::vector<Producer*> to_attach;
    std::vector<Producer*> to_detach;
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

void Service::post_control(Consumer& c, Producer* attach, Producer* detach) {
    {
        std::lock_guard<std::mutex> lk(c.control_mu);
        if (attach != nullptr) {
            c.to_attach.push_back(attach);
        }
        if (detach != nullptr) {
            c.to_detach.push_back(detach);
        }
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
        post_control(ref, p, nullptr);
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
        post_control(*c, &producer, nullptr);
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
            post_control(*c, nullptr, &producer);
        }
        wait_acks(lk);
        producers_.erase(std::find(producers_.begin(), producers_.end(), &producer));
        last = producers_.empty();
    }
    if (last) {
        log_consumer_drops();
        for (const FileSink& sink : sinks_) {
            sink.write(sink.path);
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
    std::vector<FileSink> sinks;
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
        sinks.push_back({"zone-csv", path, c, [c](const std::string& p) { c->write_csv(p); }});
    }
    if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
        auto c = std::make_shared<OpsCsvConsumer>();
        add_public("ops-csv", c);
        sinks.push_back({"ops-csv", path, c, [c](const std::string& p) { c->write_csv(p); }});
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& s : sinks) {
        sinks_.push_back(std::move(s));
    }
}

void Service::consumer_thread(Consumer& c) {
    const std::string name = "sp-con:" + c.name;
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    t_in_consumer = true;
    std::vector<RingLine> lines(kConsumerLineBatch);
    std::vector<Rec> scratch(kConsumerScratchRecs + profiler::kSpscSinkSlackRecs);  // slack, not capacity
    std::vector<experimental::streaming_profiler::Clock> clocks;
    std::vector<Attached> attached;

    auto attach = [&](Producer* p) {
        Attached a;
        a.producer = p;
        const CaptureContext& ctx = p->capture_context();
        const auto streams = p->streams();
        const size_t n = streams.size();
        a.readers.reserve(n);
        a.walkers.resize(n);
        a.decs.resize(n);
        a.states.resize(n);
        a.last_ts.resize(n);
        a.last_rec.resize(n);
        for (size_t i = 0; i < n; i++) {
            const CaptureContext::Device& dev = ctx.devices[streams[i].dev];
            a.readers.push_back(streams[i].ring->make_reader());
            a.states[i].reset(dev.lanes.size() / profiler::kSpscNRiscDecode);
            a.states[i].core_of_xy = dev.core_of_xy;
            a.last_ts[i].assign(dev.lanes.size(), 0);
            a.last_rec[i].assign(dev.lanes.size(), 0);
            a.decs[i].st = &a.states[i];
            a.decs[i].last_ts = a.last_ts[i].data();
            a.decs[i].last_rec = a.last_rec[i].data();
            a.decs[i].dev = streams[i].dev;
            a.decs[i].sink.buf = reinterpret_cast<uint8_t*>(scratch.data());
        }
        attached.push_back(std::move(a));
    };

    auto deliver = [&](Attached& a, size_t i, uint64_t dd) {
        auto& dec = a.decs[i];
        const size_t nrec = dec.sink.off / sizeof(Rec);
        const uint64_t sd = dec.stall_zones - dec.stall_mark;
        dec.stall_mark = dec.stall_zones;
        dec.sink.off = 0;
        dec.batch_seq++;
        if (nrec == 0 && dd == 0 && sd == 0) {
            return;
        }
        clocks = a.producer->clocks();
        try {
            c.cb(RecordBatch{
                .records = std::span<const Rec>(scratch.data(), nrec),
                .dropped_delta = dd,
                .context = &a.producer->capture_context(),
                .clocks = clocks,
                .stall_delta = sd});
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", c.name, e.what());
        }
    };

    auto pass = [&](Attached& a) {
        bool any = false;
        for (size_t i = 0; i < a.readers.size(); i++) {
            any |= a.walkers[i].pass(a.readers[i], lines, a.decs[i], [&](uint64_t dd) { deliver(a, i, dd); });
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
        uint64_t dropped = 0;
        for (const auto& r : a.readers) {
            dropped += r.dropped();
        }
        c.dropped.fetch_add(dropped, std::memory_order_relaxed);
        attached.erase(it);
    };

    auto handle_control = [&](bool run) {
        std::vector<Producer*> to_attach;
        std::vector<Producer*> to_detach;
        {
            std::lock_guard<std::mutex> lk(c.control_mu);
            to_attach.swap(c.to_attach);
            to_detach.swap(c.to_detach);
        }
        if (run) {
            for (Producer* p : to_attach) {
                attach(p);
            }
            for (Producer* p : to_detach) {
                detach(p);
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        pending_acks_ -= to_attach.size() + to_detach.size();
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
        uint64_t dropped = 0;
        for (const auto& r : a.readers) {
            dropped += r.dropped();
        }
        c.dropped.fetch_add(dropped, std::memory_order_relaxed);
    }
    attached.clear();
    handle_control(false);
}

}  // namespace tt::tt_metal::streaming_profiler
