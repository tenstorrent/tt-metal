// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <memory>
#include <thread>
#include <utility>

#include <tracy/Tracy.hpp>
#include <x86intrin.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "llrt/rtoptions.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_ops_csv.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"
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
    uint32_t chip = 0;
    std::deque<Parked*> pending;                               // this stream's undelivered batches, in decode order
    int64_t cover_seen = std::numeric_limits<int64_t>::min();  // the chip's cover as last read for this stream
};
struct Service::Attached {
    Producer* producer = nullptr;
    uint64_t capture = 0;
    std::vector<std::unique_ptr<AttachedStream>> streams;  // stable: dec.st points into the stream
};
// A decoded batch, waiting in the arenas until the sync covers its newest record (or at once, for a consumer that
// does not wait). Released in decode order once delivered; `a` is null once its producer detached.
struct Service::Parked {
    Attached* a;
    uint32_t stream;
    bool delivered;
    uint8_t* zones;
    uint8_t* events;
    uint8_t* data;
    StreamDecoder::Produced n;
    uint64_t dropped;
    uint64_t stalls;
    int64_t parked_at_ns;
};

// Frames decoded per delivery; bounds the consumer's scratch and the callback's batch.
constexpr uint32_t kBatchFrames = 64;

namespace {

// A consumer's decoded records live in three arenas (zones, events, timestamped data). A batch keeps its ranges until
// it is delivered and batches are released in decode order, so each arena is a ring whose free space runs from the
// write position to the oldest live range; the stretch a wrap leaves dead at the end is skipped once the tail
// reaches it. Pages are touched only as far as batches ever reached. Sized for the records a waiting consumer holds
// during the sync's cover latency (a few ms) at the highest ingest rates (records are ~3x their frame bytes), and
// well past one batch's worst case (64 full frames: 4.1 MB of records per kind, 4.8 MB of data).
constexpr size_t kZonesArenaBytes = size_t{128} << 20;
constexpr size_t kEventsArenaBytes = size_t{64} << 20;
constexpr size_t kDataArenaBytes = size_t{64} << 20;
// A batch parked this long is delivered regardless: its chip's cover stopped advancing (a link that never solved, a
// starved sync consumer), and records should not wait on it forever.
constexpr int64_t kMaxParkNs = 1'000'000'000;

struct Arena {
    std::unique_ptr<uint8_t[]> buf;
    size_t cap = 0;
    size_t head = 0;  // next write
    size_t tail = 0;  // oldest live byte
    size_t wrap = 0;  // end of the live stretch before the jump to 0, while wrapped
    size_t live = 0;  // ranges committed and not yet released
    bool wrapped = false;
    explicit Arena(size_t bytes) : buf(std::make_unique_for_overwrite<uint8_t[]>(bytes)), cap(bytes) {}
    // `n` contiguous bytes at the write position, or nullptr when the live ranges leave no such room.
    uint8_t* reserve(size_t n) {
        if (live == 0) {
            head = tail = 0;
            wrapped = false;
        }
        if (!wrapped) {
            if (head + n <= cap) {
                return buf.get() + head;
            }
            if (n <= tail) {
                wrap = head;
                head = 0;
                wrapped = true;
                return buf.get();
            }
            return nullptr;
        }
        return head + n <= tail ? buf.get() + head : nullptr;
    }
    // An empty range owns nothing: it is not live, so the range before it still ends where the next one starts.
    void commit(uint8_t* p, size_t used) {
        head = static_cast<size_t>(p - buf.get()) + used;
        live += used != 0;
    }
    // Frees the oldest live range.
    void release(uint8_t* p, size_t used) {
        if (used == 0) {
            return;
        }
        tail = static_cast<size_t>(p - buf.get()) + used;
        live--;
        if (wrapped && tail == wrap) {
            tail = 0;
            wrapped = false;
        }
    }
};

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

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
    std::atomic<uint64_t> unplaced{0};  // batches delivered before the sync covered them
    // Consumer-thread only: what a capture cost this consumer, in TSC cycles, reported when its producer detaches.
    uint64_t batches = 0, records = 0, cb_cycles = 0, decode_cycles = 0;
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
    for (const bool waits : {false, true}) {
        for (auto& c : consumers_) {
            if (c->hooks.waits_for_sync == waits) {
                post_control(*c, &producer, true);
            }
        }
        wait_acks(lk);
    }
}

void Service::detach_producer(Producer& producer) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        auto it = std::find(producers_.begin(), producers_.end(), &producer);
        TT_FATAL(it != producers_.end(), "streaming profiler: detaching a producer that is not attached");
        for (const bool waits : {false, true}) {
            for (auto& c : consumers_) {
                if (c->hooks.waits_for_sync == waits) {
                    post_control(*c, &producer, false);
                }
            }
            wait_acks(lk);
        }
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
    if (const uint64_t unplaced = c.unplaced.load(std::memory_order_relaxed); unplaced != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] consumer \"{}\" received {} batches before the d2d sync covered their records",
            c.name,
            unplaced);
    }
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    std::call_once(builtins_once_, [&] {
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            tracy_ = std::make_unique<TracySink>(*this);
        }
        {
            // Device<->device sync: consumes only the PP_CLOCK samples (idle-eth trackers and link stamps) and
            // publishes the corrections the other consumers wait on. Its batch callback is a no-op; the decode pass
            // is what routes the samples to it.
            auto c = std::make_shared<D2dSyncConsumer>();
            add_consumer(
                "d2d-sync",
                [](const api::Batch<api::RecordType::All>&, uint64_t) {},
                ConsumerHooks{
                    .on_attach = [c](const CaptureContext& ctx) { c->on_attach(ctx); },
                    .clock_sink = [c](const ClockSample& cs) { c->on_clock(cs); },
                    .on_capture_end = [c](const CaptureContext& ctx) { c->on_capture_end(ctx); },
                    .waits_for_sync = false});
        }
        auto add_public = [&]<typename C>(const char* name, const std::shared_ptr<C>& c) {
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
    const uint64_t tsc_epoch = __rdtsc();
    const int64_t tsc_epoch_ns = now_ns();
    std::vector<std::unique_ptr<Attached>> attached;
    Arena zones_arena(kZonesArenaBytes), events_arena(kEventsArenaBytes), data_arena(kDataArenaBytes);
    std::deque<Parked> parked;  // every undelivered or unreleased batch, in decode order
    uint64_t covers_seen = ~0ull;
    constexpr size_t kFramesBytes = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords * 4;
    std::array<uint32_t, kBatchFrames> frame_words;
    auto frames_buf = std::make_unique_for_overwrite<std::byte[]>(kFramesBytes);
    auto attach = [&](Producer* p) {
        auto a = std::make_unique<Attached>();
        a->producer = p;
        const CaptureContext& ctx = p->capture_context();
        for (const ProducerStream& ps : p->streams()) {
            auto s = std::make_unique<AttachedStream>();
            s->cursor = ps.walked->load(std::memory_order_acquire);
            const CaptureContext::Device& dev = ctx.devices[ps.dev];
            s->chip = dev.chip_id;
            s->state.reset(dev.lanes.size() / profiler::kSpscNRiscDecode);
            s->state.core_of_xy.load(dev.core_xy);
            s->lanes.reserve(dev.lanes.size());
            // The trailing eth cores read a different wall-clock counter than the workers, so their records need
            // the eth anchor or they land hours off on the timeline.
            const size_t n_eth_lanes = static_cast<size_t>(dev.n_eth_cores) * profiler::kSpscNRiscDecode;
            const size_t worker_lanes =
                dev.lanes.size() >= n_eth_lanes ? dev.lanes.size() - n_eth_lanes : dev.lanes.size();
            for (size_t li = 0; li < dev.lanes.size(); li++) {
                const DeviceClock& clk =
                    (li >= worker_lanes && dev.eth_clock.frequency_ghz > 0.0) ? dev.eth_clock : p->clock(ps.dev);
                s->lanes.push_back(record_consts(dev.lanes[li], clk));
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
            a->streams.push_back(std::move(s));
        }
        if (c.hooks.on_attach) {
            c.hooks.on_attach(ctx);
        }
        a->capture = ++c.captures;
        attached.push_back(std::move(a));
    };
    auto deliver = [&](Parked& pk) {
        api::Batch<api::RecordType::All> b;
        b.zones_ = std::span<const api::Zone>(reinterpret_cast<const api::Zone*>(pk.zones), pk.n.zones);
        b.events_ = std::span<const api::Event>(reinterpret_cast<const api::Event*>(pk.events), pk.n.events);
        b.timestamped_data_ = std::ranges::subrange(
            api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(pk.data)),
            api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(pk.data) + pk.n.data_bytes));
        b.dropped_ = pk.dropped;
        b.stall_count_ = pk.stalls;
        const uint64_t t0 = __rdtsc();
        try {
            c.cb(b, pk.a->capture);
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", c.name, ex.what());
        }
        c.cb_cycles += __rdtsc() - t0;
        c.batches++;
        c.records += size_t{pk.n.zones} + pk.n.events;
        pk.delivered = true;
    };
    auto release_delivered = [&] {
        while (!parked.empty() && parked.front().delivered) {
            const Parked& pk = parked.front();
            zones_arena.release(pk.zones, size_t{pk.n.zones} * profiler::kSpscRecBytes);
            events_arena.release(pk.events, size_t{pk.n.events} * profiler::kSpscRecBytes);
            data_arena.release(pk.data, pk.n.data_bytes);
            parked.pop_front();
        }
    };
    // Delivers each stream's batches in decode order while the sync covers them, then frees behind the delivered
    // ones. A chip's cover is re-read only once some cover has moved since the last drain, so a blocked stream costs
    // one compare per pass. A batch parked longer than kMaxParkNs goes out regardless and is counted.
    auto drain = [&] {
        bool any = false;
        const uint64_t gen = SyncCorrections::cover_generation();
        const bool moved = gen != covers_seen;
        covers_seen = gen;
        int64_t now = 0;
        for (auto& a : attached) {
            for (auto& sp : a->streams) {
                AttachedStream& s = *sp;
                bool reread = !moved;
                while (!s.pending.empty()) {
                    Parked& pk = *s.pending.front();
                    if (c.hooks.waits_for_sync && pk.n.newest_ns > s.cover_seen) {
                        if (!reread) {
                            s.cover_seen = SyncCorrections::cover_ns(s.chip);
                            reread = true;
                        }
                        if (pk.n.newest_ns > s.cover_seen) {
                            if (now == 0) {
                                now = now_ns();
                            }
                            if (now - pk.parked_at_ns < kMaxParkNs) {
                                break;
                            }
                            c.unplaced.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    deliver(pk);
                    s.pending.pop_front();
                    any = true;
                }
            }
        }
        release_delivered();
        return any;
    };
    // Room for a batch of `words` frame words in every arena; the oldest waiting batch goes out uncovered when the
    // arenas are full.
    auto reserve = [&](size_t words, uint32_t frames, uint8_t*& z, uint8_t*& e, uint8_t*& d) {
        const size_t rec_bytes = (words / 2 + profiler::kSpscSinkSlackRecs) * profiler::kSpscRecBytes;
        const size_t data_bytes = rec_bytes + words * 4 + size_t{32} * frames;
        for (;;) {
            z = zones_arena.reserve(rec_bytes);
            e = events_arena.reserve(rec_bytes);
            d = data_arena.reserve(data_bytes);
            if (z != nullptr && e != nullptr && d != nullptr) {
                return;
            }
            TT_FATAL(
                !parked.empty(),
                "streaming profiler: a batch of {} frame words does not fit a consumer's arenas",
                words);
            for (Parked& pk : parked) {
                if (!pk.delivered) {
                    deliver(pk);
                    pk.a->streams[pk.stream]->pending.pop_front();
                    c.unplaced.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
            release_delivered();
        }
    };
    // One batch of up to kBatchFrames frames from the stream, read in place up to the ingest's walk position and
    // decoded into the arenas. False when the stream had nothing new.
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
        const uint64_t t0 = __rdtsc();
        size_t words = 0;
        for (uint32_t i = 0; i < w.frames; i++) {
            words += frame_words[i];
        }
        uint8_t *z = nullptr, *e = nullptr, *d = nullptr;
        reserve(words, w.frames, z, e, d);
        StreamDecoder::Produced n{0, 0, 0, std::numeric_limits<int64_t>::min()};
        const std::byte* p = frames_buf.get();
        for (uint32_t i = 0; i < w.frames; i++) {
            const uint32_t fw = frame_words[i];
            const auto out = s.dec.decode_frame(
                reinterpret_cast<const uint32_t*>(p),
                fw,
                {z + size_t{n.zones} * profiler::kSpscRecBytes,
                 e + size_t{n.events} * profiler::kSpscRecBytes,
                 d + n.data_bytes});
            n.zones += out.zones;
            n.events += out.events;
            n.data_bytes += out.data_bytes;
            n.newest_ns = std::max(n.newest_ns, out.newest_ns);
            p += size_t{fw} * 4;
        }
        s.dec.commit();
        zones_arena.commit(z, size_t{n.zones} * profiler::kSpscRecBytes);
        events_arena.commit(e, size_t{n.events} * profiler::kSpscRecBytes);
        data_arena.commit(d, n.data_bytes);
        parked.push_back(Parked{
            .a = &a,
            .stream = stream_index,
            .delivered = false,
            .zones = z,
            .events = e,
            .data = d,
            .n = n,
            .dropped = w.dropped,
            .stalls = s.dec.stall_zones - s.stalls_reported,
            .parked_at_ns = c.hooks.waits_for_sync ? now_ns() : 0});
        s.pending.push_back(&parked.back());
        s.stalls_reported = s.dec.stall_zones;
        c.decode_cycles += __rdtsc() - t0;
        return true;
    };
    // One batch per stream per pass, so no stream's ring laps while an earlier one is drained to empty.
    auto pass = [&](Attached& a) {
        bool any = false;
        for (uint32_t i = 0; i < a.streams.size(); i++) {
            any |= read(a, *a.streams[i], i);
        }
        return any;
    };
    auto detach = [&](Producer* p) {
        auto it = std::find_if(attached.begin(), attached.end(), [&](const auto& a) { return a->producer == p; });
        if (it == attached.end()) {
            return;
        }
        Attached& a = **it;
        while (pass(a)) {
        }
        // The producer's waiting batches go out now: the sync's consumer detached first, so their covers are final.
        for (auto& sp : a.streams) {
            for (Parked* pk : sp->pending) {
                deliver(*pk);
            }
            sp->pending.clear();
        }
        release_delivered();
        for (Parked& pk : parked) {
            if (pk.a == &a) {
                pk.a = nullptr;
            }
        }
        if (c.hooks.on_capture_end) {
            c.hooks.on_capture_end(p->capture_context());
        }
        if (c.records != 0) {
            const double ns_per_cycle =
                static_cast<double>(now_ns() - tsc_epoch_ns) / static_cast<double>(__rdtsc() - tsc_epoch);
            log_info(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\": {} batches, {} zones and events; per record {:.2f} ns decoding "
                "and "
                "parking, {:.2f} ns in the callback",
                c.name,
                c.batches,
                c.records,
                static_cast<double>(c.decode_cycles) * ns_per_cycle / static_cast<double>(c.records),
                static_cast<double>(c.cb_cycles) * ns_per_cycle / static_cast<double>(c.records));
            c.batches = c.records = c.cb_cycles = c.decode_cycles = 0;
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
        for (auto& a : attached) {
            any |= pass(*a);
        }
        any |= drain();
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
    for (const auto& a : attached) {
        for (const auto& s : a->streams) {
            c.dropped.fetch_add(s->dropped, std::memory_order_relaxed);
        }
    }
    attached.clear();
    handle_control(false);
}

}  // namespace tt::tt_metal::streaming_profiler
