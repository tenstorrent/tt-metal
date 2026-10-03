// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/service.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <pthread.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/tt_pause.hpp>

#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/ops_csv.hpp"
#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/sync/engine.hpp"
#include "impl/streaming_profiler/sync/host_sync.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"
#include "impl/streaming_profiler/zone_csv.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

thread_local bool t_in_consumer = false;
thread_local api::detail::CallbackId t_consumer_id{};

}  // namespace

void set_thread_name(const std::string& name) {
    tracy::SetThreadName(name.c_str());
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", name.c_str());
    pthread_setname_np(pthread_self(), buf);
}

struct Service::AttachedStream {
    std::optional<StreamDecoder> decoder;  // profiler streams only
    uint64_t cursor = 0;
    uint64_t dropped = 0;
    uint64_t order_regressions = 0;
    uint32_t chip = 0;
    uint32_t dev = 0;
    uint32_t index = 0;
    std::deque<Parked*> pending;
    int64_t cover_seen = std::numeric_limits<int64_t>::min();
};
struct Service::Attached {
    Receiver* receiver = nullptr;
    std::vector<std::unique_ptr<AttachedStream>> streams;
    ClockMap::Reader reader;
};
struct Service::Parked {
    Attached* attached;
    uint32_t dev;
    bool delivered;
    StreamDecoder::Out out;
    StreamDecoder::Produced produced;
    uint64_t dropped;
};

constexpr uint32_t kBatchFrames = 64;

namespace {

// One batch's worst case.
constexpr StreamDecoder::Capacity kFullBatch =
    StreamDecoder::out_capacity(size_t{kBatchFrames} * profiler::kSpscMaxFrameWords, kBatchFrames);

struct Ring {
    std::unique_ptr<uint8_t[]> buf;
    size_t cap = 0;
    size_t head = 0;
    size_t tail = 0;
    size_t wrap = 0;
    size_t live = 0;
    bool wrapped = false;
    explicit Ring(size_t bytes) : buf(std::make_unique_for_overwrite<uint8_t[]>(bytes)), cap(bytes) {}
    bool holds(const uint8_t* addr) const { return addr >= buf.get() && addr < buf.get() + cap; }
    uint8_t* reserve(size_t bytes) {
        if (live == 0) {
            head = tail = 0;
            wrapped = false;
        }
        if (!wrapped) {
            if (head + bytes <= cap) {
                return buf.get() + head;
            }
            if (bytes <= tail) {
                wrap = head;
                head = 0;
                wrapped = true;
                return buf.get();
            }
            return nullptr;
        }
        return head + bytes <= tail ? buf.get() + head : nullptr;
    }
    // An empty range isn't live, so the range before it still ends where the next one starts.
    void commit(uint8_t* start, size_t used) {
        head = static_cast<size_t>(start - buf.get()) + used;
        live += used != 0;
    }
    void release(uint8_t* start, size_t used) {
        tail = static_cast<size_t>(start - buf.get()) + used;
        live--;
        if (wrapped && tail == wrap) {
            tail = 0;
            wrapped = false;
        }
    }
};

struct Arena {
    std::vector<Ring> rings;
    explicit Arena(size_t bytes) { rings.emplace_back(bytes); }
    uint8_t* reserve(size_t bytes) {
        if (uint8_t* start = rings.back().reserve(bytes)) {
            return start;
        }
        rings.emplace_back(std::max(rings.back().cap * 2, bytes));
        return rings.back().reserve(bytes);
    }
    void commit(uint8_t* start, size_t used) { rings.back().commit(start, used); }
    void release(uint8_t* start, size_t used) {
        if (used == 0) {
            return;
        }
        for (auto ring = rings.begin(); ring != rings.end(); ++ring) {
            if (ring->holds(start)) {
                ring->release(start, used);
                if (ring->live == 0 && ring + 1 != rings.end()) {
                    rings.erase(ring);
                }
                return;
            }
        }
    }
};

// Each arena starts at two batches' worst case and grows while parked batches fill it.
struct Arenas {
    Arena zones{2 * kFullBatch.zone_bytes}, events{2 * kFullBatch.event_bytes}, data{2 * kFullBatch.data_bytes},
        values{2 * kFullBatch.value_bytes};

    StreamDecoder::Out reserve(const StreamDecoder::Capacity& cap) {
        return {
            zones.reserve(cap.zone_bytes),
            events.reserve(cap.event_bytes),
            data.reserve(cap.data_bytes),
            values.reserve(cap.value_bytes)};
    }
    void commit(const StreamDecoder::Out& out, const StreamDecoder::Produced& produced) {
        each(out, produced, [](Arena& arena, uint8_t* start, size_t used) { arena.commit(start, used); });
    }
    void release(const StreamDecoder::Out& out, const StreamDecoder::Produced& produced) {
        each(out, produced, [](Arena& arena, uint8_t* start, size_t used) { arena.release(start, used); });
    }

private:
    template <typename Fn>
    void each(const StreamDecoder::Out& out, const StreamDecoder::Produced& produced, Fn apply) {
        apply(zones, out.zones, size_t{produced.zones} * profiler::kSpscZoneBytes);
        apply(events, out.events, size_t{produced.events} * profiler::kSpscEventBytes);
        apply(data, out.data, size_t{produced.data} * profiler::kSpscDataBytes);
        apply(values, out.values, size_t{produced.values} * sizeof(uint64_t));
    }
};

}  // namespace

struct Service::Consumer {
    std::string name;
    BatchCallback callback;
    api::detail::CallbackId id{};
    std::thread thread;
    std::atomic<bool> stop{false};
    ControlQueue control;
};

Service::Service() : steady_(new SteadyClock) { init_site_registry(); }

SteadyClock& Service::steady() { return *steady_; }

const char* Service::plot_name(const std::string& name) {
    std::lock_guard lock(plot_names_mu_);
    return plot_names_.insert(name).first->c_str();
}

void Service::set_tile_clocks(ContextId context_id, uint32_t chip, TileClocks clocks) {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    tile_clocks_.emplace(std::pair{context_id, chip}, std::move(clocks));
}

const TileClocks* Service::tile_clocks(ContextId context_id, uint32_t chip) const {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    const auto it = tile_clocks_.find({context_id, chip});
    return it == tile_clocks_.end() ? nullptr : &it->second;
}

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::post_control(ControlQueue& queue, Receiver* receiver, Control action) {
    {
        std::lock_guard<std::mutex> lk(queue.mu);
        queue.items.push_back({.receiver = receiver, .action = action});
    }
    queue.pending.store(true, std::memory_order_release);
    pending_acks_++;
    wake_walkers();
}

void Service::wait_acks(std::unique_lock<std::mutex>& lk) {
    ack_cv_.wait(lk, [&] { return pending_acks_ == 0; });
}

api::detail::CallbackId Service::add_consumer(std::string name, BatchCallback callback) {
    TT_FATAL(!t_in_consumer, "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->callback = std::move(callback);
    Consumer& ref = *c;
    std::unique_lock<std::mutex> lk(mu_);
    ref.id = api::detail::CallbackId{next_id_++};
    for (Receiver* receiver : receivers_) {
        post_control(ref.control, receiver, Control::Attach);
    }
    consumers_.push_back(std::move(c));
    ref.thread = std::thread(&Service::consumer_thread, this, std::ref(ref));
    wait_acks(lk);
    return ref.id;
}

void Service::remove_consumer(api::detail::CallbackId id) {
    const bool self = id == t_consumer_id;
    TT_FATAL(self || !t_in_consumer, "streaming profiler: a consumer callback may unregister only itself");
    std::unique_lock<std::mutex> topo(topology_mu_, std::defer_lock);
    if (!self) {
        topo.lock();
    }
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = std::find_if(
            consumers_.begin(), consumers_.end(), [&](const auto& consumer) { return consumer->id == id; });
        if (self) {
            (*it)->stop.store(true, std::memory_order_release);
            return;
        }
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->stop.store(true, std::memory_order_release);
    wake_walkers();
    victim->thread.join();
}

void Service::attach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    receivers_.push_back(&receiver);
    if (!sync_thread_.joinable()) {
        sync_thread_ = std::thread(&Service::sync_thread, this);
    }
    post_control(sync_control_, &receiver, Control::Attach);
    wait_acks(lk);
    for (auto& c : consumers_) {
        post_control(c->control, &receiver, Control::Attach);
    }
    wait_acks(lk);
}

void Service::detach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        post_control(sync_control_, &receiver, Control::Detach);
        wait_acks(lk);
        for (auto& c : consumers_) {
            post_control(c->control, &receiver, Control::Detach);
        }
        wait_acks(lk);
        std::erase(receivers_, &receiver);
        last = receivers_.empty();
    }
    if (last) {
        for (const auto& write : file_sinks_) {
            write();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !receivers_.empty();
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    std::call_once(builtins_once_, [&] {
        auto add_sink = [&]<typename Sink>(const char* name, const std::shared_ptr<Sink>& sink) {
            builtin_callbacks_.push_back(
                api::RegisterCallback([sink](const typename Sink::Batch& batch) { (*sink)(batch); }, name));
        };
        auto add_file_sink = [&]<typename Sink>(const char* name, const std::shared_ptr<Sink>& sink) {
            add_sink(name, sink);
            file_sinks_.push_back([sink] { sink->write_csv(); });
        };
#if defined(TRACY_ENABLE)
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            add_sink("tracy", std::make_shared<TracyConsumer>());
        }
#endif
        if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
            add_file_sink("zone-csv", std::make_shared<ZoneCsvConsumer>(path));
        }
        if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
            add_file_sink("ops-csv", std::make_shared<OpsCsvConsumer>(path));
        }
    });
}

class Service::StreamWalker {
public:
    virtual ~StreamWalker() = default;

    void run() {
        set_thread_name(name_);
        t_in_consumer = true;
        // A sleeping reader costs every publish a futex wake.
        constexpr uint32_t kSpinsBeforeSleep = 1000;
        uint32_t empty_passes = 0;
        while (true) {
            const uint32_t seen = service_.wake_token();
            if (control_.pending.load(std::memory_order_acquire)) {
                control_.pending.store(false, std::memory_order_release);
                apply_control();
            }
            if (stop_ != nullptr && stop_->load(std::memory_order_acquire)) {
                break;
            }
            bool any = false;
            for (auto& attached : attached_) {
                any |= pass(*attached);
            }
            any |= after_pass();
            if (any) {
                empty_passes = 0;
                continue;
            }
            if (++empty_passes < kSpinsBeforeSleep) {
                ttsl::pause();
                continue;
            }
            service_.wait_wake(seen);
        }
        stopping();
    }

protected:
    StreamWalker(
        Service& service, ControlQueue& control, std::string name, const std::atomic<bool>* stop, Streams streams) :
        service_(service),
        frames_buf_(std::make_unique_for_overwrite<std::byte[]>(kFramesBytes)),
        control_(control),
        name_(std::move(name)),
        stop_(stop),
        streams_(streams) {}

    static constexpr size_t kFramesBytes = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords * sizeof(uint32_t);

    virtual bool take_batch(Attached& attached, AttachedStream& stream) = 0;
    virtual void on_detached(Attached& attached) = 0;
    virtual bool after_pass() { return false; }
    virtual void stopping() {}

    std::optional<Walked> walk(Attached& attached, AttachedStream& stream) {
        const ReceiverStream& source = attached.receiver->streams()[stream.index];
        const Walked walked = walk_frames(
            source.fifo,
            stream.cursor,
            source.walked->load(std::memory_order_acquire),
            source.marks,
            std::span<std::byte>(frames_buf_.get(), kFramesBytes),
            frame_words_,
            [&] { return attached.receiver->live_head(stream.index); });
        if (walked.cursor == stream.cursor) {
            return std::nullopt;
        }
        stream.cursor = walked.cursor;
        stream.dropped += walked.dropped;
        return walked;
    }

    Service& service_;
    std::vector<std::unique_ptr<Attached>> attached_;
    std::array<uint32_t, kBatchFrames> frame_words_{};
    std::unique_ptr<std::byte[]> frames_buf_;

private:
    std::vector<ControlItem> take_control() {
        std::vector<ControlItem> items;
        std::lock_guard<std::mutex> lk(control_.mu);
        items.swap(control_.items);
        return items;
    }
    void ack_control(size_t count) {
        std::lock_guard<std::mutex> lk(service_.mu_);
        service_.pending_acks_ -= count;
        service_.ack_cv_.notify_all();
    }
    void apply_control() {
        const std::vector<ControlItem> items = take_control();
        for (const ControlItem& item : items) {
            item.action == Control::Attach ? attach(item.receiver) : detach(item.receiver);
        }
        ack_control(items.size());
    }

    void attach(Receiver* receiver) {
        auto attached = std::make_unique<Attached>();
        attached->receiver = receiver;
        attached->reader = receiver->clock_map().reader();
        const CaptureContext& ctx = receiver->capture_context();
        const auto sources = receiver->streams();
        for (uint32_t index = 0; index < sources.size(); index++) {
            const ReceiverStream& source = sources[index];
            if (source.sync != (streams_ == Streams::Sync)) {
                continue;
            }
            auto stream = std::make_unique<AttachedStream>();
            stream->index = index;
            stream->cursor = source.walked->load(std::memory_order_acquire);
            const CaptureContext::Device& dev = ctx.devices[source.dev];
            stream->chip = dev.chip_id;
            stream->dev = source.dev;
            if (streams_ == Streams::Profiler) {
                stream->decoder.emplace(dev);
            }
            attached->streams.push_back(std::move(stream));
        }
        attached_.push_back(std::move(attached));
    }

    void detach(Receiver* receiver) {
        auto it = std::find_if(
            attached_.begin(), attached_.end(), [&](const auto& entry) { return entry->receiver == receiver; });
        Attached& attached = **it;
        while (pass(attached)) {
        }
        on_detached(attached);
        attached_.erase(it);
    }

    // One batch per stream per pass, so no stream laps while another drains.
    bool pass(Attached& attached) {
        bool any = false;
        for (const auto& stream : attached.streams) {
            any |= take_batch(attached, *stream);
        }
        return any;
    }

    ControlQueue& control_;
    const std::string name_;
    const std::atomic<bool>* stop_;
    const Streams streams_;
};

class Service::SyncLoop : public StreamWalker {
public:
    explicit SyncLoop(Service& service) :
        StreamWalker(service, service.sync_control_, "sp-sync", nullptr, Streams::Sync) {}

private:
    bool take_batch(Attached& attached, AttachedStream& stream) override {
        const std::optional<Walked> walked = walk(attached, stream);
        if (!walked) {
            return false;
        }
        const uint32_t dev = stream.dev;
        const CoreTable& core_of_xy = attached.receiver->capture_context().devices[dev].core_of_xy;
        const std::byte* frame_bytes = frames_buf_.get();
        for (uint32_t i = 0; i < walked->frames; i++) {
            const uint32_t* frame = reinterpret_cast<const uint32_t*>(frame_bytes);
            const uint32_t record_count = frame[kernel_profiler::SPSC_PREFIX_SYNC_RECORD_COUNT];
            const uint32_t core = core_of_xy.find(frame[kernel_profiler::SPSC_PREFIX_XY]);
            TT_FATAL(
                kernel_profiler::SPSC_SPAN_PREFIX_WORDS + record_count * kernel_profiler::kSyncRecordWords <=
                    frame_words_[i],
                "streaming profiler: a sync frame of chip {} from core {:#x} holds {} records in {} words",
                stream.chip,
                frame[kernel_profiler::SPSC_PREFIX_XY],
                record_count,
                frame_words_[i]);
            const auto* records =
                reinterpret_cast<const kernel_profiler::SyncRecord*>(frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS);
            for (uint32_t r = 0; r < record_count; r++) {
                attached.receiver->sync().on_record(dev, core, records[r]);
            }
            frame_bytes += size_t{frame_words_[i]} * sizeof(uint32_t);
        }
        if (attached.receiver->sync().on_batch_end()) {
            service_.wake_walkers();
        }
        return true;
    }

    bool after_pass() override {
        for (const auto& attached : attached_) {
            attached->receiver->host_sync().burst_if_due(attached->receiver->clock_map());
        }
        return false;
    }

    void on_detached(Attached& attached) override {
        attached.receiver->host_sync().finish(attached.receiver->clock_map());
        attached.receiver->sync().on_capture_end();
        for (const auto& stream : attached.streams) {
            if (stream->dropped != 0) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] the sync engine missed {} bytes of chip {}'s sync stream",
                    stream->dropped,
                    stream->chip);
            }
        }
    }
};

void Service::sync_thread() { SyncLoop(*this).run(); }

class Service::ConsumerLoop : public StreamWalker {
public:
    ConsumerLoop(Service& service, Consumer& consumer) :
        StreamWalker(service, consumer.control, "sp-con:" + consumer.name, &consumer.stop, Streams::Profiler),
        consumer_(consumer) {}

private:
    bool take_batch(Attached& attached, AttachedStream& stream) override {
        const std::optional<Walked> walked = walk(attached, stream);
        if (!walked) {
            return false;
        }
        size_t words = 0;
        for (uint32_t i = 0; i < walked->frames; i++) {
            words += frame_words_[i];
        }
        const StreamDecoder::Out out = arenas_.reserve(StreamDecoder::out_capacity(words, walked->frames));
        const StreamDecoder::Produced produced = stream.decoder->decode_frames(
            reinterpret_cast<const uint32_t*>(frames_buf_.get()),
            std::span<const uint32_t>(frame_words_.data(), walked->frames),
            out);
        arenas_.commit(out, produced);
        stream.order_regressions += produced.order_regressions;
        parked_.push_back(Parked{
            .attached = &attached,
            .dev = stream.dev,
            .delivered = false,
            .out = out,
            .produced = produced,
            .dropped = walked->dropped});
        stream.pending.push_back(&parked_.back());
        return true;
    }

    // The sync engine detached first, so the covers are final.
    void on_detached(Attached& attached) override {
        for (auto& stream : attached.streams) {
            for (Parked* parked : stream->pending) {
                deliver(*parked);
            }
            stream->pending.clear();
        }
        release_delivered();
        report(attached);
    }

    bool after_pass() override { return drain(); }

    void stopping() override {
        for (const auto& attached : attached_) {
            report(*attached);
        }
    }

    void report(const Attached& attached) const {
        uint64_t dropped = 0, order_regressions = 0;
        for (const auto& stream : attached.streams) {
            order_regressions += stream->order_regressions;
            dropped += stream->dropped;
        }
        if (dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\" missed {} bytes of frames",
                consumer_.name,
                dropped);
        }
        if (order_regressions != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\": {} order regressions",
                consumer_.name,
                order_regressions);
        }
    }

    bool drain() {
        bool any = false;
        for (auto& attached : attached_) {
            const ClockMap& map = attached->receiver->clock_map();
            for (auto& owned : attached->streams) {
                AttachedStream& stream = *owned;
                while (!stream.pending.empty()) {
                    Parked& parked = *stream.pending.front();
                    if (parked.produced.newest_ticks > stream.cover_seen) {
                        stream.cover_seen = map.cover_ticks(stream.dev);
                        if (parked.produced.newest_ticks > stream.cover_seen) {
                            break;
                        }
                    }
                    deliver(parked);
                    stream.pending.pop_front();
                    stream.decoder->commit();
                    any = true;
                }
            }
        }
        release_delivered();
        return any;
    }

    void deliver(Parked& parked) {
        place(parked);
        const StreamDecoder::Out& out = parked.out;
        const StreamDecoder::Produced& produced = parked.produced;
        const api::detail::BatchData batch{
            .zones = std::launder(reinterpret_cast<const api::Zone*>(out.zones)),
            .zone_count = produced.zones,
            .timestamped_data = std::launder(reinterpret_cast<const api::TimestampedData*>(out.data)),
            .timestamped_data_count = produced.data,
            .events = std::launder(reinterpret_cast<const api::Event*>(out.events)),
            .event_count = produced.events,
            .dropped_bytes = parked.dropped,
            .stall_count = produced.stalls};
        try {
            if (!consumer_.stop.load(std::memory_order_relaxed)) {
                consumer_.callback(batch);
            }
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", consumer_.name, ex.what());
        }
        parked.delivered = true;
    }

    // The decoder leaves each record's tile offset in its tsc_ slot, and placement overwrites it with the host time.
    void place(Parked& parked) {
        const ClockMap& map = parked.attached->receiver->clock_map();
        ClockMap::Reader& reader = parked.attached->reader;
        const uint32_t dev = parked.dev;
        const StreamDecoder::Out& out = parked.out;
        const StreamDecoder::Produced& produced = parked.produced;
        using namespace profiler;
        // memmove onto itself compiles to nothing and is how C++20 starts objects' lifetimes over bytes already in
        // place: the zones and events here, and the array the data records are constructed into (their values too, in
        // construct_timestamped_data).
        std::memmove(out.zones, out.zones, size_t{produced.zones} * kSpscZoneBytes);
        std::memmove(out.events, out.events, size_t{produced.events} * kSpscEventBytes);
        std::memmove(out.data, out.data, size_t{produced.data} * kSpscDataBytes);
        const auto host_tsc = [&](const uint8_t* record) {
            const uint64_t* const point = reinterpret_cast<const uint64_t*>(record);
            return map.place_host(reader, dev, static_cast<int64_t>(point[kSpscQwTimestamp] + point[kSpscQwTsc]));
        };
        for (uint32_t i = 0; i < produced.zones; i++) {
            uint64_t* const zone = reinterpret_cast<uint64_t*>(out.zones + size_t{i} * kSpscZoneBytes);
            const int64_t wall = static_cast<int64_t>(zone[kSpscQwTimestamp] + zone[kSpscQwTsc]);
            const int64_t start = map.place_host(reader, dev, wall);
            const int64_t end = map.place_host(reader, dev, wall + static_cast<int64_t>(zone[kSpscQwDuration]));
            zone[kSpscQwTsc] = static_cast<uint64_t>(start);
            zone[kSpscQwEndTsc] = static_cast<uint64_t>(end);
        }
        for (uint32_t i = 0; i < produced.events; i++) {
            uint8_t* const event = out.events + size_t{i} * kSpscEventBytes;
            reinterpret_cast<uint64_t*>(event)[kSpscQwTsc] = static_cast<uint64_t>(host_tsc(event));
        }
        for (uint32_t i = 0; i < produced.data; i++) {
            uint8_t* const record = out.data + size_t{i} * kSpscDataBytes;
            construct_timestamped_data(record, host_tsc(record));
        }
    }

    void release_delivered() {
        while (!parked_.empty() && parked_.front().delivered) {
            arenas_.release(parked_.front().out, parked_.front().produced);
            parked_.pop_front();
        }
    }

    Consumer& consumer_;
    Arenas arenas_;
    std::deque<Parked> parked_;
};

void Service::consumer_thread(Consumer& consumer) {
    t_consumer_id = consumer.id;
    ConsumerLoop(*this, consumer).run();
    // A self-unregistered consumer stays listed until here, so detach still waits on its acks.
    std::unique_ptr<Consumer> self;
    std::lock_guard<std::mutex> lk(mu_);
    const auto it =
        std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& entry) { return entry.get() == &consumer; });
    if (it == consumers_.end()) {
        return;
    }
    {
        std::lock_guard<std::mutex> control_lock(consumer.control.mu);
        pending_acks_ -= consumer.control.items.size();
        consumer.control.items.clear();
    }
    ack_cv_.notify_all();
    self = std::move(*it);
    consumers_.erase(it);
    self->thread.detach();
}

}  // namespace tt::tt_metal::streaming_profiler
