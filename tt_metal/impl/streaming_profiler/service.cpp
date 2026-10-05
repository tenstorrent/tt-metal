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
#include <new>
#include <thread>
#include <utility>
#include <pthread.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/ops_csv.hpp"
#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"
#include "impl/streaming_profiler/zone_csv.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

thread_local experimental::streaming_profiler::detail::CallbackId t_consumer_id{};

}  // namespace

void set_thread_name(const std::string& name) {
    tracy::SetThreadName(name.c_str());
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", name.c_str());
    pthread_setname_np(pthread_self(), buf);
}

namespace {

// A decoded batch that waits until the receiver's clock map has final host times up to its newest record.
struct Parked {
    std::unique_ptr<uint8_t[]> buffer;
    StreamDecoder::Produced produced;
};
struct AttachedStream {
    StreamDecoder decoder;
    uint64_t cursor = 0;
    uint32_t dev = 0;
    uint32_t index = 0;
    std::deque<Parked> pending;
    int64_t final_through = std::numeric_limits<int64_t>::min();
};
struct Attached {
    Receiver* receiver = nullptr;
    std::deque<AttachedStream> streams;
    ClockMap::Reader reader;
    uint64_t dropped = 0;
    uint64_t order_regressions = 0;
};

// The most output a single batch can produce.
constexpr StreamDecoder::Capacity kFullBatch =
    StreamDecoder::out_capacity(kBatchBytes / sizeof(uint32_t), kBatchFrames);

// A batch decodes into one buffer holding its zones, events, data records and values in that order, each region sized
// for a full batch. These are the regions' offsets and the buffer's size.
constexpr size_t kEventsAt = kFullBatch.zone_bytes;
constexpr size_t kDataAt = kEventsAt + kFullBatch.event_bytes;
constexpr size_t kValuesAt = kDataAt + kFullBatch.data_bytes;
constexpr size_t kOutBufferBytes = kValuesAt + kFullBatch.value_bytes;

StreamDecoder::Out out_in(uint8_t* buffer) {
    return {buffer, buffer + kEventsAt, buffer + kDataAt, reinterpret_cast<uint64_t*>(buffer + kValuesAt)};
}

}  // namespace

struct Service::Consumer {
    std::string name;
    experimental::streaming_profiler::RecordType types{};
    BatchCallback callback;
    experimental::streaming_profiler::detail::CallbackId id;
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> control_pending{false};
    std::mutex control_mu;
    std::vector<std::pair<Receiver*, bool>> control;  // (receiver, attach), in order
};

Service::Service() : steady_(new SteadyClock) { init_site_registry(); }

SteadyClock& Service::steady() { return *steady_; }

const char* Service::plot_name(const std::string& name) {
    std::lock_guard<std::mutex> lk(plot_names_mu_);
    return plot_names_.insert(name).first->c_str();
}

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::post_control(Consumer& c, Receiver* receiver, bool attach) {
    {
        std::lock_guard<std::mutex> lk(c.control_mu);
        c.control.emplace_back(receiver, attach);
    }
    c.control_pending.store(true, std::memory_order_release);
    pending_acks_++;
    wake_consumers();
}

void Service::wait_acks(std::unique_lock<std::mutex>& lk) {
    ack_cv_.wait(lk, [&] { return pending_acks_ == 0; });
}

experimental::streaming_profiler::detail::CallbackId Service::add_consumer(
    std::string name, experimental::streaming_profiler::RecordType types, BatchCallback callback) {
    TT_FATAL(
        t_consumer_id == experimental::streaming_profiler::detail::CallbackId{},
        "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto owned = std::make_unique<Consumer>();
    owned->name = std::move(name);
    owned->types = types;
    owned->callback = std::move(callback);
    Consumer& consumer = *owned;
    std::unique_lock<std::mutex> lk(mu_);
    consumer.id = experimental::streaming_profiler::detail::CallbackId{next_id_++};
    for (Receiver* receiver : receivers_) {
        post_control(consumer, receiver, true);
    }
    consumers_.push_back(std::move(owned));
    consumer.thread = std::thread(&Service::consumer_thread, this, std::ref(consumer));
    wait_acks(lk);
    return consumer.id;
}

void Service::remove_consumer(experimental::streaming_profiler::detail::CallbackId id) {
    const bool self = id == t_consumer_id;
    TT_FATAL(
        self || t_consumer_id == experimental::streaming_profiler::detail::CallbackId{},
        "streaming profiler: a consumer callback may unregister only itself");
    std::unique_lock<std::mutex> topo(topology_mu_, std::defer_lock);
    if (!self) {
        topo.lock();
    }
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = std::ranges::find_if(consumers_, [&](const auto& consumer) { return consumer->id == id; });
        if (self) {
            (*it)->stop.store(true, std::memory_order_release);
            return;
        }
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->stop.store(true, std::memory_order_release);
    wake_consumers();
    victim->thread.join();
}

void Service::attach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    receivers_.push_back(&receiver);
    for (auto& c : consumers_) {
        post_control(*c, &receiver, true);
    }
    wait_acks(lk);
}

void Service::detach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        for (auto& c : consumers_) {
            post_control(*c, &receiver, false);
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
            builtin_callbacks_.push_back(experimental::streaming_profiler::RegisterCallback(
                [sink](const typename Sink::Batch& batch) { (*sink)(batch); }, name));
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

class Service::ConsumerLoop {
public:
    ConsumerLoop(Service& service, Consumer& consumer) :
        service_(service), consumer_(consumer), frames_buf_(std::make_unique_for_overwrite<std::byte[]>(kBatchBytes)) {}

    void run() {
        set_thread_name("sp-con:" + consumer_.name);
        IdleBackoff backoff(0);
        while (true) {
            const uint32_t seen = service_.wake_token();
            if (consumer_.control_pending.load(std::memory_order_acquire)) {
                consumer_.control_pending.store(false, std::memory_order_release);
                apply_control();
            }
            if (consumer_.stop.load(std::memory_order_acquire)) {
                break;
            }
            bool any = false;
            for (auto& attached : attached_) {
                any |= pass(*attached);
            }
            any |= drain();
            if (any) {
                backoff.reset();
                continue;
            }
            if (backoff.spin()) {
                continue;
            }
            service_.wait_wake(seen);
        }
        for (const auto& attached : attached_) {
            report(*attached);
        }
    }

private:
    void apply_control() {
        std::vector<std::pair<Receiver*, bool>> control;
        {
            std::lock_guard<std::mutex> lk(consumer_.control_mu);
            control.swap(consumer_.control);
        }
        for (const auto& [receiver, attaching] : control) {
            if (attaching) {
                attach(receiver);
            } else {
                detach(receiver);
            }
        }
        std::lock_guard<std::mutex> lk(service_.mu_);
        service_.pending_acks_ -= control.size();
        service_.ack_cv_.notify_all();
    }

    void attach(Receiver* receiver) {
        auto attached = std::make_unique<Attached>();
        attached->receiver = receiver;
        attached->reader = receiver->clock_map().reader();
        const CaptureContext& ctx = receiver->capture_context();
        const auto sources = receiver->streams();
        for (uint32_t index = 0; index < sources.size(); index++) {
            const ReceiverStream& source = sources[index];
            attached->streams.push_back(AttachedStream{
                .decoder = StreamDecoder(ctx.devices[source.dev], consumer_.types),
                .cursor = source.walked->load(std::memory_order_acquire),
                .dev = source.dev,
                .index = index});
        }
        attached_.push_back(std::move(attached));
    }

    // Drains the receiver's streams, delivers every batch and drops the receiver. A receiver detaches only after its
    // clock solver finishes, so every placement is final.
    void detach(Receiver* receiver) {
        const auto it = std::ranges::find_if(attached_, [&](const auto& entry) { return entry->receiver == receiver; });
        Attached& attached = **it;
        // Every placement is final, so each pass's batches are delivered before the next pass parks more. Otherwise a
        // consumer that is far behind would hold its whole backlog decoded at once.
        bool more = true;
        while (more) {
            more = pass(attached);
            for (AttachedStream& stream : attached.streams) {
                for (Parked& parked : stream.pending) {
                    deliver(attached, stream.dev, parked);
                }
                stream.pending.clear();
            }
        }
        report(attached);
        attached_.erase(it);
    }

    // Takes one batch from each stream and returns whether any stream had one. Taking one per stream keeps any stream
    // from getting lapped while another is being drained.
    bool pass(Attached& attached) {
        bool any = false;
        for (AttachedStream& stream : attached.streams) {
            any |= take_batch(attached, stream);
        }
        return any;
    }

    bool take_batch(Attached& attached, AttachedStream& stream) {
        const ReceiverStream& source = attached.receiver->streams()[stream.index];
        const Walked walked = walk_frames(
            source.fifo,
            stream.cursor,
            source.walked->load(std::memory_order_acquire),
            source.marks,
            std::span<std::byte>(frames_buf_.get(), kBatchBytes),
            frame_words_,
            [&] { return attached.receiver->live_head(stream.index); });
        if (walked.cursor == stream.cursor) {
            return false;
        }
        stream.cursor = walked.cursor;
        attached.dropped += walked.dropped;
        dropped_since_callback_ += walked.dropped;
        std::unique_ptr<uint8_t[]> buffer = take_buffer();
        const StreamDecoder::Produced produced = stream.decoder.decode_frames(
            stream.pending.size(),
            reinterpret_cast<const uint32_t*>(frames_buf_.get()),
            std::span<const uint32_t>(frame_words_.data(), walked.frames),
            out_in(buffer.get()));
        attached.order_regressions += produced.order_regressions;
        stream.pending.push_back(Parked{.buffer = std::move(buffer), .produced = produced});
        return true;
    }

    void report(const Attached& attached) const {
        if (attached.dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\" missed {} bytes of frames",
                consumer_.name,
                attached.dropped);
        }
        if (attached.order_regressions != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\": {} order regressions",
                consumer_.name,
                attached.order_regressions);
        }
    }

    bool drain() {
        bool any = false;
        for (auto& attached : attached_) {
            const ClockMap& map = attached->receiver->clock_map();
            for (AttachedStream& stream : attached->streams) {
                while (!stream.pending.empty()) {
                    Parked& parked = stream.pending.front();
                    const int64_t newest = parked.produced.newest_ticks;
                    if (newest > stream.final_through) {
                        if (!map.is_final(attached->reader, stream.dev, newest)) {
                            break;
                        }
                        stream.final_through = newest;
                    }
                    deliver(*attached, stream.dev, parked);
                    stream.pending.pop_front();
                    any = true;
                }
            }
        }
        return any;
    }

    void deliver(Attached& attached, uint32_t dev, Parked& parked) {
        const StreamDecoder::Out out = out_in(parked.buffer.get());
        const StreamDecoder::Produced& produced = parked.produced;
        place(attached, dev, out, produced);
        const experimental::streaming_profiler::detail::BatchData batch{
            .zones = std::launder(reinterpret_cast<const experimental::streaming_profiler::Zone*>(out.zones)),
            .zone_count = produced.zones,
            .timestamped_data =
                std::launder(reinterpret_cast<const experimental::streaming_profiler::TimestampedData*>(out.data)),
            .timestamped_data_count = produced.data,
            .events = std::launder(reinterpret_cast<const experimental::streaming_profiler::Event*>(out.events)),
            .event_count = produced.events,
            .dropped_bytes = std::exchange(dropped_since_callback_, 0),
            .stall_count = produced.stalls};
        try {
            if (!consumer_.stop.load(std::memory_order_relaxed)) {
                consumer_.callback(batch);
            }
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", consumer_.name, ex.what());
        }
        free_buffers_.push_back(std::move(parked.buffer));
    }

    // Puts the batch's records on host time through the clock map. The decoder leaves each record's clock offset in its
    // tsc_ slot, and this overwrites it with the host time.
    void place(
        Attached& attached, uint32_t dev, const StreamDecoder::Out& out, const StreamDecoder::Produced& produced) {
        const ClockMap& map = attached.receiver->clock_map();
        ClockMap::Reader& reader = attached.reader;
        using namespace profiler;
        // A memmove onto itself compiles to nothing and implicitly creates the zones, events and data-record array
        // already in these bytes.
        std::memmove(out.zones, out.zones, size_t{produced.zones} * kSpscZoneBytes);
        std::memmove(out.events, out.events, size_t{produced.events} * kSpscEventBytes);
        std::memmove(out.data, out.data, size_t{produced.data} * kSpscDataBytes);
        const auto wall_of = [](const uint64_t* record) {
            return static_cast<int64_t>(record[kSpscQwTimestamp] + record[kSpscQwTsc]);
        };
        const auto host_tsc = [&](const uint8_t* record) {
            return map.place_host(reader, dev, wall_of(reinterpret_cast<const uint64_t*>(record)));
        };
        for (uint32_t i = 0; i < produced.zones; i++) {
            uint64_t* const zone = reinterpret_cast<uint64_t*>(out.zones + size_t{i} * kSpscZoneBytes);
            const int64_t wall = wall_of(zone);
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

    // Returns an output buffer, reusing the most recently freed one, which is still in cache and already faulted in.
    std::unique_ptr<uint8_t[]> take_buffer() {
        if (free_buffers_.empty()) {
            return std::make_unique_for_overwrite<uint8_t[]>(kOutBufferBytes);
        }
        std::unique_ptr<uint8_t[]> buffer = std::move(free_buffers_.back());
        free_buffers_.pop_back();
        return buffer;
    }

    Service& service_;
    Consumer& consumer_;
    std::vector<std::unique_ptr<Attached>> attached_;
    std::array<uint32_t, kBatchFrames> frame_words_{};
    std::unique_ptr<std::byte[]> frames_buf_;
    std::vector<std::unique_ptr<uint8_t[]>> free_buffers_;
    uint64_t dropped_since_callback_ = 0;
};

void Service::consumer_thread(Consumer& consumer) {
    t_consumer_id = consumer.id;
    ConsumerLoop(*this, consumer).run();
    // A consumer that unregistered itself stays listed until here, so attach_receiver and detach_receiver may have
    // posted it controls that only this thread can acknowledge.
    std::unique_ptr<Consumer> self;
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = std::ranges::find_if(consumers_, [&](const auto& entry) { return entry.get() == &consumer; });
    if (it == consumers_.end()) {
        return;
    }
    {
        std::lock_guard<std::mutex> control_lock(consumer.control_mu);
        pending_acks_ -= consumer.control.size();
        consumer.control.clear();
    }
    ack_cv_.notify_all();
    self = std::move(*it);
    consumers_.erase(it);
    self->thread.detach();
}

}  // namespace tt::tt_metal::streaming_profiler
