// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_h2h_socket.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "hostdevcommon/erisc_bridge_layout.h"
#include "hostdevcommon/erisc_host_bridge.h"
#include "tt_metal/distributed/host_rdma_window.hpp"

namespace tt::tt_metal::experimental {

using namespace tt::tt_fabric;

// Named, not anonymous: Impl has external linkage and holds Geom, so internal
// linkage here is -Wsubobject-linkage under gcc -Werror. Not `detail` -- unity builds collide.
namespace erisc_h2h_detail {

// Geometry both ranks must agree on, or a put lands somewhere the peer does not read.
struct Geom {
    uint32_t arenas = 0;
    uint32_t chans_per_link = 1;
    uint32_t ring = 0;
    uint32_t capacity = 0;

    uint32_t slot() const { return bridge_slot_size(capacity); }
    uint64_t region() const { return bridge_region_bytes(arenas, ring, capacity); }
    // One fingerprint, agreed by allreduce, hashed as delimited text: shifted-and-XORed fields
    // collide once a value outgrows its slot. chans_per_link is in it -- arena indices depend on it.
    uint64_t fingerprint() const {
        return std::hash<std::string>{}(
            "arenas " + std::to_string(arenas) + ", chans_per_link " + std::to_string(chans_per_link) + ", ring " +
            std::to_string(ring) + ", capacity " + std::to_string(capacity));
    }
};

}  // namespace erisc_h2h_detail

namespace {

uint64_t slot_off(const erisc_h2h_detail::Geom& g, uint32_t arena, BridgeArena dir, uint32_t slot) {
    return bridge_segment_offset(arena, dir, g.arenas, g.ring, g.capacity) +
           bridge_slot_offset_in_segment(slot, g.capacity);
}
uint64_t desc_off(const erisc_h2h_detail::Geom& g, uint32_t arena, BridgeArena dir, uint32_t slot) {
    return bridge_desc_offset(arena, dir, slot, g.ring);
}
uint64_t credit_off(const erisc_h2h_detail::Geom& g, uint32_t arena) { return bridge_credit_offset(arena, g.ring); }

// The wire's "far slot unknown": slot_idx is a fixed uint16_t, so absence must travel as a value.
// A real index is below the receiver channel's slot count, far under 0xFFFF.
constexpr uint16_t kWireUnknownFarSlot = 0xFFFF;

// The descriptor that travels in the control block, where the RX host polls. release_cyc and
// elapsed are CARRIED, not regenerated -- this hop is a relay, and zeroing them loses the stamp.
void make_desc(
    BridgeDescriptor* d,
    const erisc_h2h_detail::Geom& g,
    uint32_t seq,
    uint16_t far_slot,
    uint32_t release_cyc,
    uint64_t elapsed,
    uint32_t length) {
    // The CALLER'S length: a slot rounds up to kBridgeSlotAlign, so g.capacity is almost never
    // what the sender filled, and reporting it fails every frame the RX checks. 0 means unsaid.
    d->length = length != 0 ? std::min(length, g.capacity) : g.capacity;
    d->ordering_cntr = seq;
    // THE FAR RING'S SLOT, carried from the sending router -- not this hop's ring slot, which
    // the receiver already knows from where it found the descriptor and never reads from here.
    d->slot_idx = far_slot;
    d->reserved0 = 0;
    d->release_cyc = release_cyc;
    d->elapsed = elapsed;
    d->guard = bridge_guard(kBridgeVersion, seq);  // written last by the second put
}

}  // namespace

struct EriscH2HSocket::Impl {
    Config cfg{};
    erisc_h2h_detail::Geom g{};
    std::unique_ptr<RdmaWindow> win;

    // Per arena, because fabric guarantees order only WITHIN a channel and an arena IS one.
    std::vector<uint64_t> posted;     // frames put, per arena
    std::vector<uint32_t> delivered;  // the unbroken run the RX has seen
    // Last ordering_cntr flagged as disorder per (arena, slot), so bad_order counts
    // descriptors rather than scan passes. See the assign() in create().
    std::vector<uint32_t> flagged;
    // Put time per (arena, slot), so put_to_credit_ns() has something to return. One clock stamps
    // both ends, so the round trip is offset-free even though the two hosts are not synchronised.
    std::vector<std::chrono::steady_clock::time_point> put_at;
    std::vector<uint64_t> credited;        // what the peer has credited us, per arena
    std::vector<uint64_t> consumed_total;  // frames our consumer released: the credit we publish
    std::vector<uint64_t> published;

    std::vector<uint32_t> rx_slot;  // next slot to look at, per arena: the RX walks in order
    std::deque<BridgeSendTask> queue;
    std::vector<RdmaWindow::Op> ops;   // one per put of the batch in flight
    std::vector<uint32_t> sel;         // queue indices in this batch, ascending
    std::vector<uint32_t> batch_slot;  // Rx slot of sel[k]
    std::vector<uint8_t> closed;       // arena starved this batch: its later frames wait, order holds
    PassStats stats{};
    Series series{};
    std::vector<uint64_t> rtt_ns;
    std::string err;
    bool broken = false;

    void fail(const std::string& w) {
        if (err.empty()) {
            err = w;
        }
        broken = true;
    }
    uint8_t* base() const { return cfg.region_base; }
    volatile uint64_t* credit_word(uint32_t arena) const {
        return reinterpret_cast<volatile uint64_t*>(cfg.region_base + credit_off(g, arena));
    }
};

EriscH2HSocket::EriscH2HSocket() : impl_(std::make_unique<Impl>()) {}
EriscH2HSocket::~EriscH2HSocket() = default;

std::unique_ptr<EriscH2HSocket> EriscH2HSocket::create(const Config& cfg, std::string& err) {
    err.clear();
    // A LOCAL failure must not return before the collectives below: the ranks that passed would
    // wait in them for good. Every rank agrees on the verdict first.
    std::string agree_err;
    const bool args_ok = cfg.region_base != nullptr && cfg.arenas != 0 && cfg.page_bytes != 0 && cfg.ring_pages != 0;
    if (!RdmaWindow::agree(args_ok, agree_err)) {
        err = args_ok ? agree_err
                      : "EriscH2HSocket::create: region_base, arenas, page_bytes and ring_pages are all required";
        return nullptr;
    }
    std::unique_ptr<EriscH2HSocket> s(new EriscH2HSocket());
    auto& im = *s->impl_;
    im.cfg = cfg;
    im.g.arenas = cfg.arenas;
    im.g.chans_per_link = cfg.chans_per_link != 0 ? cfg.chans_per_link : 1;
    im.g.ring = cfg.ring_pages;
    // page_bytes is the SLOT; capacity is what fits before the descriptor.
    im.g.capacity = cfg.page_bytes > kBridgeDescriptorBytes ? cfg.page_bytes - kBridgeDescriptorBytes : cfg.page_bytes;

    // AGREED BEFORE THE WINDOW EXISTS. RdmaWindow::create is collective, so a rank that laid
    // out differently would put into the wrong place with no error anywhere.
    if (!RdmaWindow::agree_value(im.g.fingerprint(), agree_err)) {
        err = "EriscH2HSocket::create: geometry disagrees between ranks: " + agree_err;
        return nullptr;
    }
    im.win = RdmaWindow::create(cfg.region_base, cfg.region_bytes, cfg.host_rank, cfg.host_count, err);
    // A rank that failed locally must still reach agree(), or the ranks that succeeded hang.
    if (!RdmaWindow::agree(im.win != nullptr, agree_err)) {
        if (err.empty()) {
            err = agree_err;
        }
        return nullptr;
    }

    im.posted.assign(cfg.arenas, 0);
    im.delivered.assign(cfg.arenas, 0);
    im.rx_slot.assign(cfg.arenas, 0);
    // Edge triggered: the scan below re-examines every armed slot each pass, so a level-triggered
    // counter reports one stuck descriptor millions of times. 0 is free -- ordering_cntr is 1-based.
    im.flagged.assign(static_cast<std::size_t>(cfg.arenas) * im.g.ring, 0);
    im.put_at.assign(static_cast<std::size_t>(cfg.arenas) * im.g.ring, std::chrono::steady_clock::time_point{});
    im.rtt_ns.reserve(cfg.timing_samples);
    im.credited.assign(cfg.arenas, 0);
    im.consumed_total.assign(cfg.arenas, 0);
    im.published.assign(cfg.arenas, 0);
    return s;
}

bool EriscH2HSocket::submit(const BridgeSendTask& task) {
    Impl& im = *impl_;
    if (im.broken || task.arena >= im.cfg.arenas) {
        return false;
    }
    // Bounded by the FRAME limit, so a caller cannot queue past the Tx slot count and overwrite a
    // slot whose put has not run.
    if (im.cfg.max_queued_frames != 0 && im.queue.size() >= im.cfg.max_queued_frames) {
        ++im.stats.starved_window;
        return false;
    }
    im.queue.push_back(task);
    return true;
}

uint32_t EriscH2HSocket::poll(const Retire& retire, const Deliver& deliver) {
    Impl& im = *impl_;
    if (im.broken) {
        return 0;
    }
    ++im.stats.passes;
    uint32_t progress = 0;
    const uint32_t peer = im.cfg.peer_rank;
    const uint32_t slot_sz = im.g.slot();

    // ---- send: payloads, ONE flush, descriptors, ONE flush -- two round trips per batch ----
    // Half the window per batch, so one half is put while the RX drains the other (a whole-window
    // batch is stop-and-wait). A starved arena is skipped rather than blocking the rest.
    const uint32_t cap =
        im.cfg.max_batch != 0 ? im.cfg.max_batch : std::max<uint32_t>(1, im.cfg.arenas * im.g.ring / 2);
    im.closed.assign(im.cfg.arenas, 0);
    im.sel.clear();
    im.batch_slot.clear();
    for (uint32_t i = 0; i < im.queue.size() && im.sel.size() < cap; ++i) {
        const uint32_t a = im.queue[i].arena;
        if (im.closed[a] != 0) {
            continue;
        }
        // A slot is reused only once the RX consumed it. The credit is the PEER's put, so
        // poke_progress advances it, not flush(peer).
        if (im.posted[a] + 1 - *im.credit_word(a) > im.g.ring) {
            ++im.stats.starved_credit;
            im.closed[a] = 1;
            continue;
        }
        im.sel.push_back(i);
        im.batch_slot.push_back(static_cast<uint32_t>(im.posted[a]++ % im.g.ring));
    }
    const std::size_t n = im.sel.size();
    if (im.ops.size() < n) {
        im.ops.resize(n);
    }
    // Runs of consecutive slots in one arena: one put per run, then one flush. Descriptors sit
    // 32 B apart and payloads one slot apart, so a run is contiguous at both ends.
    const auto put_runs = [&](bool descs) {
        std::size_t nops = 0;
        std::string pe;
        for (std::size_t i = 0, j = 0; i < n && pe.empty(); i = j) {
            const BridgeSendTask& t = im.queue[im.sel[i]];
            const uint32_t s0 = im.batch_slot[i];
            for (j = i + 1; j < n; ++j) {
                const BridgeSendTask& u = im.queue[im.sel[j]];
                const bool joins =
                    u.arena == t.arena && im.batch_slot[j] == s0 + (j - i) &&
                    (descs || (im.queue[im.sel[j - 1]].page_bytes >= slot_sz &&
                               u.page_offset == t.page_offset + (j - i) * slot_sz &&
                               (im.cfg.max_put_bytes == 0 || (j - i + 1) * slot_sz <= im.cfg.max_put_bytes)));
                if (!joins) {
                    break;
                }
            }
            if (descs) {
                pe = im.win->put(
                    im.base() + desc_off(im.g, t.arena, BridgeArena::Tx, s0),
                    (j - i) * kBridgeDescriptorBytes,
                    peer,
                    desc_off(im.g, t.arena, BridgeArena::Rx, s0),
                    im.ops[nops++]);
                ++im.stats.trailer_puts;
            } else {
                pe = im.win->put(
                    im.base() + t.page_offset,
                    (j - i - 1) * slot_sz + std::min<uint32_t>(im.queue[im.sel[j - 1]].page_bytes, slot_sz),
                    peer,
                    slot_off(im.g, t.arena, BridgeArena::Rx, s0),
                    im.ops[nops++]);
                ++im.stats.payload_puts;
            }
        }
        if (pe.empty()) {
            pe = im.win->flush(peer);
        }
        for (std::size_t i = 0; i < nops; ++i) {
            (void)im.win->test(im.ops[i]);  // complete after the flush; releases the Rput request
        }
        ++im.stats.flushes;
        return pe;
    };
    if (n != 0) {
        // Payloads first: no guard may be issued until every payload is remote.
        std::string e = put_runs(false);
        for (std::size_t i = 0; i < n && e.empty(); ++i) {
            const BridgeSendTask& t = im.queue[im.sel[i]];
            make_desc(
                reinterpret_cast<BridgeDescriptor*>(
                    im.base() + desc_off(im.g, t.arena, BridgeArena::Tx, im.batch_slot[i])),
                im.g,
                t.ordering_cntr,
                t.far_slot.value_or(kWireUnknownFarSlot),
                t.release_cyc,
                t.elapsed,
                t.length);
        }
        if (e.empty()) {
            e = put_runs(true);
        }
        if (!e.empty()) {
            im.fail(e);
            return progress;
        }
        // After the guards' flush: a frame is not visible to the peer before its trailer is remote.
        if (im.cfg.collect_timing) {
            const auto now = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < n; ++i) {
                im.put_at[static_cast<std::size_t>(im.queue[im.sel[i]].arena) * im.g.ring + im.batch_slot[i]] = now;
            }
        }
        // Drop the sent frames, keeping the rest in order.
        std::size_t w = 0;
        for (std::size_t r = 0, k = 0; r < im.queue.size(); ++r) {
            if (k < n && im.sel[k] == r) {
                ++k;
            } else {
                im.queue[w++] = im.queue[r];
            }
        }
        im.queue.erase(im.queue.begin() + static_cast<std::ptrdiff_t>(w), im.queue.end());
        im.stats.posts += n;
        progress += static_cast<uint32_t>(n);
    }
    if (im.queue.empty()) {
        ++im.stats.starved_empty;
    }

    // Receive. Poke first: flush completes only OUR puts and sync is a barrier, so neither
    // applies the peer's (contract §7.2d).
    (void)im.win->poke_progress();
    (void)im.win->sync();

    // In order from the next expected slot, so one pass delivers every frame that has landed.
    for (uint32_t a = 0; a < im.cfg.arenas; ++a) {
        for (uint32_t k = 0; k < im.g.ring; ++k) {
            const uint32_t s = im.rx_slot[a];
            auto* d = reinterpret_cast<volatile BridgeDescriptor*>(im.base() + desc_off(im.g, a, BridgeArena::Rx, s));
            if (!bridge_guard_armed(d->guard)) {
                break;
            }
            // ACQUIRE: the payload reads below must not be hoisted above the guard.
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint32_t cntr = d->ordering_cntr;
            if (bridge_guard_seq(d->guard) != cntr) {
                break;  // MPI orders no bytes inside one put: the rest of the descriptor is still landing
            }
            if (cntr != im.delivered[a] + 1) {
                // The expected slot holds a repeat or a stray. Counted once per descriptor, not per pass.
                auto& last = im.flagged[static_cast<std::size_t>(a) * im.g.ring + s];
                if (last != cntr) {
                    last = cntr;
                    ++im.stats.bad_order;
                }
                break;
            }

            BridgeDeliverTask dt;
            dt.arena = a;
            dt.slot = s;
            dt.page_offset = slot_off(im.g, a, BridgeArena::Rx, s);
            dt.page_bytes = slot_sz;
            dt.ordering_cntr = cntr;
            dt.length = d->length;
            dt.origin = a;
            dt.elapsed = d->elapsed;
            dt.release_cyc = d->release_cyc;
            if (const uint16_t far = d->slot_idx; far != kWireUnknownFarSlot) {
                dt.far_slot = far;
            }

            // FALSE MEANS NOT TAKEN: leave the guard armed and re-offer next pass.
            if (!deliver(dt)) {
                break;
            }
            im.delivered[a] = cntr;
            d->guard = 0;  // disarm, so a reused slot cannot read fresh
            im.rx_slot[a] = (s + 1) % im.g.ring;
            ++progress;
        }
    }

    // ---- retire what the peer credited (RTT stamps close here even with no retire callback) ----
    for (uint32_t a = 0; a < im.cfg.arenas; ++a) {
        const uint64_t done = *im.credit_word(a);
        if (done > im.credited[a]) {
            // put -> credit closes here, one stamp per acknowledged frame, and only while
            // that frame's stamp is still in the ring -- a stale slot would time another frame.
            if (im.cfg.collect_timing) {
                const auto now = std::chrono::steady_clock::now();
                for (uint64_t k = im.credited[a]; k < done; ++k) {
                    if (done - k <= im.g.ring) {
                        const auto& t0 = im.put_at[static_cast<std::size_t>(a) * im.g.ring + (k % im.g.ring)];
                        if (t0.time_since_epoch().count() != 0) {
                            im.rtt_ns.push_back(static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count()));
                        }
                    }
                }
            }
            if (retire) {
                retire(a, static_cast<uint32_t>(done - im.credited[a]));
            }
            im.credited[a] = done;
        }
    }
    if (progress == 0) {
        ++im.stats.starved;
    }
    im.stats.in_flight_sum += im.queue.size();
    if (im.cfg.collect_timing && im.series.in_flight.size() < 65536) {
        im.series.in_flight.push_back(static_cast<uint32_t>(im.queue.size()));
    }
    return progress;
}

void EriscH2HSocket::consumed(uint32_t arena, uint32_t pages) {
    Impl& im = *impl_;
    if (arena < im.cfg.arenas) {
        im.consumed_total[arena] += pages;
    }
}

bool EriscH2HSocket::publish_credits() {
    Impl& im = *impl_;
    if (im.broken) {
        return false;
    }
    bool any = false;
    for (uint32_t a = 0; a < im.cfg.arenas; ++a) {
        // ABSOLUTE total of what the consumer released (not what was delivered), so a lost or
        // duplicated credit is a no-op and a slot still being read is never handed back.
        if (im.consumed_total[a] == im.published[a]) {
            continue;
        }
        if (const std::string e = im.win->put_word(im.consumed_total[a], im.cfg.peer_rank, credit_off(im.g, a));
            !e.empty()) {
            im.fail(e);
            return false;
        }
        im.published[a] = im.consumed_total[a];
        ++im.stats.credit_puts;
        any = true;
    }
    if (any) {
        if (const std::string e = im.win->flush(im.cfg.peer_rank); !e.empty()) {
            im.fail(e);
            return false;
        }
    }
    return any;
}

uint64_t EriscH2HSocket::credit_total(uint32_t arena) const {
    const Impl& im = *impl_;
    return arena < im.cfg.arenas ? static_cast<uint64_t>(*im.credit_word(arena)) : 0;
}

const std::vector<uint64_t>& EriscH2HSocket::put_to_credit_ns() const { return impl_->rtt_ns; }
const EriscH2HSocket::PassStats& EriscH2HSocket::pass_stats() const { return impl_->stats; }
const EriscH2HSocket::Series& EriscH2HSocket::series() const { return impl_->series; }

void EriscH2HSocket::reset_stats() {
    impl_->stats = PassStats{};
    impl_->series = Series{};
    impl_->rtt_ns.clear();
}

std::string EriscH2HSocket::barrier() { return impl_->win ? impl_->win->barrier() : std::string{}; }
bool EriscH2HSocket::failed() const { return impl_->broken; }
std::string EriscH2HSocket::first_error() const { return impl_->err; }

std::string EriscH2HSocket::describe() const {
    const Impl& im = *impl_;
    return "erisc H2H: " + std::to_string(im.cfg.arenas) + " arenas x " + std::to_string(im.g.ring) + " slots x " +
           std::to_string(im.g.slot()) + " B, peer rank " + std::to_string(im.cfg.peer_rank) + ", posts " +
           std::to_string(im.stats.posts) + ", flushes " + std::to_string(im.stats.flushes);
}

}  // namespace tt::tt_metal::experimental
