// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_h2h_socket.hpp"

#include <deque>
#include <vector>

#include <fmt/format.h>

#include "tt_metal/distributed/host_rdma_window.hpp"
#include "tt_metal/distributed/host_uva_frame.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

namespace {

// A load the compiler may not hoist out of a poll loop; acquire orders the trailer's other
// fields after the guard that vouches for them.
uint64_t load_acquire(const volatile uint64_t* p) {
    return __atomic_load_n(const_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE);
}
void store_release(volatile uint64_t* p, uint64_t v) {
    __atomic_store_n(const_cast<uint64_t*>(p), v, __ATOMIC_RELEASE);
}

}  // namespace

// Single-threaded: the caller drives poll(). No atomics, no locks.
struct H2HSocket::Impl {
    Config cfg{};
    std::unique_ptr<RdmaWindow> win;
    uint32_t window_cap = 0;

    // Per core, oldest first: one shared deque would park a completed put behind an
    // outstanding one on another core, and the D2H FIFO is freed per core anyway.
    struct InFlight {
        RdmaWindow::Op op{};
    };
    // 1 tx_queue per core. A shared tx_queue lets a core waiting on credit park every other
    // core's sends behind it, which is the stall the previous design fixed the same way.
    std::vector<std::deque<SendTask>> tx_queue;
    std::vector<std::deque<InFlight>> tx_flight;
    uint64_t in_flight = 0;
    uint64_t tx_queued = 0;
    uint32_t rr = 0;

    // Per (my core, peer host). Two DIFFERENT counts that must not share storage: what we
    // have posted to that peer, and what we have credited back to it.
    std::vector<uint64_t> posted;

    // Frames handed to the H2D leg and not yet reported consumed, per core, IN ORDER.
    // Recording the order removes every place an origin had to be re-derived.
    struct Delivered {
        uint32_t origin = 0;  // sender's full selector: its host AND its core
        uint32_t slot = 0;    // the RX slot it landed in, so consumed() can disarm its guard
    };
    std::vector<std::deque<Delivered>> rx_pending;
    std::vector<uint64_t> credit_out;  // frames this host has credited back, per RECEIVING core
    std::vector<uint64_t> done_out;    // the same frames counted per SENDING core
    std::vector<uint32_t> next_slot;   // next RX slot to inspect, per core
    std::vector<bool> dirty;           // peers with an unflushed put

    bool broken = false;
    std::string err;

    void fail(const std::string& what) {
        broken = true;
        if (err.empty()) {
            err = what;
        }
    }

    // Payload and credit puts alike are flush_local only, so nothing is visible on the peer
    // until this runs. Batched per peer: a flush is a round trip, so an idle one is latency.
    void flush_dirty() {
        for (uint32_t h = 0; h < cfg.topo.num; ++h) {
            if (!dirty[h]) {
                continue;
            }
            dirty[h] = false;
            if (const std::string e = win->flush(h); !e.empty()) {
                fail("h2h: " + e);
            }
        }
    }

    uint64_t& posted_at(uint32_t core, uint32_t host) { return posted[core * kMaxCreditPeers + host]; }
    uint64_t& credit_out_at(uint32_t core, uint32_t host) { return credit_out[core * kMaxCreditPeers + host]; }
    uint64_t& done_out_at(uint32_t core, uint32_t host) { return done_out[core * kMaxCreditPeers + host]; }

    // The peer writes an absolute count into our region at credit_offset(my core, its host).
    uint64_t credit_seen(uint32_t core, uint32_t host) const {
        const auto* w = reinterpret_cast<const volatile uint64_t*>(cfg.region_base + credit_offset(core, host));
        return load_acquire(w);
    }
    // The same, from the array keyed on the SENDING core: what OUR core has had pulled.
    uint64_t done_seen(uint32_t core, uint32_t host) const {
        const auto* w = reinterpret_cast<const volatile uint64_t*>(cfg.region_base + done_offset(core, host));
        return load_acquire(w);
    }

    volatile uint64_t* trailer_guard(uint32_t core, uint32_t slot) const {
        uint8_t* const page = cfg.region_base + rx_slot_offset(core, slot, cfg.page_bytes, cfg.rx_data_offset);
        return reinterpret_cast<volatile uint64_t*>(page + cfg.page_bytes - kFrameTrailerBytes);
    }
    const FrameTrailer* trailer(uint32_t core, uint32_t slot) const {
        uint8_t* const page = cfg.region_base + rx_slot_offset(core, slot, cfg.page_bytes, cfg.rx_data_offset);
        return reinterpret_cast<const FrameTrailer*>(page + cfg.page_bytes - kFrameTrailerBytes);
    }
};

H2HSocket::H2HSocket() : impl_(std::make_unique<Impl>()) {}
H2HSocket::~H2HSocket() = default;

std::unique_ptr<H2HSocket> H2HSocket::create(const Config& cfg, std::string& err) {
    err.clear();
    if (!host_topology_ok(cfg.topo) || cfg.topo.num < 2) {
        err =
            "H2HSocket: the path is chip->host->host->chip and needs an addressable topology of "
            "at least two hosts";
        return nullptr;
    }
    if (cfg.topo.num > kMaxHosts) {
        err = "H2HSocket: " + std::to_string(cfg.topo.num) + " hosts exceeds the " + std::to_string(kMaxHosts) +
              " the credit array indexes";
        return nullptr;
    }
    // Refused, not merely unimplemented: rx_slot_offset() has no host dimension and the
    // receiver keeps one cursor per core, so two senders into one ring would collide.
    if (cfg.topo.num > 2) {
        err = "H2HSocket: " + std::to_string(cfg.topo.num) +
              " hosts is not supported; the RX ring is not partitioned per origin (2 max)";
        return nullptr;
    }
    if (cfg.cores == 0 || cfg.page_bytes == 0 || cfg.region_base == nullptr) {
        err = "H2HSocket: cores, page_bytes and region_base are all required";
        return nullptr;
    }
    // This class computes credit_offset() and rx_slot_offset() itself, so it has to bound
    // them itself: HostRegion and RingAlias each only police their own view of the region.
    if (cfg.cores > kProvisionedCores) {
        err = "H2HSocket: " + std::to_string(cfg.cores) + " cores exceeds the " + std::to_string(kProvisionedCores) +
              " the credit and done arrays index";
        return nullptr;
    }
    if (cfg.ring_pages == 0) {
        err = "H2HSocket: ring_pages must be at least 1";
        return nullptr;
    }
    if (cfg.rx_data_offset + static_cast<uint64_t>(cfg.ring_pages) * cfg.page_bytes > kArenaBytes) {
        err = "H2HSocket: rx_data_offset + ring_pages x page_bytes (" + std::to_string(cfg.rx_data_offset) + " + " +
              std::to_string(cfg.ring_pages) + " x " + std::to_string(cfg.page_bytes) + ") exceeds the " +
              std::to_string(kArenaBytes >> 10) + " KiB arena";
        return nullptr;
    }

    std::unique_ptr<H2HSocket> s(new H2HSocket());
    Impl& im = *s->impl_;
    im.cfg = cfg;
    im.window_cap = cfg.send_window != 0 ? cfg.send_window : cfg.cores * cfg.ring_pages;

    im.win = RdmaWindow::create(cfg.region_base, cfg.region_bytes, cfg.topo.ident, cfg.topo.num, err);
    if (!im.win) {
        return nullptr;
    }

    const size_t per_peer = static_cast<size_t>(cfg.cores) * kMaxCreditPeers;
    im.posted.assign(per_peer, 0);
    im.credit_out.assign(per_peer, 0);
    im.done_out.assign(per_peer, 0);
    im.rx_pending.assign(cfg.cores, {});
    im.next_slot.assign(cfg.cores, 0);
    im.tx_queue.assign(cfg.cores, {});
    im.tx_flight.assign(cfg.cores, {});
    im.dirty.assign(cfg.topo.num, false);
    return s;
}

bool H2HSocket::submit(const SendTask& task) {
    Impl& im = *impl_;
    if (im.broken || task.core >= im.cfg.cores) {
        return false;
    }
    // The put below takes its length from the frame and its target offset from the socket's
    // geometry, so a mismatched page would overrun into the peer's next arena, remotely.
    if (task.page_bytes != im.cfg.page_bytes) {
        im.fail(
            "h2h: a frame's page size (" + std::to_string(task.page_bytes) + ") does not match the ring's (" +
            std::to_string(im.cfg.page_bytes) + ")");
        return false;
    }
    // A core can have at most ring_pages in tx_flight, so queueing more just defers the gate.
    if (im.tx_queue[task.core].size() >= im.cfg.ring_pages) {
        return false;
    }
    im.tx_queue[task.core].push_back(task);
    ++im.tx_queued;
    return true;
}

uint32_t H2HSocket::poll(const Retire& retire, const Deliver& deliver) {
    Impl& im = *impl_;
    uint32_t progress = 0;
    if (im.broken || !deliver) {
        return 0;
    }

    // Front only, per core: bytes_acked is one counter and can only cross a contiguous
    // prefix. LOAD-BEARING: flush_dirty() ran a pass earlier, so a test here is remote.
    for (uint32_t c = 0; c < im.cfg.cores; ++c) {
        while (!im.tx_flight[c].empty() && im.win->test(im.tx_flight[c].front().op)) {
            im.tx_flight[c].pop_front();
            --im.in_flight;
            if (retire) {
                retire(c, 1);
            }
            ++progress;
        }
    }

    // Start what the window allows, round-robin across cores. A gated core is SKIPPED,
    // never broken on: that is the whole point of the per-core queues.
    for (uint32_t k = 0; k < im.cfg.cores && im.tx_queued != 0 && im.in_flight < im.window_cap; ++k) {
        const uint32_t c = (im.rr + k) % im.cfg.cores;
        if (im.tx_queue[c].empty()) {
            continue;
        }
        const SendTask& t = im.tx_queue[c].front();
        const uint32_t host = tt_uva_target_host(t.dst, im.cfg.topo);
        const uint32_t dest_core = tt_uva_t6_core(t.dst);
        // dest_core indexes our own per-peer arrays as well as the target's ring, and it
        // comes out of a UVA, so it is bounded here and not trusted to be one of ours.
        if (host == kHostNone || host >= im.cfg.topo.num || host == im.cfg.topo.ident || dest_core >= im.cfg.cores) {
            im.fail(fmt::format(
                "h2h: core {} addressed host {} core {}, which is not a peer of this symmetric socket",
                t.core,
                host,
                dest_core));
            im.tx_queue[c].pop_front();
            --im.tx_queued;
            break;
        }

        // Keyed on the DESTINATION core, not this one: the ring being filled belongs to the
        // target, so every sender into it must draw slots and credit from one counter.
        if (im.posted_at(dest_core, host) - im.credit_seen(dest_core, host) >= im.cfg.ring_pages) {
            continue;
        }

        const uint32_t slot = static_cast<uint32_t>(im.posted_at(dest_core, host) % im.cfg.ring_pages);
        Impl::InFlight f;
        if (const std::string e = im.win->put(
                im.cfg.region_base + t.page_offset,
                t.page_bytes,
                host,
                rx_slot_offset(dest_core, slot, im.cfg.page_bytes, im.cfg.rx_data_offset),
                f.op);
            !e.empty()) {
            im.fail("h2h: " + e);
            break;
        }
        im.posted_at(dest_core, host)++;
        im.dirty[host] = true;
        im.tx_flight[t.core].push_back(f);
        ++im.in_flight;
        im.tx_queue[c].pop_front();
        --im.tx_queued;
        ++progress;
    }
    im.rr = im.cfg.cores != 0 ? (im.rr + 1) % im.cfg.cores : 0;
    // Both failure paths above land here; neither should go on to flush or harvest.
    if (im.broken) {
        return progress;
    }

    // After the starts and before the next pass's retire loop: that gap is what makes an
    // acked frame mean "in the peer's window" rather than "handed to MPI" -- see tt_uva_quiet().
    im.flush_dirty();

    // Harvest arrivals. The trailer is the last thing the peer's put writes, so an armed
    // guard means the payload ahead of it landed -- see the trailing-flag note.
    for (uint32_t c = 0; c < im.cfg.cores; ++c) {
        // The whole ring, not one slot: at depth > 1 a later arrival is otherwise invisible
        // until every poll before it has run.
        for (uint32_t n = 0; n < im.cfg.ring_pages; ++n) {
            // The guard stays armed until consumed(), so it no longer says "not yet taken".
            // This bound does: at ring_pages outstanding, next_slot cannot lap onto a live one.
            if (im.rx_pending[c].size() >= im.cfg.ring_pages) {
                break;
            }
            const uint32_t slot = im.next_slot[c];
            volatile uint64_t* const guard = im.trailer_guard(c, slot);
            if (!tt_uva_frame_armed(load_acquire(guard))) {
                break;
            }
            const FrameTrailer* const t = im.trailer(c, slot);

            DeliverTask d;
            d.core = c;
            d.slot = slot;
            d.page_offset = rx_slot_offset(c, slot, im.cfg.page_bytes, im.cfg.rx_data_offset);
            d.page_bytes = im.cfg.page_bytes;
            d.dst = static_cast<tt_uva_t>(t->dst);
            d.length = t->length;
            d.origin = t->origin;
            d.elapsed = t->elapsed;
            if (!deliver(d)) {
                break;
            }

            // Disarmed in consumed(), not here: deliver() above already released the far
            // device to pull this page, trailer included, and a zero would race that read.
            im.rx_pending[c].push_back(Impl::Delivered{t->origin, slot});
            im.next_slot[c] = (slot + 1) % im.cfg.ring_pages;
            ++progress;
        }
    }
    return progress;
}

// Credits the oldest `pages` frames this core was handed; each carries its own origin.
// Counts are absolute, so a lost or duplicated credit is a no-op.
void H2HSocket::consumed(uint32_t core, uint32_t pages) {
    Impl& im = *impl_;
    if (core >= im.cfg.cores) {
        return;
    }
    for (; pages != 0 && !im.rx_pending[core].empty(); --pages) {
        const Impl::Delivered d = im.rx_pending[core].front();
        im.rx_pending[core].pop_front();

        // The H2D leg has reported this page drained, so the device is done reading it.
        // Still before the credit: a credit lets the peer re-arm the slot.
        store_release(im.trailer_guard(core, d.slot), 0);

        const uint32_t host = tt_uva_t6_selector_host(d.origin, im.cfg.topo.chips_per_host);
        const uint32_t src_core = tt_uva_t6_selector_core(d.origin);
        // cfg.cores, not kProvisionedCores: src_core indexes done_out, which is sized by it.
        if (host >= im.cfg.topo.num || host == im.cfg.topo.ident || src_core >= im.cfg.cores) {
            im.fail(
                "h2h: a delivered frame named origin selector " + std::to_string(d.origin) +
                ", which is not a peer core");
            return;
        }
        // Keyed on THIS core -- the ring that just freed a slot -- at our host id, which is
        // exactly where every sender into this ring reads its gate.
        uint64_t& n = im.credit_out_at(core, host);
        ++n;
        if (const std::string e = im.win->put_word(n, host, credit_offset(core, im.cfg.topo.ident)); !e.empty()) {
            im.fail("h2h: credit: " + e);
            return;
        }
        // The same frame counted against its SENDER, which is what tt_uva_sync() waits on.
        // Two counts because a slot freeing and a sender's frame landing are different facts.
        uint64_t& m = im.done_out_at(src_core, host);
        ++m;
        if (const std::string e = im.win->put_word(m, host, done_offset(src_core, im.cfg.topo.ident)); !e.empty()) {
            im.fail("h2h: done: " + e);
            return;
        }
        im.dirty[host] = true;
    }
    // More consumed than delivered means the two legs disagree about what was handed over.
    if (pages != 0) {
        im.fail(
            "h2h: the H2D leg reported more frames consumed on core " + std::to_string(core) +
            " than were delivered to it");
    }
}

// Frames THIS core put that a far device has pulled -- the done array, not the credit one.
// tt_uva_sync() compares it against its own put count, so it has to be exactly that.
uint64_t H2HSocket::credit_total(uint32_t core) const {
    const Impl& im = *impl_;
    uint64_t sum = 0;
    for (uint32_t h = 0; h < im.cfg.topo.num; ++h) {
        if (h != im.cfg.topo.ident) {
            sum += im.done_seen(core, h);
        }
    }
    return sum;
}

std::string H2HSocket::barrier() {
    impl_->flush_dirty();
    return impl_->win->barrier();
}
bool H2HSocket::failed() const { return impl_->broken; }
std::string H2HSocket::first_error() const { return impl_->err; }

std::string H2HSocket::describe() const {
    return fmt::format(
        "h2h: {}, window {} frame(s), ring depth {}", impl_->win->describe(), impl_->window_cap, impl_->cfg.ring_pages);
}

}  // namespace tt::tt_metal::experimental
