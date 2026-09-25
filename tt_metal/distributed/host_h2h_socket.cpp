// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_h2h_socket.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <tt_stl/aligned_allocator.hpp>

#include "tt_metal/distributed/host_rdma_window.hpp"
#include "tt_metal/distributed/host_uva_frame.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Every field both hosts must lay out identically, rendered once so the hash they agree on
// and the message a mismatch prints can never describe different fields.
// Not region_bytes: each host pins its own prefix, and the bound on it is checked locally.
std::string geometry_text(const H2HSocket::Config& cfg) {
    // Delimited rather than run together: "1" then "23" and "12" then "3" would otherwise
    // render alike, which is the one way a text encoding of a tuple stops being injective.
    return fmt::format(
        "cores {}, page {} B, ring {}, rx_data_offset {}, hosts {}, chips_per_host {}",
        cfg.cores,
        cfg.page_bytes,
        cfg.ring_pages,
        cfg.rx_data_offset,
        cfg.topo.num,
        cfg.topo.chips_per_host);
}

// Hashed as text, so neither padding nor byte order reaches std::hash. agree_value compares
// this across processes, which holds while both hosts run one build of one standard library.
std::size_t geometry_fingerprint(const H2HSocket::Config& cfg) {
    return std::hash<std::string>{}(geometry_text(cfg));
}

// A load the compiler may not hoist out of a poll loop; acquire orders the trailer's other
// fields after the guard that vouches for them.
// One flat allocation per container, sized at create() and never grown: every queue here is
// bounded by ring_pages, so a deque's chunk churn buys nothing and costs a pointer chase.
template <typename T>
class CoreRings {
public:
    // Rounded up to a power of two so the wrap is a mask: a runtime `% cap` is an integer
    // division, and it would run on every push and every pop of the hot path.
    void reset(uint32_t cores, uint32_t cap) {
        cap_ = 1;
        while (cap_ < cap) {
            cap_ <<= 1;
        }
        mask_ = cap_ - 1;
        limit_ = cap;
        buf_.assign(static_cast<size_t>(cores) * cap_, T{});
        head_.assign(cores, 0);
        count_.assign(cores, 0);
        live_[0] = 0;
        live_[1] = 0;
    }
    // Cores holding at least one entry. A sweep over all of them costs 64 visits per pass
    // whether two cores have work or none do, and a pass runs several times per frame.
    template <typename F>
    void for_each_live(F&& f) const {
        for (uint32_t w = 0; w < 2; ++w) {
            // Snapshotted, so f() popping a core's last entry cannot disturb this walk.
            uint64_t m = live_[w];
            while (m != 0) {
                const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(m));
                m &= m - 1;
                f(w * 64 + b);
            }
        }
    }
    bool any_live() const { return (live_[0] | live_[1]) != 0; }
    bool empty(uint32_t c) const { return count_[c] == 0; }
    uint32_t size(uint32_t c) const { return count_[c]; }
    T& front(uint32_t c) { return buf_[static_cast<size_t>(c) * cap_ + head_[c]]; }
    // i-th from the front. Coalescing has to look past the head to measure how long a
    // contiguous run is before it issues anything.
    const T& at(uint32_t c, uint32_t i) const {
        return buf_[static_cast<size_t>(c) * cap_ + ((head_[c] + i) & mask_)];
    }
    // False means full. Every caller checks its own bound first; this is the backstop that
    // turns a protocol bug into a dropped frame rather than an overwrite.
    bool push_back(uint32_t c, const T& v) {
        // limit_, not cap_: the protocol bound is ring_pages, and the rounding is slack.
        if (count_[c] >= limit_) {
            return false;
        }
        buf_[static_cast<size_t>(c) * cap_ + ((head_[c] + count_[c]) & mask_)] = v;
        if (++count_[c] == 1) {
            live_[c >> 6] |= 1ull << (c & 63);
        }
        return true;
    }
    void pop_front(uint32_t c) {
        if (count_[c] != 0) {
            head_[c] = (head_[c] + 1) & mask_;
            if (--count_[c] == 0) {
                live_[c >> 6] &= ~(1ull << (c & 63));
            }
        }
    }

private:
    // Cache-line aligned: once detector threads own contiguous core ranges, an unaligned
    // base lets one thread's tail share a line with the next thread's head.
    std::vector<T, ttsl::aligned_allocator<T, 64>> buf_;
    std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, 64>> head_;
    std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, 64>> count_;
    // One bit per core; kProvisionedCores is 128, so two words cover every layout.
    uint64_t live_[2] = {0, 0};
    uint32_t cap_ = 0;    // allocation stride, a power of two
    uint32_t mask_ = 0;   // cap_ - 1
    uint32_t limit_ = 0;  // the protocol bound this was asked for
};

// The backward counters and their puts. Both are absolute and monotone, so every value but
// the last in a batch is superseded: one put per peer per pass replaces two per frame.
class CreditPublisher {
public:
    void reset(uint32_t cores, uint32_t hosts) {
        cores_ = cores;
        credit_.assign(static_cast<size_t>(hosts) * cores, 0);
        done_.assign(static_cast<size_t>(hosts) * cores, 0);
        credit_dirty_.assign(hosts, Span{});
        done_dirty_.assign(hosts, Span{});
    }

    uint64_t& credit(uint32_t host, uint32_t core) { return credit_[idx(host, core)]; }
    uint64_t& done(uint32_t host, uint32_t core) { return done_[idx(host, core)]; }

    void note_credit(uint32_t host, uint32_t core) {
        ++credit_[idx(host, core)];
        credit_dirty_[host].add(core);
    }
    void note_done(uint32_t host, uint32_t core) {
        ++done_[idx(host, core)];
        done_dirty_[host].add(core);
    }

    bool pending(uint32_t host) const { return !credit_dirty_[host].empty() || !done_dirty_[host].empty(); }

    // One put per array over lo..hi; unchanged cores inside the span are re-put, which is
    // idempotent. Source is the live array: a racing increment can only raise a word.
    template <typename Put>
    std::string publish(uint32_t host, uint32_t ident, Put&& put) {
        if (const std::string e = emit(credit_dirty_[host], credit_, host, ident, credit_offset, put, credit_puts_);
            !e.empty()) {
            return e;
        }
        return emit(done_dirty_[host], done_, host, ident, done_offset, put, done_puts_);
    }

    uint64_t credit_puts() const { return credit_puts_; }
    uint64_t done_puts() const { return done_puts_; }

private:
    // Lowest and highest core touched since the last publish. A span, not a bitmask: the
    // put is one contiguous range either way, so the endpoints are all emit needs.
    struct Span {
        uint32_t lo = UINT32_MAX;
        uint32_t hi = 0;
        bool empty() const { return lo > hi; }
        void add(uint32_t c) {
            lo = c < lo ? c : lo;
            hi = c > hi ? c : hi;
        }
        void clear() {
            lo = UINT32_MAX;
            hi = 0;
        }
    };

    size_t idx(uint32_t host, uint32_t core) const { return static_cast<size_t>(host) * cores_ + core; }

    template <typename Off, typename Put>
    std::string emit(
        Span& d, std::vector<uint64_t>& src, uint32_t host, uint32_t ident, Off&& off, Put&& put, uint64_t& count) {
        if (d.empty()) {
            return {};
        }
        const uint32_t n = d.hi - d.lo + 1;
        const std::string e =
            put(&src[idx(host, d.lo)], static_cast<uint64_t>(n) * sizeof(uint64_t), off(d.lo, ident));
        if (!e.empty()) {
            return e;
        }
        d.clear();
        ++count;
        return {};
    }

    std::vector<uint64_t> credit_;  // [host][core]
    std::vector<uint64_t> done_;    // [host][core]
    std::vector<Span> credit_dirty_;
    std::vector<Span> done_dirty_;
    uint64_t credit_puts_ = 0;
    uint64_t done_puts_ = 0;
    uint32_t cores_ = 0;
};

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

    // Per core, oldest first: one shared ring would park a completed put behind an
    // outstanding one on another core, and the D2H FIFO is freed per core anyway.
    struct InFlight {
        RdmaWindow::Op op{};
        // Kept so the trailer can be put in a later pass: the payload put does not carry
        // the guard, so everything needed to publish it has to survive until then.
        uint32_t host = 0;
        uint32_t dest_core = 0;
        uint32_t slot = 0;   // first slot of the run
        uint32_t count = 1;  // frames in the run; one put covers all of them
        uint64_t src_off = 0;
        // The flush count this put was issued under. A flush may now span several passes,
        // so the pass boundary no longer proves visibility -- this does.
        uint64_t epoch = 0;
    };
    // 1 tx_queue per core. A shared tx_queue lets a core waiting on credit park every other
    // core's sends behind it, which is the stall the previous design fixed the same way.
    CoreRings<SendTask> tx_queue;
    // Three stages per frame: payload put (tx_payload), then -- once a flush has made it
    // remotely visible -- the trailer put (tx_flight), then retire on its completion.
    CoreRings<InFlight> tx_payload;
    CoreRings<InFlight> tx_trailer;
    CoreRings<InFlight> tx_flight;
    uint64_t in_flight = 0;
    uint64_t tx_queued = 0;
    uint32_t rr = 0;
    PassStats stats{};

    // Per (my core, peer host). Two DIFFERENT counts that must not share storage: what we
    // have posted to that peer, and what we have credited back to it.
    std::vector<uint64_t> posted;

    // Frames handed to the H2D leg and not yet reported consumed, per core, IN ORDER.
    // Recording the order removes every place an origin had to be re-derived.
    struct Delivered {
        uint32_t origin = 0;  // sender's full selector: its host AND its core
        uint32_t slot = 0;    // the RX slot it landed in, so consumed() can disarm its guard
    };
    CoreRings<Delivered> rx_pending;
    // Owns both backward counters and the coalesced puts that publish them.
    CreditPublisher credits;
    RdmaWindow::Op credit_op{};  // MPI_Put takes no request; kept only to satisfy put()
    std::vector<uint32_t> next_slot;   // next RX slot to inspect, per core
    // The origin's frame index this ring expects next. next_slot wraps at ring_pages and so
    // cannot tell a slot re-armed before its credit from the frame that belongs there.
    std::vector<uint64_t> rx_seq;
    // Bytes, not a flag: a flush costs one ~7 us round trip whatever it covers, so the
    // amount pending is the only thing that says whether issuing one now is worth it.
    std::vector<uint64_t> pending;
    // Credit bytes awaiting confirmation, and when the oldest was issued.
    std::vector<uint64_t> ctrl_pending;
    std::vector<std::chrono::steady_clock::time_point> first_ctrl;

    // put -> credit, collected here because nothing above this class can see either end:
    // submit() only queues, and the credit word is read by credit_seen() alone.
    // One stamp per live frame, indexed exactly as the RX slot is, so the ring that bounds
    // frames in flight bounds this too.
    std::vector<std::chrono::steady_clock::time_point> put_at;
    std::vector<uint64_t> credit_closed;  // sequences already turned into samples
    std::vector<uint64_t> put_to_credit_ns;

    bool broken = false;
    std::string err;

    // One-way: agree() decides bring-up together, but a frame has no such channel -- the
    // credit words are absolute counts with no value spare for a poison marker.
    void fail(const std::string& what) {
        broken = true;
        if (err.empty()) {
            err = what;
        }
    }

    // 768 KiB: a flush is a fixed ~7.2 us round trip, and at the measured 12.18 GB/s
    // marginal rate that many bytes puts its cost near 10% of the transfer -- 55 frames at
    // 14 KB, 3 at 256 KB. Bytes rather than frames so one constant spans the whole sweep.
    static constexpr uint64_t kFlushWatermark = 768u * 1024u;
    // Overridable so the right value can be swept without a rebuild: the measured wire rate
    // moved 10.67 -> 10.04 GB/s when batches grew 1 MB -> 2.7 MB, so it is worth tuning.
    static uint64_t watermark_bytes() {
        if (const char* s = std::getenv("TT_H2H_FLUSH_KB"); s != nullptr && *s != '\0') {
            if (const long v = std::atol(s); v > 0) {
                return static_cast<uint64_t>(v) * 1024u;
            }
        }
        return kFlushWatermark;
    }
    // Breaks the credit deadlock, not just a stuck queue: holding the flush stops the peer
    // crediting, which blocks posting, and no force fires because frames are still queued.
    static constexpr auto kFlushDeadline = std::chrono::microseconds(500);

    uint64_t watermark = kFlushWatermark;
    std::vector<std::chrono::steady_clock::time_point> first_pending;
    // Bumped only AFTER a flush returns, so epoch < flush_epoch[h] means that put's bytes
    // are on the peer. Nothing else can say so once flushes are withheld.
    std::vector<uint64_t> flush_epoch;
    // The guard array is contiguous on the TARGET, but the sources are page tails at stride
    // page_bytes. Gathered here so one put arms a run; indexed by (dest_core, slot), which
    // is unique while that slot is outstanding, and an Rput reads its origin after returning.
    std::vector<uint64_t> guard_stage;

    void mark_pending(uint32_t host, uint64_t bytes) {
        if (pending[host] == 0) {
            first_pending[host] = std::chrono::steady_clock::now();
        }
        pending[host] += bytes;
    }

    // Credits only. Kept apart from pending because a guard put is small too, and holding
    // a guard stalls the peer's drain -- holding a credit does not.
    void mark_ctrl_pending(uint32_t host, uint64_t bytes) {
        if (ctrl_pending[host] == 0) {
            first_ctrl[host] = std::chrono::steady_clock::now();
        }
        ctrl_pending[host] += bytes;
    }

    // Payload and credit puts alike are flush_local only, so nothing is visible on the peer
    // until this runs. final_flush is barrier and teardown, where everything must go out.
    void flush_dirty(bool force, bool final_flush = false) {
        const auto now = std::chrono::steady_clock::now();
        for (uint32_t h = 0; h < cfg.topo.num; ++h) {
            if (pending[h] + ctrl_pending[h] == 0) {
                continue;
            }
            if (!final_flush) {
                if (pending[h] != 0) {
                    if (!force && pending[h] < watermark && now - first_pending[h] < kFlushDeadline) {
                        ++stats.flushes_held;
                        continue;
                    }
                // Credits alone: a receiver's no_supply fires every pass, so `force` must not
                // apply here. The put is at the NIC already; the flush only confirms it.
                } else if (now - first_ctrl[h] < kFlushDeadline) {
                    ++stats.flushes_held;
                    continue;
                }
            }
            ++stats.flushes;
            const uint64_t covered = pending[h] + ctrl_pending[h];
            stats.pending_sum += covered;
            stats.pending_max = std::max(stats.pending_max, covered);
            if (covered < cfg.page_bytes) {
                ++stats.flushes_tiny;
            }
            pending[h] = 0;
            ctrl_pending[h] = 0;
            if (const std::string e = win->flush(h); !e.empty()) {
                fail("h2h: " + e);
            }
            ++flush_epoch[h];
        }
    }

    uint64_t& posted_at(uint32_t core, uint32_t host) { return posted[core * kMaxCreditPeers + host]; }
    // One put per array per peer, covering every core touched since the last call. Called
    // after the drain loop, not inside consumed(), so it can span cores rather than one.
    bool publish_credits() {
        for (uint32_t h = 0; h < topo_hosts(); ++h) {
            if (h == cfg.topo.ident || !credits.pending(h)) {
                continue;
            }
            const std::string e =
                credits.publish(h, cfg.topo.ident, [&](const void* src, uint64_t bytes, uint64_t off) {
                    mark_ctrl_pending(h, bytes);
                    return win->put(src, bytes, h, off, credit_op);
                });
            if (!e.empty()) {
                fail("h2h: credit: " + e);
                return false;
            }
        }
        stats.credit_puts = credits.credit_puts();
        stats.done_puts = credits.done_puts();
        return true;
    }

    // The peer writes an absolute count into our region at credit_offset(my core, its host).
    uint64_t credit_seen(uint32_t core, uint32_t host) const {
        const auto* w = reinterpret_cast<const volatile uint64_t*>(cfg.region_base + credit_offset(core, host));
        return load_acquire(w);
    }
    std::chrono::steady_clock::time_point& put_at_of(uint32_t core, uint32_t host, uint64_t seq) {
        const size_t pair = static_cast<size_t>(core) * kMaxCreditPeers + host;
        return put_at[pair * cfg.ring_pages + static_cast<size_t>(seq % cfg.ring_pages)];
    }
    uint64_t& credit_closed_at(uint32_t core, uint32_t host) {
        return credit_closed[static_cast<size_t>(core) * kMaxCreditPeers + host];
    }

    // Turns every newly credited sequence into a sample. MUST run before this pass starts
    // any send: a stamp lives until posted laps it by ring_pages, and the gate only permits
    // that lap once the frame it would overwrite has been credited -- that is, once this
    // has already read it.
    void harvest_credits() {
        if (!cfg.collect_timing) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        for (uint32_t c = 0; c < cfg.cores; ++c) {
            for (uint32_t h = 0; h < topo_hosts(); ++h) {
                const uint64_t seen = credit_seen(c, h);
                uint64_t& closed = credit_closed_at(c, h);
                for (; closed < seen; ++closed) {
                    if (put_to_credit_ns.size() >= kMaxTimingSamples) {
                        continue;
                    }
                    const auto d = now - put_at_of(c, h, closed);
                    put_to_credit_ns.push_back(
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()));
                }
            }
        }
    }
    uint32_t topo_hosts() const { return cfg.topo.num < kMaxCreditPeers ? cfg.topo.num : kMaxCreditPeers; }

    // The same, from the array keyed on the SENDING core: what OUR core has had pulled.
    uint64_t done_seen(uint32_t core, uint32_t host) const {
        const auto* w = reinterpret_cast<const volatile uint64_t*>(cfg.region_base + done_offset(core, host));
        return load_acquire(w);
    }

    // The compact array, not the page tail. The device still writes a guard into the trailer
    // and it rides along in the payload put, but nothing reads it: a run of K slots is armed
    // by one contiguous put here, and the harvest scans K adjacent words instead of K pages.
    volatile uint64_t* rx_guard(uint32_t core, uint32_t slot) const {
        return reinterpret_cast<volatile uint64_t*>(cfg.region_base + guard_offset(core, slot));
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
    // Refused, not merely unimplemented -- see kMaxH2HHostsSupported for why it is below
    // kMaxHosts. The receiver also keeps one cursor per core, not per core and origin.
    if (cfg.topo.num > kMaxH2HHostsSupported) {
        err = "H2HSocket: " + std::to_string(cfg.topo.num) + " hosts is not supported; the RX ring serves " +
              std::to_string(kMaxH2HHostsSupported) + " (kMaxH2HHostsSupported), though the credit array indexes " +
              std::to_string(kMaxHosts);
        return nullptr;
    }
    // Not just non-zero: the trailer has to fit inside the page, and a smaller page wraps
    // below that and would ask MPI for a nearly 2^64 transfer.
    if (cfg.page_bytes != 0 && cfg.page_bytes <= kFrameTrailerBytes) {
        err = "H2HSocket: page_bytes " + std::to_string(cfg.page_bytes) + " leaves no room for the " +
              std::to_string(kFrameTrailerBytes) + " B trailer";
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
    // Refused for the same reason as the host bound above: rx_arena_offset() takes a core
    // and no chip, so one region holds one chip's arenas and chip 1 would alias chip 0.
    if (cfg.topo.chips_per_host != 1) {
        err = "H2HSocket: chips_per_host " + std::to_string(cfg.topo.chips_per_host) +
              " is not supported; the arenas are not partitioned per chip (exactly 1)";
        return nullptr;
    }
    // Checked here, not inherited from HostRegion: the chip comparisons below decode 0 for a
    // single-chip host, so a non-zero cfg.chip would reject every frame instead of routing it.
    if (cfg.chip >= cfg.topo.chips_per_host) {
        err = "H2HSocket: chip " + std::to_string(cfg.chip) + " is outside 0.." +
              std::to_string(cfg.topo.chips_per_host - 1);
        return nullptr;
    }
    if (cfg.ring_pages == 0) {
        err = "H2HSocket: ring_pages must be at least 1";
        return nullptr;
    }
    // ring_pages indexes the guard array as well as the ring, and guard_offset() strides by
    // kMaxRingSlots: past it a core's guards would land in the next core's entries.
    if (cfg.ring_pages > kMaxRingSlots) {
        err = fmt::format("H2HSocket: ring_pages {} exceeds the guard array's {} slots per core",
                          cfg.ring_pages, kMaxRingSlots);
        return nullptr;
    }
    // The same offsets are a local load in rx_guard() and a remote displacement in the
    // peer's window, so a short region is an out-of-bounds read here and an invalid RMA there.
    if (const uint64_t need = pinned_bytes_for(cfg.cores); cfg.region_bytes < need) {
        err = fmt::format(
            "H2HSocket: region is {} B but {} cores need at least {} B", cfg.region_bytes, cfg.cores, need);
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

    // Every guard above is local, but the offsets they bound are displacements into the PEER's
    // window: a peer that provisioned fewer cores or a different ring faults on the first put.
    if (!RdmaWindow::agree_value(geometry_fingerprint(cfg), err)) {
        if (err.empty()) {
            // The same text the fingerprint hashed, so the message cannot name other fields.
            err = fmt::format(
                "H2HSocket: the peer's layout differs from this host's ({}); both hosts must provision "
                "identically",
                geometry_text(cfg));
        }
        return nullptr;
    }

    im.win = RdmaWindow::create(cfg.region_base, cfg.region_bytes, cfg.topo.ident, cfg.topo.num, err);
    if (!im.win) {
        return nullptr;
    }

    const size_t per_peer = static_cast<size_t>(cfg.cores) * kMaxCreditPeers;
    im.posted.assign(per_peer, 0);
    im.credits.reset(cfg.cores, cfg.topo.num);
    // ring_pages deep: submit() refuses past it, the harvest breaks at it, and the credit
    // gate bounds payload+trailer+flight COMBINED, so each is safe at that cap.
    im.rx_pending.reset(cfg.cores, cfg.ring_pages);
    im.next_slot.assign(cfg.cores, 0);
    im.rx_seq.assign(cfg.cores, 0);
    im.tx_queue.reset(cfg.cores, cfg.ring_pages);
    im.tx_payload.reset(cfg.cores, cfg.ring_pages);
    im.tx_trailer.reset(cfg.cores, cfg.ring_pages);
    im.tx_flight.reset(cfg.cores, cfg.ring_pages);
    im.pending.assign(cfg.topo.num, 0);
    im.first_pending.assign(cfg.topo.num, std::chrono::steady_clock::time_point{});
    im.ctrl_pending.assign(cfg.topo.num, 0);
    im.first_ctrl.assign(cfg.topo.num, std::chrono::steady_clock::time_point{});
    im.flush_epoch.assign(cfg.topo.num, 0);
    im.guard_stage.assign(static_cast<size_t>(cfg.cores) * cfg.ring_pages, 0);
    // Capped at a quarter of what can be outstanding: holding more than the rings can hold
    // would stall waiting for bytes that cannot arrive until we publish the ones we have.
    const uint64_t in_flight_cap =
        static_cast<uint64_t>(cfg.cores) * cfg.ring_pages * cfg.page_bytes;
    im.watermark = std::min<uint64_t>(Impl::watermark_bytes(), std::max<uint64_t>(in_flight_cap / 4, 1));
    if (cfg.collect_timing) {
        // Sized by cfg.cores, not kProvisionedCores: a 4-core run should not carry 128.
        im.put_at.assign(per_peer * cfg.ring_pages, std::chrono::steady_clock::time_point{});
        im.credit_closed.assign(per_peer, 0);
    }
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
    if (im.tx_queue.size(task.core) >= im.cfg.ring_pages) {
        return false;
    }
    (void)im.tx_queue.push_back(task.core, task);
    ++im.tx_queued;
    return true;
}

uint32_t H2HSocket::poll(const Retire& retire, const Deliver& deliver) {
    Impl& im = *impl_;
    uint32_t progress = 0;
    if (im.broken || !deliver) {
        return 0;
    }
    im.harvest_credits();
    ++im.stats.passes;
    const uint64_t posts_before = im.stats.posts;
    bool credit_blocked = false;

    // A flush can now span several passes, so the epoch -- not the pass boundary -- is what
    // says these bytes are on the peer. test() is still required: it returns the request slot.
    im.tx_payload.for_each_live([&](uint32_t c) {
        while (!im.tx_payload.empty(c) && im.flush_epoch[im.tx_payload.front(c).host] > im.tx_payload.front(c).epoch &&
               im.win->test(im.tx_payload.front(c).op)) {
            (void)im.tx_trailer.push_back(c, im.tx_payload.front(c));
            im.tx_payload.pop_front(c);
        }
    });

    // Publish the guard, now that the payload under it is visible. The trailer is one 64 B
    // line at the slot's tail, so nothing can observe an armed guard over stale bytes.
    im.tx_trailer.for_each_live([&](uint32_t c) {
        while (!im.broken && !im.tx_trailer.empty(c)) {
            Impl::InFlight f = im.tx_trailer.front(c);
            // Gather the run's guards out of the page tails they ride in, so one put arms
            // the whole run. Staged by (dest_core, slot), which is unique while outstanding.
            const uint64_t tail = im.cfg.page_bytes - kFrameTrailerBytes;
            const size_t stage = static_cast<size_t>(f.dest_core) * im.cfg.ring_pages + f.slot;
            for (uint32_t k = 0; k < f.count; ++k) {
                const uint8_t* const src =
                    im.cfg.region_base + f.src_off + static_cast<uint64_t>(k) * im.cfg.page_bytes + tail;
                std::memcpy(&im.guard_stage[stage + k], src, sizeof(uint64_t));
            }
            if (const std::string e = im.win->put(
                    &im.guard_stage[stage],
                    static_cast<uint64_t>(f.count) * sizeof(uint64_t),
                    f.host,
                    guard_offset(f.dest_core, f.slot),
                    f.op);
                !e.empty()) {
                // fail() sets broken, which the loop condition above and the check below
                // both read -- a lambda body cannot break out of the sweep.
                im.fail("h2h: " + e);
                return;
            }
            // Re-stamped: retirement below waits on the TRAILER's flush, not the payload's.
            f.epoch = im.flush_epoch[f.host];
            im.mark_pending(f.host, static_cast<uint64_t>(f.count) * sizeof(uint64_t));
            ++im.stats.trailer_puts;
            (void)im.tx_flight.push_back(c, f);
            im.tx_trailer.pop_front(c);
            ++progress;
        }
    });
    if (im.broken) {
        return progress;
    }

    // Retire on the TRAILER's completion: the frame is not delivered until the guard is out,
    // and the D2H page behind it must outlive both puts. Front only, per core.
    im.tx_flight.for_each_live([&](uint32_t c) {
        while (!im.tx_flight.empty(c) && im.flush_epoch[im.tx_flight.front(c).host] > im.tx_flight.front(c).epoch &&
               im.win->test(im.tx_flight.front(c).op)) {
            const uint32_t n = im.tx_flight.front(c).count;
            im.tx_flight.pop_front(c);
            im.in_flight -= n;
            if (retire) {
                retire(c, n);
            }
            progress += n;
        }
    });

    // Start what the window allows, round-robin across cores. A gated core is SKIPPED,
    // never broken on: that is the whole point of the per-core queues.
    for (uint32_t k = 0; k < im.cfg.cores && im.tx_queued != 0 && im.in_flight < im.window_cap; ++k) {
        const uint32_t c = (im.rr + k) % im.cfg.cores;
        if (im.tx_queue.empty(c)) {
            continue;
        }
        const SendTask& t = im.tx_queue.front(c);
        const uint32_t host = tt_uva_target_host(t.dst, im.cfg.topo);
        const uint32_t dest_core = tt_uva_t6_core(t.dst);
        // The selector carries a chip this layout cannot express, so it is checked rather
        // than dropped: chip 1 core N would otherwise land in chip 0 core N's arena.
        const uint32_t dest_chip = tt_uva_t6_chip(t.dst, im.cfg.topo.chips_per_host);
        // A kRegionHost address also yields a host, but its selector IS the host id -- decoding
        // a core out of it would name a real ring. Only a T6 selector addresses a core.
        const bool t6 = tt_uva_selector_is_t6(t.dst);
        // dest_core indexes our own per-peer arrays as well as the target's ring, and it
        // comes out of a UVA, so it is bounded here and not trusted to be one of ours.
        if (!t6 || host == kHostNone || host >= im.cfg.topo.num || host == im.cfg.topo.ident ||
            dest_core >= im.cfg.cores || dest_chip != im.cfg.chip) {
            im.fail(fmt::format(
                "h2h: core {} addressed host {} chip {} core {}, which is not a peer of this symmetric socket",
                t.core,
                host,
                dest_chip,
                dest_core));
            im.tx_queue.pop_front(c);
            --im.tx_queued;
            break;
        }

        // Keyed on the DESTINATION core, not this one: the ring being filled belongs to the
        // target, so every sender into it must draw slots and credit from one counter.
        if (im.posted_at(dest_core, host) - im.credit_seen(dest_core, host) >= im.cfg.ring_pages) {
            credit_blocked = true;
            continue;
        }

        const uint32_t slot = static_cast<uint32_t>(im.posted_at(dest_core, host) % im.cfg.ring_pages);
        // How many queued frames form ONE contiguous transfer. Slots are adjacent at both
        // ends, so a run that shares a destination, advances by one slot and one source page,
        // and neither wraps nor outruns the credit gate, is a single put of run x page_bytes.
        const uint64_t credit_room = im.cfg.ring_pages - (im.posted_at(dest_core, host) - im.credit_seen(dest_core, host));
        uint32_t run = 1;
        while (run < im.tx_queue.size(c) && run < credit_room && slot + run < im.cfg.ring_pages) {
            const SendTask& n = im.tx_queue.at(c, run);
            if (tt_uva_target_host(n.dst, im.cfg.topo) != host || tt_uva_t6_core(n.dst) != dest_core ||
                !tt_uva_selector_is_t6(n.dst) || n.page_bytes != t.page_bytes ||
                n.page_offset != t.page_offset + static_cast<uint64_t>(run) * t.page_bytes) {
                break;
            }
            ++run;
        }

        Impl::InFlight f;
        f.host = host;
        f.dest_core = dest_core;
        f.slot = slot;
        f.count = run;
        f.src_off = t.page_offset;
        // The WHOLE page, trailer included: the guard the device armed in that trailer is
        // now inert, because the peer polls the compact array instead. The ordering hazard
        // that split this put is handled by arming the array entries in a later pass.
        if (const std::string e = im.win->put(
                im.cfg.region_base + t.page_offset,
                static_cast<uint64_t>(run) * t.page_bytes,
                host,
                rx_slot_offset(dest_core, slot, im.cfg.page_bytes, im.cfg.rx_data_offset),
                f.op);
            !e.empty()) {
            im.fail("h2h: " + e);
            break;
        }
        if (im.cfg.collect_timing) {
            // Every sequence in the run, not just the first: harvest_credits() closes them
            // one at a time, and an unstamped slot reads a default time_point (224G us).
            const auto now = std::chrono::steady_clock::now();
            for (uint32_t k = 0; k < run; ++k) {
                im.put_at_of(dest_core, host, im.posted_at(dest_core, host) + k) = now;
            }
        }
        im.posted_at(dest_core, host) += run;
        f.epoch = im.flush_epoch[host];
        im.mark_pending(host, static_cast<uint64_t>(run) * t.page_bytes);
        (void)im.tx_payload.push_back(t.core, f);
        im.in_flight += run;
        im.stats.posts += run;
        ++im.stats.payload_puts;
        for (uint32_t k = 0; k < run; ++k) {
            im.tx_queue.pop_front(c);
        }
        im.tx_queued -= run;
        progress += run;
        // A pass used to post everything queued before flush_dirty ran, so pending reached
        // 2.7 MB against a 768 KiB mark. Stop here and let the flush go out at its size.
        if (im.pending[host] >= im.watermark) {
            break;
        }
    }
    im.rr = im.cfg.cores != 0 ? (im.rr + 1) % im.cfg.cores : 0;
    // Both failure paths above land here; neither should go on to flush or harvest.
    if (im.broken) {
        return progress;
    }

    // After the starts and before the next pass's retire loop: that gap is what makes an
    // acked frame mean "in the peer's window" rather than "handed to MPI" -- see tt_uva_quiet().
    // Conditional: poll() is the spin loop, so an unconditional clock read would tax every
    // pass of a run that asked for none, and the split says which half of a pass to go after.
    const auto flush_t0 =
        im.cfg.collect_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Before the flush, so counters noted this pass ride the flush already happening.
    if (!im.publish_credits()) {
        return progress;
    }
    // No-supply only. Forcing on credit-gating too was measured to defeat the watermark:
    // holding a publish is what stops the peer crediting, so the batch never grows.
    im.flush_dirty(im.tx_queued == 0);
    if (im.cfg.collect_timing) {
        im.stats.flush_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - flush_t0).count());
    }

    // Harvest arrivals. The peer puts the trailer in a pass AFTER the payload it describes,
    // with a flush between, so an armed guard means those bytes are already visible.
    for (uint32_t c = 0; c < im.cfg.cores; ++c) {
        // The whole ring, not one slot: at depth > 1 a later arrival is otherwise invisible
        // until every poll before it has run.
        for (uint32_t n = 0; n < im.cfg.ring_pages; ++n) {
            // The guard stays armed until consumed(), so it no longer says "not yet taken".
            // This bound does: at ring_pages outstanding, next_slot cannot lap onto a live one.
            if (im.rx_pending.size(c) >= im.cfg.ring_pages) {
                break;
            }
            const uint32_t slot = im.next_slot[c];
            volatile uint64_t* const guard = im.rx_guard(c, slot);
            const uint64_t g = load_acquire(guard);
            if (!tt_uva_frame_armed(g)) {
                break;
            }
            // Armed says a frame is here; the sequence says WHICH. The guard is the sending
            // DEVICE's frame index, carried through verbatim, and this ring has one origin.
            if (const uint32_t want = static_cast<uint32_t>(im.rx_seq[c]); tt_uva_frame_seq(g) != want) {
                im.fail(fmt::format(
                    "h2h: core {} slot {} carries frame {} but this ring expects {}",
                    c,
                    slot,
                    tt_uva_frame_seq(g),
                    want));
                break;
            }
            ++im.rx_seq[c];
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
            (void)im.rx_pending.push_back(c, Impl::Delivered{t->origin, slot});
            im.next_slot[c] = (slot + 1) % im.cfg.ring_pages;
            ++progress;
        }
    }
    if (im.stats.posts == posts_before) {
        ++im.stats.starved;
        if (credit_blocked) {
            ++im.stats.starved_credit;
        } else {
            ++im.stats.starved_empty;
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
    for (; pages != 0 && !im.rx_pending.empty(core); --pages) {
        const Impl::Delivered d = im.rx_pending.front(core);
        im.rx_pending.pop_front(core);

        // The H2D leg has reported this page drained, so the device is done reading it.
        // Still before the credit: a credit lets the peer re-arm the slot.
        store_release(im.rx_guard(core, d.slot), 0);

        const uint32_t host = tt_uva_t6_selector_host(d.origin, im.cfg.topo.chips_per_host);
        const uint32_t src_core = tt_uva_t6_selector_core(d.origin);
        const uint32_t src_chip = tt_uva_t6_selector_chip(d.origin, im.cfg.topo.chips_per_host);
        // cfg.cores, not kProvisionedCores: src_core indexes the done array, sized by it.
        if (host >= im.cfg.topo.num || host == im.cfg.topo.ident || src_core >= im.cfg.cores ||
            src_chip != im.cfg.chip) {
            im.fail(
                "h2h: a delivered frame named origin selector " + std::to_string(d.origin) +
                ", which is not a peer core");
            return;
        }
        // Counters only. publish_credits() puts them once per pass, across every core at
        // once -- keyed on THIS core for the gate, and on the SENDER for tt_uva_sync().
        im.credits.note_credit(host, core);
        im.credits.note_done(host, src_core);
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
const std::vector<uint64_t>& H2HSocket::put_to_credit_ns() const { return impl_->put_to_credit_ns; }

const H2HSocket::PassStats& H2HSocket::pass_stats() const { return impl_->stats; }

// Publishes everything consumed() has noted. The chain calls this after its drain loop:
// deferring one pass is what lets a single put span every core that freed a slot.
bool H2HSocket::publish_credits() { return impl_->publish_credits(); }

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
    impl_->publish_credits();
    impl_->flush_dirty(true, /*final_flush=*/true);
    return impl_->win->barrier();
}
bool H2HSocket::failed() const { return impl_->broken; }
std::string H2HSocket::first_error() const { return impl_->err; }

std::string H2HSocket::describe() const {
    return fmt::format(
        "h2h: {}, window {} frame(s), ring depth {}", impl_->win->describe(), impl_->window_cap, impl_->cfg.ring_pages);
}

}  // namespace tt::tt_metal::experimental
