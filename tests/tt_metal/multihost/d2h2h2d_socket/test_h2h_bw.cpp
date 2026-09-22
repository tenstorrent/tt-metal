// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg alone: RdmaWindow directly, no device and no H2HSocket, so what is
// measured is the transport rather than the scheduler in front of it.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/distributed/host_rdma_window.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>

using namespace tt::tt_metal::experimental;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

// The peer's running consumed count, alone on its own page so a credit never shares a line
// with a slot. Slots follow, then the origin buffers, which only this rank reads.
constexpr uint64_t kCreditOff = 0;
constexpr uint64_t kSlotsOff = 4096;

// A stalled rank must fail rather than spin: the closing agree() is collective, so one side
// hanging strands the other in it.
constexpr auto kStall = std::chrono::seconds(30);

struct Options {
    uint32_t payload = 16384;
    uint32_t window = 8;
    uint32_t iters = 20000;
    uint32_t warmup_pct = 10;
    uint32_t pp_iters = 2000;
};

// strtoul, not stoul: parse() runs with MPI already up, where an uncaught exception aborts
// this rank and strands its peer in whatever collective it had reached.
bool parse_u32(const char* text, uint32_t& out) {
    errno = 0;
    char* end = nullptr;
    const unsigned long v = std::strtoul(text, &end, 10);
    if (end == text || errno == ERANGE) {
        return false;
    }
    uint64_t mult = 1;
    if (*end != '\0' && end[1] == '\0') {
        switch (*end) {
            case 'K': mult = 1024ull; break;
            case 'M': mult = 1ull << 20; break;
            default: return false;
        }
    } else if (*end != '\0') {
        return false;
    }
    const uint64_t scaled = static_cast<uint64_t>(v) * mult;
    if (scaled == 0 || scaled > UINT32_MAX) {
        return false;
    }
    out = static_cast<uint32_t>(scaled);
    return true;
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = i + 1 < argc;
        if (a == "--payload" && has_next && parse_u32(argv[++i], o.payload)) {
            continue;
        }
        if (a == "--window" && has_next && parse_u32(argv[++i], o.window)) {
            continue;
        }
        if (a == "--iters" && has_next && parse_u32(argv[++i], o.iters)) {
            continue;
        }
        if (a == "--warmup-pct" && has_next && parse_u32(argv[++i], o.warmup_pct)) {
            continue;
        }
        if (a == "--pp-iters" && has_next && parse_u32(argv[++i], o.pp_iters)) {
            continue;
        }
        std::cerr << "usage: " << argv[0]
                  << " [--payload B] [--window frames] [--iters N] [--warmup-pct P] [--pp-iters N]\n";
        return false;
    }
    return o.warmup_pct < 100;
}

uint64_t ns_since(std::chrono::steady_clock::time_point t) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count());
}

// Nearest-rank, on an already-sorted vector. Only ever called with a non-empty one.
uint64_t pct(const std::vector<uint64_t>& v, double p) {
    const size_t i = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
    return v[i];
}

// Spins until the peer's frame lands in this slot, then disarms it. The periodic flush is
// the progress turn a target needs on transports that do not land a put unaided.
bool take_frame(RdmaWindow& win, volatile uint64_t* guard, uint32_t peer) {
    const auto deadline = std::chrono::steady_clock::now() + kStall;
    for (uint32_t idle = 0;; ++idle) {
        if (tt_uva_frame_armed(__atomic_load_n(guard, __ATOMIC_ACQUIRE))) {
            __atomic_store_n(guard, uint64_t{0}, __ATOMIC_RELEASE);
            return true;
        }
        if (idle >= 1024) {
            idle = 0;
            (void)win.flush(peer);
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
        }
    }
}

// Blocks until the Rput retires, so the single origin buffer is reusable next iteration.
bool send_frame(RdmaWindow& win, const uint8_t* src, uint32_t page, uint32_t peer, std::string& err) {
    RdmaWindow::Op op;
    if (err = win.put(src, page, peer, kSlotsOff, op); !err.empty()) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + kStall;
    while (!win.test(op)) {
        if (std::chrono::steady_clock::now() > deadline) {
            err = "ping-pong: an Rput never retired";
            return false;
        }
    }
    return true;
}

// Equal payload each way, so halving the round trip is defensible -- unlike the credit path,
// where an 8 B credit answers a full frame. Slot 0 only: the point is latency, not depth.
bool ping_pong(
    RdmaWindow& win,
    uint8_t* base,
    uint64_t src_off,
    const Options& o,
    uint32_t page,
    uint32_t peer,
    bool initiator,
    std::vector<uint64_t>& rt,
    std::string& err) {
    uint8_t* const src = base + src_off;
    volatile uint64_t* const guard = reinterpret_cast<volatile uint64_t*>(base + kSlotsOff + o.payload);

    // Constant across iterations, so it is built once rather than inside the timed loop.
    FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(src + o.payload);
    std::memset(t, 0, sizeof(*t));
    t->length = o.payload;
    t->guard = tt_uva_frame_guard(kFrameVersion);

    const uint32_t warmup = o.pp_iters * o.warmup_pct / 100;
    rt.reserve(o.pp_iters - warmup);

    for (uint32_t i = 0; i < o.pp_iters; ++i) {
        if (initiator) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!send_frame(win, src, page, peer, err) || !take_frame(win, guard, peer)) {
                err = err.empty() ? "ping-pong: no reply from the peer" : err;
                return false;
            }
            if (i >= warmup) {
                rt.push_back(ns_since(t0));
            }
        } else {
            if (!take_frame(win, guard, peer)) {
                err = "ping-pong: no frame from the peer";
                return false;
            }
            if (!send_frame(win, src, page, peer, err)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    mh::DistributedContext::create(argc, argv);
    Options o;
    if (!parse(argc, argv, o)) {
        return 2;
    }

    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    const uint32_t rank = static_cast<uint32_t>(*world->rank());
    const uint32_t ranks = static_cast<uint32_t>(*world->size());
    if (ranks != 2) {
        std::cerr << "error: this test needs exactly 2 ranks; launch with `mpirun -n 2`\n";
        return 2;
    }
    const uint32_t peer = 1 - rank;

    const uint32_t page = tt_uva_frame_page_size(o.payload);
    const uint64_t src_off = kSlotsOff + static_cast<uint64_t>(o.window) * page;
    const uint64_t bytes = src_off + static_cast<uint64_t>(o.window) * page;

    // Page-aligned so a slot never straddles one; both ranks size it identically.
    // posix_memalign, not std::aligned_alloc: the latter is absent from libc++.
    void* raw = nullptr;
    if (::posix_memalign(&raw, 4096, (bytes + 4095) & ~uint64_t{4095}) != 0) {
        raw = nullptr;
    }
    uint8_t* const base = static_cast<uint8_t*>(raw);
    if (base == nullptr) {
        std::cerr << "rank " << rank << ": could not allocate " << (bytes >> 20) << " MiB\n";
        return 1;
    }
    std::memset(base, 0, bytes);

    std::string err;
    std::unique_ptr<RdmaWindow> win = RdmaWindow::create(base, bytes, rank, ranks, err);
    if (!RdmaWindow::agree(win != nullptr, err)) {
        std::cerr << "rank " << rank << ": window bringup failed: " << err << "\n";
        win.reset();  // before the region it registered
        std::free(base);
        return 1;
    }

    const uint32_t warmup = o.iters * o.warmup_pct / 100;
    const uint32_t total = o.iters;
    const uint32_t measured = total - warmup;
    if (const std::string e = win->barrier(); !e.empty()) {
        std::cerr << "barrier: " << e << "\n";
    }

    bool ok = true;
    uint64_t elapsed_ns = 0;
    std::vector<uint64_t> rt_ns;

    if (rank == 0) {
        // Rank 0 sends and is the only side that times: the credit closes each frame on the
        // clock that opened it, so nothing here needs the two clocks related.
        std::vector<RdmaWindow::Op> ops(o.window);
        std::vector<std::chrono::steady_clock::time_point> posted_at(o.window);
        rt_ns.reserve(measured);

        volatile uint64_t* const credit = reinterpret_cast<volatile uint64_t*>(base + kCreditOff);
        auto t0 = std::chrono::steady_clock::now();
        uint32_t posted = 0;
        uint32_t credited = 0;
        // Re-armed on forward motion, so a long run never trips it and a stall always does.
        auto deadline = std::chrono::steady_clock::now() + kStall;

        while (credited < total && ok) {
            while (posted < total && posted - credited < o.window) {
                const uint32_t s = posted % o.window;
                // A credit says the TARGET has the frame, not that this Rput retired. Reusing
                // the origin or the Op before it does races the send and leaks the request.
                if (ops[s].valid() && !win->test(ops[s])) {
                    break;
                }
                uint8_t* const src = base + src_off + static_cast<uint64_t>(s) * page;
                // Trailer last in the page, so an armed guard on the peer means the payload
                // ahead of it landed -- the same trailing-flag rule H2HSocket relies on.
                FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(src + o.payload);
                std::memset(t, 0, sizeof(*t));
                t->length = o.payload;
                t->guard = tt_uva_frame_guard(kFrameVersion);

                if (const std::string e =
                        win->put(src, page, peer, kSlotsOff + static_cast<uint64_t>(s) * page, ops[s]);
                    !e.empty()) {
                    std::cerr << "rank 0: " << e << "\n";
                    ok = false;
                    break;
                }
                posted_at[s] = std::chrono::steady_clock::now();
                ++posted;
            }

            // Local completion only -- it frees the origin buffer and turns the progress
            // engine. Remote arrival is what the credit reports, so nothing waits here.
            for (uint32_t s = 0; s < o.window; ++s) {
                if (ops[s].valid()) {
                    (void)win->test(ops[s]);
                }
            }

            const uint64_t seen = __atomic_load_n(credit, __ATOMIC_ACQUIRE);
            while (credited < seen && credited < total) {
                const uint32_t s = credited % o.window;
                if (credited == warmup) {
                    t0 = std::chrono::steady_clock::now();
                }
                if (credited >= warmup) {
                    rt_ns.push_back(ns_since(posted_at[s]));
                }
                ++credited;
                deadline = std::chrono::steady_clock::now() + kStall;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "rank 0: stalled at " << credited << " of " << total << " credited\n";
                ok = false;
            }
        }
        elapsed_ns = ns_since(t0);
    } else {
        // Rank 1 walks its slots in order and credits one frame at a time, which is the
        // granularity the round trip is being measured at.
        uint64_t consumed = 0;
        uint64_t bad = 0;
        uint32_t idle = 0;
        auto deadline = std::chrono::steady_clock::now() + kStall;

        while (consumed < total && ok) {
            const uint32_t s = static_cast<uint32_t>(consumed % o.window);
            uint8_t* const slot = base + kSlotsOff + static_cast<uint64_t>(s) * page;
            volatile uint64_t* const guard = reinterpret_cast<volatile uint64_t*>(slot + o.payload);

            if (tt_uva_frame_armed(__atomic_load_n(guard, __ATOMIC_ACQUIRE))) {
                if (reinterpret_cast<const FrameTrailer*>(slot + o.payload)->length != o.payload) {
                    ++bad;
                }
                // Zeroed before the credit, never after: the credit lets the peer re-arm this
                // slot, and a late zero would erase a fresh frame.
                __atomic_store_n(guard, uint64_t{0}, __ATOMIC_RELEASE);
                ++consumed;
                if (const std::string e = win->put_word(consumed, peer, kCreditOff); !e.empty()) {
                    std::cerr << "rank 1: " << e << "\n";
                    ok = false;
                }
                idle = 0;
                deadline = std::chrono::steady_clock::now() + kStall;
            } else if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "rank 1: stalled at " << consumed << " of " << total << " frames\n";
                ok = false;
            } else if (++idle >= 1024) {
                // Passive target, but some transports only move an inbound put while this
                // rank is inside MPI. Costs nothing when there is nothing outstanding.
                (void)win->flush(peer);
                idle = 0;
            }
        }
        if (bad != 0) {
            std::cerr << "rank 1: " << bad << " frames carried the wrong length\n";
            ok = false;
        }
    }  // close Rank 1 conditional

    if (!RdmaWindow::agree(ok, err)) {
        std::cerr << "rank " << rank << ": " << err << "\n";
        if (rank == 0) {
            std::cout << "FAIL\n";
        }
        win.reset();  // before the region it registered
        std::free(base);
        return 1;
    }

    // Second phase. Both ranks enter it or neither does -- the agree above is what makes
    // that true, and a one-sided entry would hang the other in take_frame().
    std::vector<uint64_t> pp_ns;
    if (o.pp_iters != 0) {
        ok = ping_pong(*win, base, src_off, o, page, peer, rank == 0, pp_ns, err);
        if (!ok) {
            std::cerr << "rank " << rank << ": " << err << "\n";
        }
        if (!RdmaWindow::agree(ok, err)) {
            std::cerr << "rank " << rank << ": " << err << "\n";
            if (rank == 0) {
                std::cout << "FAIL\n";
            }
            win.reset();  // before the region it registered
            std::free(base);
            return 1;
        }
    }

    if (rank == 0) {
        std::sort(rt_ns.begin(), rt_ns.end());
        std::sort(pp_ns.begin(), pp_ns.end());
        const double secs = static_cast<double>(elapsed_ns) / 1e9;
        const double gb = static_cast<double>(measured) * o.payload / 1e9;

        std::cout << std::fixed << std::setprecision(2);
        if (secs > 0.0) {
            std::cout << "bandwidth " << (gb / secs) << " GB/s\n";
        }
        // Both are latencies: one-way is the symmetric half-rtt, put->credit is what a slot
        // actually costs before it can be reused.
        std::cout << "latency";
        if (!pp_ns.empty()) {
            std::cout << " " << (pct(pp_ns, 0.50) / 2e3) << " us one-way";
        }
        if (!rt_ns.empty()) {
            std::cout << (pp_ns.empty() ? " " : ", ") << (pct(rt_ns, 0.50) / 1e3) << " us put->credit";
        }
        std::cout << "\nPASS\n";
    }

    // Before the window dies: it is collective, and the region must outlive it.
    if (const std::string e = win->barrier(); !e.empty()) {
        std::cerr << "barrier: " << e << "\n";
    }
    win.reset();
    std::free(base);
    return 0;
}
