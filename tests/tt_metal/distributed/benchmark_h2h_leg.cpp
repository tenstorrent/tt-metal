// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg alone: RdmaWindow directly, no device and no H2HSocket.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/distributed/host_rdma_window.hpp"
#include "tt_metal/distributed/host_uva_frame.hpp"
#include "leg_benchmark_common.hpp"

using namespace tt::tt_metal::experimental;
namespace mh = tt::tt_metal::distributed::multihost;
using namespace leg_bench;

namespace {

// The credit gets its own page so it never shares a line with a slot.
constexpr uint64_t kCreditOff = 0;
constexpr uint64_t kSlotsOff = 4096;

// Both ranks build this identically, which is what keeps them on the same case list.
const std::vector<int64_t> kPageSizes = {4096, 16384, 65536, 262144};
const std::vector<int64_t> kWindowFrames = {8, 32};
const std::vector<int64_t> kIterations = {2000, 20000, 200000};
const std::vector<int64_t> kWarmupPct = {0, 10, 25};
const std::vector<int64_t> kPingPongIters = {2000};
// Both, as the other legs sweep it. Note frame_is_good() only checks the trailer length
// here, so this proves delivery and framing rather than payload integrity.
const std::vector<int64_t> kVerify = {0, 1};

// Pre-registered so a skipped case keeps the CSV shape.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["frames"] = 0;
    state.counters["bad_frames"] = 0;
    // How well the loop keeps the window occupied: a flush costs the same whether it covers
    // one frame or thirty-two, so posts_per_flush is what sets throughput.
    state.counters["posts_per_flush"] = 0;
    state.counters["starved_pass_pct"] = 0;
    state.counters["flush_pct"] = 0;
    set_latency_counters(state, LatencySummary{}, 0);
    set_latency_counters(state, LatencySummary{}, 0, "oneway_");
}

// The periodic flush is the progress turn some transports need to land an inbound put.
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

class H2HLegFixture : public benchmark::Fixture {
public:
    // Collective: a rank that skips alone strands its peer inside the next collective.
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<uint32_t>(state.range(0));
        window_ = static_cast<uint32_t>(state.range(1));
        iters_ = static_cast<uint32_t>(state.range(2));
        warmup_pct_ = static_cast<uint32_t>(state.range(3));
        pp_iters_ = static_cast<uint32_t>(state.range(4));
        verify_ = state.range(5) != 0;
        page_ = tt_uva_frame_page_size(payload_bytes_);

        const mh::ContextPtr& world = mh::DistributedContext::get_current_world();
        rank_ = static_cast<uint32_t>(*world->rank());
        ranks_ = static_cast<uint32_t>(*world->size());
        peer_ = 1 - rank_;

        src_off_ = kSlotsOff + static_cast<uint64_t>(window_) * page_;
        bytes_ = src_off_ + static_cast<uint64_t>(window_) * page_;

        // posix_memalign, not std::aligned_alloc: the latter is absent from libc++.
        void* raw = nullptr;
        if (::posix_memalign(&raw, 4096, (bytes_ + 4095) & ~uint64_t{4095}) != 0) {
            raw = nullptr;
        }
        // Owned here so the allocation cannot outlive the fixture on an early return; base_
        // stays a raw pointer because every offset below is arithmetic on it.
        owned_.reset(static_cast<uint8_t*>(raw));
        base_ = owned_.get();
        std::string err;
        if (base_ == nullptr) {
            err = "could not allocate " + std::to_string(bytes_ >> 20) + " MiB";
        }
        // Agreed BEFORE create(): MPI_Win_create is collective, so a rank that allocated must
        // not enter it while its peer has already fallen through to the agreement.
        if (!RdmaWindow::agree(base_ != nullptr, err)) {
            fail(state, "rank " + std::to_string(rank_) + ": allocation failed: " + err);
            return;
        }
        std::memset(base_, 0, bytes_);
        win_ = RdmaWindow::create(base_, bytes_, rank_, ranks_, err);
        if (!RdmaWindow::agree(win_ != nullptr, err)) {
            fail(state, "rank " + std::to_string(rank_) + ": window bringup failed: " + err);
            return;
        }
        if (const std::string e = win_->barrier(); !e.empty()) {
            fail(state, "barrier: " + e);
            return;
        }
        fill_origins();
    }

    // Collective too, and the region must outlive the window it registered.
    void TearDown(benchmark::State& state) override {
        (void)state;
        if (win_) {
            if (const std::string e = win_->barrier(); !e.empty()) {
                std::fprintf(stderr, "barrier: %s\n", e.c_str());
            }
            win_.reset();
        }
        // After win_: the window names these pages, so they must outlive it.
        owned_.reset();
        base_ = nullptr;
    }

protected:
    // Filled once: only word 0 and the trailer move per frame.
    void fill_origins() const {
        if (!verify_) {
            return;
        }
        const uint32_t want = pattern_word(rank_);
        for (uint32_t s = 0; s < window_; ++s) {
            uint32_t* const w = reinterpret_cast<uint32_t*>(base_ + src_off_ + static_cast<uint64_t>(s) * page_);
            for (uint32_t k = 1; k < payload_bytes_ / sizeof(uint32_t); ++k) {
                w[k] = want;
            }
        }
    }

    // Trailer last in the page: an armed guard means the payload ahead of it landed.
    void arm_frame(uint8_t* src, uint32_t stamp) const {
        if (verify_) {
            *reinterpret_cast<uint32_t*>(src) = stamp;
        }
        FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(src + payload_bytes_);
        std::memset(t, 0, sizeof(*t));
        t->length = payload_bytes_;
        // The same stamp the payload carries: this benchmark drives the window itself, so
        // the frame index is in hand at post time -- see the arm_frame(src, posted) call.
        t->guard = tt_uva_frame_guard(kFrameVersion, stamp);
    }

    // The stamp is the sender's frame index, so a stale slot is caught, not just a corrupt one.
    bool frame_is_good(const uint8_t* slot, uint64_t expect_stamp) const {
        if (reinterpret_cast<const FrameTrailer*>(slot + payload_bytes_)->length != payload_bytes_) {
            return false;
        }
        if (!verify_) {
            return true;
        }
        const uint32_t* const w = reinterpret_cast<const uint32_t*>(slot);
        if (w[0] != static_cast<uint32_t>(expect_stamp)) {
            return false;
        }
        const uint32_t want = pattern_word(peer_);
        const uint32_t words = payload_bytes_ / sizeof(uint32_t);
        for (const uint32_t k : {1u, words / 2u, words - 1u}) {
            if (k != 0 && k < words && w[k] != want) {
                return false;
            }
        }
        return true;
    }

    // Equal payload each way, so halving the round trip is defensible.
    bool ping_pong(std::vector<double>& rt, std::string& err) {
        uint8_t* const src = base_ + src_off_;
        volatile uint64_t* const guard = reinterpret_cast<volatile uint64_t*>(base_ + kSlotsOff + payload_bytes_);
        const uint32_t warmup = pp_iters_ * warmup_pct_ / 100;
        rt.reserve(pp_iters_ - warmup);
        const bool initiator = rank_ == 0;
        // Trailer is constant, so it is built once rather than inside the timed loop.
        arm_frame(src, 0);

        for (uint32_t i = 0; i < pp_iters_; ++i) {
            if (verify_) {
                *reinterpret_cast<uint32_t*>(src) = i;
            }
            if (initiator) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!send_frame(*win_, src, page_, peer_, err) || !take_frame(*win_, guard, peer_)) {
                    err = err.empty() ? "ping-pong: no reply from the peer" : err;
                    return false;
                }
                if (i >= warmup) {
                    rt.push_back(us_since(t0));
                }
            } else {
                if (!take_frame(*win_, guard, peer_)) {
                    err = "ping-pong: no frame from the peer";
                    return false;
                }
                if (!send_frame(*win_, src, page_, peer_, err)) {
                    return false;
                }
            }
        }
        return true;
    }

    std::unique_ptr<RdmaWindow> win_;
    std::unique_ptr<uint8_t, void (*)(void*)> owned_{nullptr, &std::free};
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t src_off_ = 0;
    uint32_t payload_bytes_ = 0;
    uint32_t page_ = 0;
    uint32_t window_ = 0;
    uint32_t iters_ = 0;
    uint32_t warmup_pct_ = 0;
    uint32_t pp_iters_ = 0;
    uint32_t rank_ = 0;
    uint32_t ranks_ = 0;
    uint32_t peer_ = 0;
    bool verify_ = false;
};

BENCHMARK_DEFINE_F(H2HLegFixture, Bandwidth)(benchmark::State& state) {
    init_counters(state);

    for ([[maybe_unused]] auto _ : state) {
        const uint32_t warmup = iters_ * warmup_pct_ / 100;
        const uint32_t total = iters_;
        const uint32_t measured = total - warmup;
        std::string err;
        bool ok = true;
        double window_us = 0.0;
        uint64_t bad = 0;
        std::vector<double> rt_us;
        uint64_t passes = 0;    // outer iterations, i.e. flushes
        uint64_t starved = 0;   // passes that posted nothing: the window was gated shut
        double flush_us = 0.0;

        if (rank_ == 0) {
            // Rank 0 times: the credit closes each frame on the clock that opened it.
            std::vector<RdmaWindow::Op> ops(window_);
            std::vector<std::chrono::steady_clock::time_point> posted_at(window_);
            rt_us.reserve(measured);

            volatile uint64_t* const credit = reinterpret_cast<volatile uint64_t*>(base_ + kCreditOff);
            auto t0 = std::chrono::steady_clock::now();
            uint32_t posted = 0;
            uint32_t credited = 0;
            auto deadline = std::chrono::steady_clock::now() + kStall;

            while (credited < total && ok) {
                ++passes;
                const uint32_t posted_before = posted;
                while (posted < total && posted - credited < window_) {
                    const uint32_t s = posted % window_;
                    // A credit says the TARGET has the frame, not that this Rput retired.
                    if (ops[s].valid() && !win_->test(ops[s])) {
                        break;
                    }
                    uint8_t* const src = base_ + src_off_ + static_cast<uint64_t>(s) * page_;
                    arm_frame(src, posted);

                    if (const std::string e =
                            win_->put(src, page_, peer_, kSlotsOff + static_cast<uint64_t>(s) * page_, ops[s]);
                        !e.empty()) {
                        err = "rank 0: " + e;
                        ok = false;
                        break;
                    }
                    posted_at[s] = std::chrono::steady_clock::now();
                    ++posted;
                }

                // Local completion frees the origin buffers; it does NOT land them.
                for (uint32_t s = 0; s < window_; ++s) {
                    if (ops[s].valid()) {
                        (void)win_->test(ops[s]);
                    }
                }
                // The peer's own flush completes ITS outbound credits, never our puts, so
                // without this the credit we then wait on is for frames that never arrived.
                if (posted == posted_before) {
                    ++starved;
                }
                const auto tf = std::chrono::steady_clock::now();
                if (const std::string e = win_->flush(peer_); !e.empty()) {
                    err = "rank 0: " + e;
                    ok = false;
                    break;
                }
                flush_us += us_since(tf);

                const uint64_t seen = __atomic_load_n(credit, __ATOMIC_ACQUIRE);
                while (credited < seen && credited < total) {
                    const uint32_t s = credited % window_;
                    if (credited == warmup) {
                        t0 = std::chrono::steady_clock::now();
                    }
                    if (credited >= warmup) {
                        rt_us.push_back(us_since(posted_at[s]));
                    }
                    ++credited;
                    deadline = std::chrono::steady_clock::now() + kStall;
                }
                if (std::chrono::steady_clock::now() > deadline) {
                    err = "rank 0: stalled at " + std::to_string(credited) + " of " + std::to_string(total) +
                          " credited";
                    ok = false;
                }
            }
            window_us = us_since(t0);
        } else {
            // One credit per frame: the granularity the round trip is measured at.
            uint64_t consumed = 0;
            uint32_t idle = 0;
            auto deadline = std::chrono::steady_clock::now() + kStall;

            while (consumed < total && ok) {
                const uint32_t s = static_cast<uint32_t>(consumed % window_);
                uint8_t* const slot = base_ + kSlotsOff + static_cast<uint64_t>(s) * page_;
                volatile uint64_t* const guard = reinterpret_cast<volatile uint64_t*>(slot + payload_bytes_);

                if (tt_uva_frame_armed(__atomic_load_n(guard, __ATOMIC_ACQUIRE))) {
                    if (!frame_is_good(slot, consumed)) {
                        ++bad;
                    }
                    // Zeroed before the credit, never after: a late zero erases a fresh frame.
                    __atomic_store_n(guard, uint64_t{0}, __ATOMIC_RELEASE);
                    ++consumed;
                    if (const std::string e = win_->put_word(consumed, peer_, kCreditOff); !e.empty()) {
                        err = "rank 1: " + e;
                        ok = false;
                    }
                    idle = 0;
                    deadline = std::chrono::steady_clock::now() + kStall;
                } else if (std::chrono::steady_clock::now() > deadline) {
                    err = "rank 1: stalled at " + std::to_string(consumed) + " of " + std::to_string(total) +
                          " frames";
                    ok = false;
                } else if (++idle >= 1024) {
                    // Some transports only move an inbound put while this rank is inside MPI.
                    (void)win_->flush(peer_);
                    idle = 0;
                }
            }
            if (bad != 0) {
                err = "rank 1: " + std::to_string(bad) + " frames failed the frame check";
                ok = false;
            }
        }

        // Both ranks leave with the same verdict, so ping-pong is entered by both or neither.
        if (!RdmaWindow::agree(ok, err)) {
            fail(state, err.empty() ? "h2h streaming phase failed" : err);
            return;
        }

        std::vector<double> pp_us;
        if (pp_iters_ != 0) {
            ok = ping_pong(pp_us, err);
            if (!RdmaWindow::agree(ok, err)) {
                fail(state, err.empty() ? "h2h ping-pong phase failed" : err);
                return;
            }
        }

        // Rank 1 measured nothing, and main() gives it a reporter that prints nothing.
        if (rank_ != 0) {
            continue;
        }
        state.counters["frames"] = static_cast<double>(total);
        state.counters["bad_frames"] = static_cast<double>(bad);
        if (passes != 0) {
            state.counters["posts_per_flush"] = static_cast<double>(total) / static_cast<double>(passes);
            state.counters["starved_pass_pct"] = 100.0 * static_cast<double>(starved) / static_cast<double>(passes);
        }
        if (window_us > 0.0) {
            state.counters["flush_pct"] = 100.0 * flush_us / window_us;
        }
        if (window_us > 0.0) {
            const double gb = static_cast<double>(measured) * payload_bytes_ / 1e9;
            state.counters["throughput_gbps"] = gb / (window_us / 1e6);
        }
        // put->credit is what a slot costs before reuse; one-way is half the ping-pong rtt.
        set_latency_counters(state, summarize_latency_us(rt_us, 0.0), rt_us.size());
        for (double& v : pp_us) {
            v /= 2.0;
        }
        set_latency_counters(state, summarize_latency_us(pp_us, 0.0), pp_us.size(), "oneway_");
    }
}

BENCHMARK_REGISTER_F(H2HLegFixture, Bandwidth)
    // Named explicitly: analyze_hd_sockets.py keys on the name up to the first slash.
    ->Name("BM_H2HLegBandwidth")
    ->ArgsProduct({
        kPageSizes,      // page_size
        kWindowFrames,   // window
        kIterations,     // iters
        kWarmupPct,      // warmup_pct
        kPingPongIters,  // pp_iters
        kVerify,         // verify
    })
    ->ArgNames({"page_size", "window", "iters", "warmup_pct", "pp_iters", "verify"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

}  // namespace

int main(int argc, char** argv) {
    // Before Initialize: the context takes MPI's own argv entries first.
    mh::DistributedContext::create(argc, argv);
    const mh::ContextPtr& world = mh::DistributedContext::get_current_world();
    const uint32_t rank = static_cast<uint32_t>(*world->rank());
    if (*world->size() != 2) {
        if (rank == 0) {
            std::fprintf(stderr, "error: this leg needs exactly 2 ranks; launch with `mpirun -n 2`\n");
        }
        return 2;
    }

    // A filter or repetition count given to one rank only will deadlock the pair.
    benchmark::Initialize(&argc, argv);
    if (rank == 0) {
        benchmark::RunSpecifiedBenchmarks();
    } else {
        NullReporter quiet;
        benchmark::RunSpecifiedBenchmarks(&quiet);
    }
    benchmark::Shutdown();
    return g_run_failed ? 1 : 0;
}
