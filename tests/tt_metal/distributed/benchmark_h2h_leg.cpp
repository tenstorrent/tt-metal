// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg alone, under google-benchmark: RdmaWindow directly, no device and no
// H2HSocket, so what is measured is the transport rather than the scheduler in front of it.
#include <algorithm>
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

// The swept axes. Both ranks build this list identically from the same binary, which is
// what keeps them running the same cases in the same order -- see the note in main().
const std::vector<int64_t> kPageSizes = {4096, 16384, 65536, 262144};
const std::vector<int64_t> kWindowFrames = {8, 32};
const std::vector<int64_t> kIterations = {20000};
const std::vector<int64_t> kWarmupPct = {10};
const std::vector<int64_t> kPingPongIters = {2000};
const std::vector<int64_t> kVerify = {0};

// SkipWithError marks the report but not the exit status, and analyze_hd_sockets.py drops
// errored rows -- so a failed run would read as no data. main() returns this instead.
bool g_run_failed = false;

void fail(benchmark::State& state, const std::string& why) {
    g_run_failed = true;
    state.SkipWithError(why);
}

// ---------------------------------------------------------------------------------------
// Metrics. Copied from benchmark_hd_sockets.cpp:159-168, :194-215 and :518-528, which is not
// to be modified; the symbols there are file-local.
// ---------------------------------------------------------------------------------------

struct LatencySummary {
    double avg_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double avg_cycles = 0.0;
    uint64_t min_cycles = 0;
    uint64_t max_cycles = 0;
};

// cycles_per_us is 0 on this leg: there is no device in the path, so the cycle columns the
// CSV header carries have nothing to report and come out zero rather than invented.
LatencySummary summarize_latency_us(const std::vector<double>& us_values, double cycles_per_us) {
    if (us_values.empty()) {
        return {};
    }
    auto sorted = us_values;
    std::sort(sorted.begin(), sorted.end());
    double avg_us = 0.0;
    for (auto v : us_values) {
        avg_us += v;
    }
    avg_us /= static_cast<double>(us_values.size());

    return {
        .avg_us = avg_us,
        .min_us = sorted.front(),
        .max_us = sorted.back(),
        .p50_us = sorted[sorted.size() / 2],
        .p99_us = sorted[(sorted.size() * 99) / 100],
        .avg_cycles = avg_us * cycles_per_us,
        .min_cycles = static_cast<uint64_t>(sorted.front() * cycles_per_us),
        .max_cycles = static_cast<uint64_t>(sorted.back() * cycles_per_us),
    };
}

// `prefix` is not in the original: two distributions are reported here, and the unprefixed
// names are the ones analyze_hd_sockets.py looks for, so only one set can go unprefixed.
void set_latency_counters(
    benchmark::State& state, const LatencySummary& s, uint64_t num_iterations, const std::string& prefix = "") {
    state.counters[prefix + "num_iterations"] = static_cast<double>(num_iterations);
    state.counters[prefix + "avg_us"] = s.avg_us;
    state.counters[prefix + "min_us"] = s.min_us;
    state.counters[prefix + "max_us"] = s.max_us;
    state.counters[prefix + "p50_us"] = s.p50_us;
    state.counters[prefix + "p99_us"] = s.p99_us;
    state.counters[prefix + "avg_cycles"] = s.avg_cycles;
    state.counters[prefix + "min_cycles"] = static_cast<double>(s.min_cycles);
    state.counters[prefix + "max_cycles"] = static_cast<double>(s.max_cycles);
}

// Pre-registered so the CSV header carries every column even when a case is skipped, which
// keeps the shape stable across a sweep.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["frames"] = 0;
    state.counters["bad_frames"] = 0;
    set_latency_counters(state, LatencySummary{}, 0);
    set_latency_counters(state, LatencySummary{}, 0, "oneway_");
}

double us_since(std::chrono::steady_clock::time_point t) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
    return static_cast<double>(ns) / 1e3;
}

// What the receiving rank expects in every payload word but word 0, derived from the
// sender's rank so a frame from the wrong origin is distinguishable.
uint32_t pattern_word(uint32_t rank) {
    const uint32_t b = 0x40u + (rank & 0x1Fu);
    return b | (b << 8) | (b << 16) | (b << 24);
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

class H2HLegFixture : public benchmark::Fixture {
public:
    // Collective throughout: every refusal below goes through RdmaWindow::agree, because a
    // rank that skips alone strands its peer inside the next collective.
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<uint32_t>(state.range(0));
        window_ = static_cast<uint32_t>(state.range(1));
        iters_ = static_cast<uint32_t>(state.range(2));
        warmup_pct_ = static_cast<uint32_t>(state.range(3));
        pp_iters_ = static_cast<uint32_t>(state.range(4));
        verify_ = state.range(5) != 0;
        page_ = tt_uva_frame_page_size(payload_bytes_);

        const mh::ContextPtr world = mh::DistributedContext::get_current_world();
        rank_ = static_cast<uint32_t>(*world->rank());
        ranks_ = static_cast<uint32_t>(*world->size());
        peer_ = 1 - rank_;

        src_off_ = kSlotsOff + static_cast<uint64_t>(window_) * page_;
        bytes_ = src_off_ + static_cast<uint64_t>(window_) * page_;

        // Page-aligned so a slot never straddles one; both ranks size it identically.
        // posix_memalign, not std::aligned_alloc: the latter is absent from libc++.
        void* raw = nullptr;
        if (::posix_memalign(&raw, 4096, (bytes_ + 4095) & ~uint64_t{4095}) != 0) {
            raw = nullptr;
        }
        base_ = static_cast<uint8_t*>(raw);
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

    // Collective too: the window's teardown is, and the region it registered must outlive
    // it. Runs on paths a mid-body return would skip.
    void TearDown(benchmark::State& state) override {
        (void)state;
        if (win_) {
            if (const std::string e = win_->barrier(); !e.empty()) {
                std::fprintf(stderr, "barrier: %s\n", e.c_str());
            }
            win_.reset();
        }
        std::free(base_);
        base_ = nullptr;
    }

protected:
    // Filled once: the payload is constant across frames, so only word 0 and the trailer
    // are touched per frame in the timed loop.
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

    // Trailer last in the page, so an armed guard on the peer means the payload ahead of it
    // landed -- the same trailing-flag rule H2HSocket relies on.
    void arm_frame(uint8_t* src, uint32_t stamp) const {
        if (verify_) {
            *reinterpret_cast<uint32_t*>(src) = stamp;
        }
        FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(src + payload_bytes_);
        std::memset(t, 0, sizeof(*t));
        t->length = payload_bytes_;
        t->guard = tt_uva_frame_guard(kFrameVersion);
    }

    // Rank 1's side of verify. The stamp is the sender's frame index, so a stale or
    // duplicated slot is caught, not just a corrupted one.
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

    // Equal payload each way, so halving the round trip is defensible -- unlike the credit
    // path, where an 8 B credit answers a full frame. Slot 0 only: latency, not depth.
    bool ping_pong(std::vector<double>& rt, std::string& err) {
        uint8_t* const src = base_ + src_off_;
        volatile uint64_t* const guard = reinterpret_cast<volatile uint64_t*>(base_ + kSlotsOff + payload_bytes_);
        const uint32_t warmup = pp_iters_ * warmup_pct_ / 100;
        rt.reserve(pp_iters_ - warmup);
        const bool initiator = rank_ == 0;
        // Trailer is constant across iterations, so it is built once rather than inside the
        // timed loop; only word 0 moves per frame, and only under verify.
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

        if (rank_ == 0) {
            // Rank 0 sends and is the only side that times: the credit closes each frame on
            // the clock that opened it, so nothing here needs the two clocks related.
            std::vector<RdmaWindow::Op> ops(window_);
            std::vector<std::chrono::steady_clock::time_point> posted_at(window_);
            rt_us.reserve(measured);

            volatile uint64_t* const credit = reinterpret_cast<volatile uint64_t*>(base_ + kCreditOff);
            auto t0 = std::chrono::steady_clock::now();
            uint32_t posted = 0;
            uint32_t credited = 0;
            // Re-armed on forward motion, so a long run never trips it and a stall always does.
            auto deadline = std::chrono::steady_clock::now() + kStall;

            while (credited < total && ok) {
                while (posted < total && posted - credited < window_) {
                    const uint32_t s = posted % window_;
                    // A credit says the TARGET has the frame, not that this Rput retired.
                    // Reusing the origin or the Op before it races the send.
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

                // Local completion only -- it frees the origin buffer and turns the progress
                // engine. Remote arrival is what the credit reports, so nothing waits here.
                for (uint32_t s = 0; s < window_; ++s) {
                    if (ops[s].valid()) {
                        (void)win_->test(ops[s]);
                    }
                }

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
            // Rank 1 walks its slots in order and credits one frame at a time, which is the
            // granularity the round trip is being measured at.
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
                    // Zeroed before the credit, never after: the credit lets the peer re-arm
                    // this slot, and a late zero would erase a fresh frame.
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
                    // Passive target, but some transports only move an inbound put while
                    // this rank is inside MPI. Costs nothing with nothing outstanding.
                    (void)win_->flush(peer_);
                    idle = 0;
                }
            }
            if (bad != 0) {
                err = "rank 1: " + std::to_string(bad) + " frames failed the frame check";
                ok = false;
            }
        }

        // Both ranks leave the streaming phase with the same verdict, which is what lets the
        // ping-pong below be entered by both or neither.
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

        // Rank 1 measured nothing; its counters stay at the zeros init_counters set, and
        // main() gives it a reporter that prints none of them.
        if (rank_ != 0) {
            continue;
        }
        state.counters["frames"] = static_cast<double>(total);
        state.counters["bad_frames"] = static_cast<double>(bad);
        if (window_us > 0.0) {
            const double gb = static_cast<double>(measured) * payload_bytes_ / 1e9;
            state.counters["throughput_gbps"] = gb / (window_us / 1e6);
        }
        // Both are latencies: put->credit is what a slot actually costs before it can be
        // reused, one-way is the symmetric half of the ping-pong round trip.
        set_latency_counters(state, summarize_latency_us(rt_us, 0.0), rt_us.size());
        for (double& v : pp_us) {
            v /= 2.0;
        }
        set_latency_counters(state, summarize_latency_us(pp_us, 0.0), pp_us.size(), "oneway_");
    }
}

BENCHMARK_REGISTER_F(H2HLegFixture, Bandwidth)
    // Named explicitly: a fixture benchmark is otherwise reported as Fixture/Case, and
    // analyze_hd_sockets.py takes the name up to the first slash as the benchmark key.
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
    // The run is the measurement, so a repeat would only re-pay the window bringup. Both
    // ranks must agree on the repetition count, so do not pass --benchmark_repetitions to one.
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

// Rank 1 runs every case in lockstep with rank 0 but reports nothing, so the two ranks do
// not interleave output on one terminal.
class NullReporter : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
};

}  // namespace

int main(int argc, char** argv) {
    // Before Initialize: the context takes MPI's own argv entries first, and every collective
    // below needs it up.
    mh::DistributedContext::create(argc, argv);
    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    const uint32_t rank = static_cast<uint32_t>(*world->rank());
    if (*world->size() != 2) {
        if (rank == 0) {
            std::fprintf(stderr, "error: this leg needs exactly 2 ranks; launch with `mpirun -n 2`\n");
        }
        return 2;
    }

    // Identical argv on both ranks is what keeps them on the same case list. A filter or a
    // repetition count given to one rank only will deadlock the pair inside a collective.
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
