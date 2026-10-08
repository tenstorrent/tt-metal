// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// erisc H2H benchmark: two ranks, MPI one-sided, no device. The headline is posts_per_flush, not
// GB/s, and the epoch split caps it at 0.5. Two clocks, so round trip only -- no one-way latency.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include "hostdevcommon/erisc_bridge_layout.h"
#include "tests/tt_metal/distributed/erisc_bridge_bench_common.hpp"
#include "tt_metal/distributed/erisc_h2h_socket.hpp"

using namespace tt::tt_metal::experimental;
namespace bb = tt::tt_fabric::erisc_bridge::bench;

namespace {

std::uint32_t env_u32(const char* n, std::uint32_t d) {
    const char* v = std::getenv(n);
    return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 0)) : d;
}

int g_rank = 0;
int g_size = 0;

// Rank 1 runs in lockstep but reports nothing; a null reporter would select the console one.
struct SilentReporter : benchmark::BenchmarkReporter {
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
};

struct H2HFixture : public benchmark::Fixture {
    void SetUp(benchmark::State& state) override {
        capacity_ = static_cast<std::uint32_t>(state.range(bb::kPacketSize));
        verify_ = state.range(bb::kVerify) != 0;
        if (bb::sweep_requested(state)) {
            bb::sweep_not_implemented(state);
            return;
        }
        arenas_ = static_cast<std::uint32_t>(state.range(bb::kLegSpecific0 + 0));
        ring_ = env_u32("TT_BRIDGE_H2H_RING", static_cast<std::uint32_t>(state.range(bb::kLegSpecific0 + 1)));
        // Frames PER ARENA, so total_amt is split across them: the arenas run concurrently and
        // the byte total is what the leg moved, not what one arena moved.
        // TT_BRIDGE_H2H_MIB overrides total_amt so a sweep can use sizes the grid does not register.
        const auto mib = env_u32("TT_BRIDGE_H2H_MIB", static_cast<std::uint32_t>(state.range(bb::kTotalAmt)));
        const std::uint64_t amt = static_cast<std::uint64_t>(mib) * bb::kMiB;
        const std::uint64_t per = amt / capacity_ / (arenas_ != 0 ? arenas_ : 1);
        measured_ = static_cast<std::uint32_t>(per != 0 ? per : 1);
        warmup_ = static_cast<std::uint32_t>(static_cast<std::uint64_t>(measured_) * state.range(bb::kWarmupPct) / 100);
        frames_ = measured_ + warmup_;
    }

    std::uint32_t capacity_ = 0;
    std::uint32_t arenas_ = 4;
    std::uint32_t ring_ = 32;
    std::uint32_t measured_ = 0;
    std::uint32_t warmup_ = 0;
    std::uint32_t frames_ = 0;
    bool verify_ = false;
};

}  // namespace

BENCHMARK_DEFINE_F(H2HFixture, Bridge)(benchmark::State& state) {
    const std::uint32_t page = tt::tt_fabric::bridge_slot_size(capacity_);
    const std::uint64_t region_bytes = tt::tt_fabric::bridge_region_bytes(arenas_, ring_, capacity_);

    for (auto _ : state) {
        state.PauseTiming();
        // 2 MiB aligned like the real region, so segment offsets land where the layout says.
        void* raw = nullptr;
        if (posix_memalign(
                &raw, static_cast<size_t>(tt::tt_fabric::kBridgeRegionAlign), static_cast<size_t>(region_bytes)) != 0) {
            state.SkipWithError("posix_memalign failed for the bridge region");
            return;
        }
        auto* base = static_cast<std::uint8_t*>(raw);
        std::memset(base, 0, static_cast<size_t>(region_bytes));
        // Tag words depend only on their index, so every Tx slot is patterned once, outside the
        // timed loop; per frame only word 0 (the frame number) is written, as a producer would.
        for (std::uint32_t a = 0; a < arenas_; ++a) {
            for (std::uint32_t sl = 0; sl < ring_; ++sl) {
                auto* w = reinterpret_cast<std::uint32_t*>(
                    base +
                    tt::tt_fabric::bridge_segment_offset(a, tt::tt_fabric::BridgeArena::Tx, arenas_, ring_, capacity_) +
                    tt::tt_fabric::bridge_slot_offset_in_segment(sl, capacity_));
                for (std::uint32_t k = bb::kPayloadFirstTagWord; k < capacity_ / 4; ++k) {
                    w[k] = bb::payload_tag_word(k);
                }
            }
        }

        EriscH2HSocket::Config cfg;
        cfg.arenas = arenas_;
        cfg.chans_per_link = env_u32("TT_BRIDGE_CHANS_PER_LINK", 2);
        cfg.page_bytes = page;
        cfg.ring_pages = ring_;
        cfg.peer_rank = g_rank == 0 ? 1 : 0;
        cfg.host_rank = static_cast<std::uint32_t>(g_rank);
        cfg.host_count = static_cast<std::uint32_t>(g_size);
        cfg.region_base = base;
        cfg.region_bytes = region_bytes;
        // Bounded by the Tx slot count: submit() only queues, so a deeper queue reuses an owed slot.
        cfg.max_queued_frames = env_u32("TT_H2H_MAX_QUEUED", arenas_ * ring_);
        cfg.max_put_bytes = env_u32("TT_H2H_MAX_PUT", 32768);  // match btl_sm_eager_limit
        cfg.max_batch = env_u32("TT_H2H_MAX_BATCH", 0);
        cfg.collect_timing = true;
        cfg.timing_samples = static_cast<std::uint64_t>(frames_) * arenas_;

        std::string err;
        auto sock = EriscH2HSocket::create(cfg, err);
        if (!sock) {
            std::free(raw);
            state.SkipWithError(err.c_str());
            return;
        }
        (void)sock->barrier();
        state.ResumeTiming();

        std::uint64_t moved = 0;
        std::uint64_t bad_payload = 0;
        const auto t0 = std::chrono::steady_clock::now();
        // A deadline that resets on progress: a fixed 60 s wall let a rank stuck at one frame sit
        // for a minute and then report no error, which reads as a pass.
        const auto kStall = std::chrono::seconds(2);
        auto give_up = std::chrono::steady_clock::now() + kStall;
        std::string stall_why;
        const std::uint64_t want = static_cast<std::uint64_t>(frames_) * arenas_;
        // The clock and the counters restart once the warmup frames are credited, so they cover only
        // the measured frames (rank 0 reports; rank 1 keeps its whole-run verify counts).
        const std::uint64_t warm = static_cast<std::uint64_t>(warmup_) * arenas_;
        auto t_meas = t0;
        std::uint64_t m_meas = 0;

        if (g_rank == 0) {
            // Per arena, not one counter across all: the RX advances arena a on delivered[a] + 1,
            // so a global counter gives arena 1 a first frame it will never accept.
            std::vector<std::uint32_t> cntr(arenas_, 0);
            // Done when the RX has CREDITED every frame, not when the last one is queued.
            while (moved < want && std::chrono::steady_clock::now() < give_up) {
                // Fill every arena up to its credited ring, so one poll() can batch many frames.
                // Tx slot cntr % ring is free only once frame cntr - ring was credited.
                for (std::uint32_t a = 0; a < arenas_; ++a) {
                    while (cntr[a] < frames_ && cntr[a] - sock->credit_total(a) < ring_) {
                        BridgeSendTask t;
                        t.arena = a;
                        t.page_offset = tt::tt_fabric::bridge_segment_offset(
                                            a, tt::tt_fabric::BridgeArena::Tx, arenas_, ring_, capacity_) +
                                        tt::tt_fabric::bridge_slot_offset_in_segment(cntr[a] % ring_, capacity_);
                        t.page_bytes = page;
                        t.ordering_cntr = cntr[a] + 1;
                        t.length = capacity_;
                        *reinterpret_cast<std::uint32_t*>(base + t.page_offset) = cntr[a] + 1;
                        if (!sock->submit(t)) {
                            break;
                        }
                        ++cntr[a];
                        give_up = std::chrono::steady_clock::now() + kStall;
                    }
                }
                (void)sock->poll(nullptr, [](const BridgeDeliverTask&) { return true; });
                std::uint64_t credited = 0;
                for (std::uint32_t a = 0; a < arenas_; ++a) {
                    credited += sock->credit_total(a);
                }
                if (credited != moved) {
                    // Restart only if most measured frames remain: in a tiny run one batch can leap past
                    // warmup and leave a handful of frames to time.
                    if (moved < warm && credited >= warm && 2 * (want - credited) >= want - warm) {
                        t_meas = std::chrono::steady_clock::now();
                        m_meas = credited;
                        sock->reset_stats();
                    }
                    moved = credited;
                    give_up = std::chrono::steady_clock::now() + kStall;
                }
            }
        } else {
            std::uint64_t got = 0;
            while (got < want && std::chrono::steady_clock::now() < give_up) {
                // t.arena, NOT 0: credit is per arena because the ring is, and crediting the wrong
                // one leaves that arena permanently full after exactly `ring` frames.
                const std::uint64_t before = got;
                got += sock->poll(nullptr, [&](const BridgeDeliverTask& t) -> bool {
                    if (verify_) {
                        const auto* w = reinterpret_cast<const std::uint32_t*>(base + t.page_offset);
                        const std::uint32_t words = t.length / 4;
                        if (w[0] != t.ordering_cntr) {
                            ++bad_payload;
                        } else {
                            for (std::uint32_t k = bb::kPayloadFirstTagWord; k < words; ++k) {
                                if (w[k] != bb::payload_tag_word(k)) {
                                    ++bad_payload;
                                    break;
                                }
                            }
                        }
                    }
                    sock->consumed(t.arena, 1);
                    return true;
                });
                (void)sock->publish_credits();
                if (got != before) {
                    give_up = std::chrono::steady_clock::now() + kStall;
                }
            }
            moved = got;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_meas).count();

        state.PauseTiming();
        const auto& st = sock->pass_stats();
        // Only the RX checks payload and order, and only rank 0 reports: sum both ranks' counts.
        std::uint64_t health[3] = {bad_payload, st.bad_order, moved < want ? 1u : 0u};
        MPI_Allreduce(MPI_IN_PLACE, health, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (g_rank == 0 && secs > 0.0) {
            const double mb = static_cast<double>(moved - m_meas) * capacity_ / 1e6;
            bb::emit_throughput(state, mb / secs, moved, /*sustained=*/moved > ring_ * arenas_);
            // THE HEADLINE: half the frames per batch, since each batch costs two flushes.
            state.counters["posts_per_flush"] =
                st.flushes != 0 ? static_cast<double>(st.posts) / static_cast<double>(st.flushes) : 0.0;
            state.counters["flushes"] = static_cast<double>(st.flushes);
            state.counters["payload_puts"] = static_cast<double>(st.payload_puts);
            state.counters["trailer_puts"] = static_cast<double>(st.trailer_puts);
            state.counters["starved_credit"] = static_cast<double>(st.starved_credit);
        }
        state.counters["arenas"] = static_cast<double>(arenas_);
        state.counters["ring_pages"] = static_cast<double>(ring_);
        state.counters["moved"] = static_cast<double>(moved);
        state.counters["want"] = static_cast<double>(want);
        if (health[2] != 0) {
            stall_why = "stalled at " + std::to_string(moved) + " of " + std::to_string(want) + " frames (rank " +
                        std::to_string(g_rank) + ") after " + std::to_string(kStall.count()) + "s without progress";
        }
        // PUT -> CREDIT, a ROUND TRIP. Two hosts, two clocks: there is no common time base to
        // stamp a one-way span against, so this is reported as rtt_us_* and never halved.
        {
            const auto& ns = sock->put_to_credit_ns();
            std::vector<double> us;
            us.reserve(ns.size());
            for (const auto v : ns) {
                us.push_back(static_cast<double>(v) / 1000.0);
            }
            bb::emit_rtt_us(state, std::move(us));
        }
        // bad_payload measured HERE: the socket has no pattern to check against, so its own field
        // is a literal. The benchmark wrote the pattern, so the benchmark checks it.
        state.counters["verified"] = verify_ ? 1.0 : 0.0;
        bb::emit_health(state, health[0], st.starved_credit, health[1]);
        if (!sock->first_error().empty()) {
            state.SkipWithError(sock->first_error().c_str());
        } else if (!stall_why.empty()) {
            state.SkipWithError(stall_why.c_str());
        }
        sock.reset();
        std::free(raw);
        state.ResumeTiming();
    }
}

BENCHMARK_REGISTER_F(H2HFixture, Bridge)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    // The common six, in the shared order, then this leg's own.
    ->ArgNames(bb::arg_names_with({"arenas", "ring"}))
    ->ArgsProduct({
        {1, 4, 16},           // total_amt: MiB of MEASURED payload across all arenas
        {0, 10},              // warmup_pct
        {1024, 4096, 16384},  // packet_size: BYTES -- a TRUE runtime axis here, host only
        {0, 1},               // verify
        {0},                  // pace: not used by H2H
        {0},                  // sweep: one configuration per case
        {1, 2, 4},            // arenas
        {32},                 // ring pages per segment
    });

int main(int argc, char** argv) {
    // May own MPI only because it never creates a device: tt-metal's distributed context calls
    // MPI_Init_thread itself, and a second init aborts the rank. A MeshDevice here breaks that.
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    const int size = g_size;
    if (size != 2) {
        if (g_rank == 0) {
            std::fprintf(stderr, "erisc H2H benchmark needs exactly 2 ranks, got %d\n", size);
        }
        MPI_Finalize();
        return 2;
    }
    benchmark::Initialize(&argc, argv);
    if (g_rank == 0) {
        benchmark::RunSpecifiedBenchmarks();
    } else {
        SilentReporter silent;
        benchmark::RunSpecifiedBenchmarks(&silent);
    }
    benchmark::Shutdown();
    MPI_Finalize();
    return 0;
}
