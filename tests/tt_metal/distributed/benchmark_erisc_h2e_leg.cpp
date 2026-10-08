// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// H2E leg benchmark: host -> ERISC -> T6, against a STOCK router. Throughput is credit-bounded;
// latency_us_* is a true one-way span, since one clock stamps both ends.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tests/tt_metal/distributed/erisc_bridge_bench_common.hpp"
#include "tests/tt_metal/distributed/erisc_bridge_bench_mesh.hpp"
#include "tt_metal/distributed/erisc_h2e_leg.hpp"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal::experimental;
namespace eb = tt::tt_fabric::erisc_bridge;
namespace bb = tt::tt_fabric::erisc_bridge::bench;

namespace {

// Frames injected per eth channel in this process: every case shares the router's receiver cursor.
std::map<std::uint32_t, std::uint64_t> g_frames_on_chan;

std::uint32_t env_u32(const char* n, std::uint32_t d) {
    const char* v = std::getenv(n);
    return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 0)) : d;
}

struct H2ELegFixture : public benchmark::Fixture {
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<std::uint32_t>(state.range(bb::kPacketSize));
        verify_ = state.range(bb::kVerify) != 0;
        pace_ = state.range(bb::kPace) != 0;
        // FRAMES PER drained() REFRESH. The occupancy read is one non-posted PCIe round trip
        // per call, not per frame, so at batch:1 it is paid on every packet and dominates.
        batch_ = static_cast<std::uint32_t>(state.range(bb::kLegSpecific0));
        if (batch_ == 0) {
            batch_ = 1;
        }
        if (bb::sweep_requested(state)) {
            bb::sweep_not_implemented(state);
            return;
        }
        // TT_BRIDGE_H2E_MIB overrides total_amt so a sweep can use sizes the grid does not register.
        const auto mib = env_u32("TT_BRIDGE_H2E_MIB", static_cast<std::uint32_t>(state.range(bb::kTotalAmt)));
        const std::uint64_t amt = static_cast<std::uint64_t>(mib) * bb::kMiB;
        measured_ = static_cast<std::uint32_t>(amt / payload_bytes_);
        // ADDED on top, so the measured count stays exactly total_amt across warmup_pct.
        warmup_ = static_cast<std::uint32_t>(static_cast<std::uint64_t>(measured_) * state.range(bb::kWarmupPct) / 100);
        total_ = measured_ + warmup_;
    }

    std::uint32_t payload_bytes_ = 0;
    std::uint32_t measured_ = 0;
    std::uint32_t warmup_ = 0;
    std::uint32_t total_ = 0;
    bool verify_ = false;
    bool pace_ = false;
    std::uint32_t batch_ = 1;
};

}  // namespace

BENCHMARK_DEFINE_F(H2ELegFixture, Bridge)(benchmark::State& state) {
    auto mesh = bb::the_mesh();
    if (!mesh) {
        state.SkipWithError("mesh not initialised");
        return;
    }
    auto* dev = mesh->get_device(MeshCoordinate(0, 0));
    const auto dst_virt = dev->worker_core_from_logical_core(CoreCoord{0, 0});

    H2ELeg::Config cfg;
    cfg.page_bytes = payload_bytes_;
    cfg.link_idx = env_u32("TT_BRIDGE_LINK_IDX", 0);
    cfg.eth_chan = env_u32("TT_BRIDGE_H2E_CHAN", H2ELeg::Config::kUnsetChan);  // a sweep names the channel
    const std::uint32_t ring = env_u32("TT_BRIDGE_RING_PAGES", 32);            // source pages, reused round robin
    cfg.credit_margin = env_u32("TT_H2E_CREDIT_MARGIN", 2);
    cfg.dst_noc_x = static_cast<std::uint32_t>(dst_virt.x);
    cfg.dst_noc_y = static_cast<std::uint32_t>(dst_virt.y);
    cfg.dst_l1_addr = env_u32("TT_H2E_DST_L1", 0x40000);
    // No E2H producer here, so no arena holds prebuilt packets -- the leg synthesises a header
    // around each payload. In the pipeline it does not have to.
    cfg.frames_carry_header = false;
    cfg.collect_timing = true;
    cfg.timing_from_frame = warmup_;  // percentiles describe the measured frames, not the warmup
    cfg.frames_before = g_frames_on_chan[cfg.eth_chan];

    // A real payload, so verify has something to check. Stands in for the Rx arena H2H fills.
    std::vector<std::uint8_t> payload_src(static_cast<std::uint64_t>(ring) * cfg.page_bytes, 0);
    cfg.alias_region_base = payload_src.data();

    // Word 0 is the frame number, so a mismatch says WHICH frame arrived. The rest mixes frame
    // and offset, so a stuck slot, a short write and a stale page fail differently.
    const std::uint32_t words = payload_bytes_ / 4;
    auto fill_slot = [&](std::uint32_t slot, std::uint32_t frame) {
        auto* w =
            reinterpret_cast<std::uint32_t*>(payload_src.data() + static_cast<std::uint64_t>(slot) * cfg.page_bytes);
        w[0] = frame;
        for (std::uint32_t i = 1; i < words; ++i) {
            w[i] = frame * 0x9E3779B9u + i;
        }
    };

    std::string err;
    auto leg = H2ELeg::create(mesh, cfg, err);
    if (!leg) {
        state.SkipWithError(err.c_str());
        return;
    }

    // Synthetic source, so the leg is measured ALONE: publish() builds the header itself and
    // writes the payload straight from payload_src above.
    std::uint32_t injected = 0;
    std::uint64_t verify_fail = 0;
    std::uint64_t bad_order = 0;
    std::uint32_t last_bad_frame = 0;  // what dst held instead, for a bad_order report
    std::chrono::steady_clock::time_point t_first{}, t_last{};
    std::uint32_t f_first = 0;  // frames injected when t_first was stamped

    for (auto _ : state) {
        for (std::uint32_t f = 0; f < total_;) {
            const std::uint32_t want = std::min(batch_, total_ - f);
            std::uint32_t n = 0;
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);

            while (n < want) {
                BridgeDeliverTask t;
                t.arena = 0;
                t.slot = (f + n) % ring;
                // REGION-relative, as H2H delivers it: the leg reads alias_region_base + this.
                t.page_offset = static_cast<std::uint64_t>(t.slot) * payload_bytes_;
                t.page_bytes = payload_bytes_;
                t.ordering_cntr = f + n + 1;
                t.length = payload_bytes_;
                // Rewritten every lap: the slot is reused every ring_pages frames, so a stale
                // pattern would verify against the wrong frame.
                fill_slot(t.slot, f + n + 1);
                // FALSE IS BACKPRESSURE, not failure: the router has not kept up. Refresh and
                // re-offer rather than dropping.
                if (!leg->publish(t)) {
                    (void)leg->drained(0);
                    if (!leg->first_error().empty() || std::chrono::steady_clock::now() > give_up) {
                        break;
                    }
                    continue;
                }
                ++n;
            }
            if (n == 0 && !leg->first_error().empty()) {
                state.SkipWithError(leg->first_error().c_str());
                return;
            }
            if (n == 0) {
                // Name the count, as the other legs do: "stalled" alone does not say whether
                // nothing ever moved or it died at frame 1000 of 1024.
                const std::string why = "router stopped consuming -- stalled at " + std::to_string(f) + " of " +
                                        std::to_string(total_) + " frames injected, " +
                                        std::to_string(leg->drained(0)) + " consumed";
                state.SkipWithError(why.c_str());
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            if (f <= warmup_ && f + n > warmup_) {
                t_first = now;
                f_first = f + n;
            }
            if (f + n > warmup_) {
                t_last = now;
            }
            f += n;
            injected += n;

            // Read back what the router delivered: dst_l1_addr holds the most recent frame, so
            // this checks the batch's last. Verify only -- it costs a non-posted PCIe read.
            if (verify_) {
                const auto give_up_v = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                while (leg->drained(0) < injected && std::chrono::steady_clock::now() < give_up_v) {
                }
                std::vector<std::uint32_t> got(words, 0);
                // The router frees the slot when it ISSUES the NOC write, so the last frame may still be
                // landing: re-read until it does, and only a frame still wrong at the deadline counts.
                bool read_ok = false;
                const auto land_by = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
                do {
                    read_ok = eb::read_bytes_from_l1(
                        dev, CoreCoord{0, 0}, cfg.dst_l1_addr, got.data(), payload_bytes_, tt::CoreType::WORKER);
                } while (read_ok && got[0] != f && std::chrono::steady_clock::now() < land_by);
                if (!read_ok) {
                    ++verify_fail;
                } else {
                    // Word 0 names the frame. Wrong frame and wrong bytes are different
                    // faults: one is ordering or a missed slot, the other is corruption.
                    if (got[0] != f) {
                        ++bad_order;
                        last_bad_frame = got[0];
                    }
                    const std::uint32_t frame = got[0];
                    for (std::uint32_t i = 1; i < words; ++i) {
                        if (got[i] != frame * 0x9E3779B9u + i) {
                            ++verify_fail;
                            break;
                        }
                    }
                }
            }
            // PACED: let the pipe empty so each sample is transit, not queueing.
            if (pace_) {
                while (leg->drained(0) < injected && std::chrono::steady_clock::now() < give_up) {
                }
            }
        }
    }

    // DRAIN before reporting: a run otherwise ends with a frame still in the router, counted as
    // injected but not consumed, purely because the loop stopped.
    {
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (leg->drained(0) < injected && std::chrono::steady_clock::now() < give_up) {
        }
    }
    g_frames_on_chan[cfg.eth_chan] += injected;

    state.counters["injected"] = static_cast<double>(injected);
    state.counters["consumed"] = static_cast<double>(leg->drained(0));
    state.counters["packet_size"] = static_cast<double>(payload_bytes_);
    state.counters["warmup_frames"] = static_cast<double>(warmup_);

    // Not under verify: draining to empty each batch serialises the pipeline as pace does, so the
    // rate measures that wait, not the link.
    if (!pace_ && !verify_ && injected > f_first) {
        const double us = std::chrono::duration<double, std::micro>(t_last - t_first).count();
        if (us > 0.0) {
            // Frames after t_first: the batch that crossed warmup is already in by then.
            const double mb = static_cast<double>(injected - f_first) * payload_bytes_ / 1e6;
            bb::emit_throughput(state, mb / us * 1e6, injected - warmup_, /*sustained=*/injected > ring * 4);
        }
    }
    // Measured, all three -- credit_stalls had been counted in the leg all along with no accessor.
    // Under verify the pipe drains each batch, so the percentiles describe a quiesced link.
    state.counters["drained_per_batch"] = verify_ ? 1.0 : 0.0;
    bb::emit_latency_oneway(state, leg->latency_us());
    state.counters["oversize"] = static_cast<double>(leg->oversize());
    state.counters["slot_mismatch"] = static_cast<double>(leg->slot_mismatch());
    if (bad_order != 0) {
        state.counters["last_bad_frame"] = static_cast<double>(last_bad_frame);
    }
    bb::emit_health(state, verify_fail, leg->credit_stalls(), bad_order);
    if (!verify_) {
        // Say so, rather than let a 0 read as "checked and clean".
        state.counters["verified"] = 0;
    } else {
        state.counters["verified"] = 1;
    }
    if (!leg->first_error().empty()) {
        state.SkipWithError(leg->first_error().c_str());
    }
}

BENCHMARK_REGISTER_F(H2ELegFixture, Bridge)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    // The common six, in the shared order, then this leg's own.
    ->ArgNames(bb::arg_names_with({"batch"}))
    ->ArgsProduct({
        {1, 4, 16},          // total_amt: MiB of MEASURED payload
        {0, 10},             // warmup_pct: ADDED on top
        {1024, 2048, 4096},  // packet_size: BYTES
        {0, 1},              // verify
        {0, 1},              // pace: 0 throughput, 1 latency
        {0},                 // sweep: one channel per process; the script iterates channels
        {1, 8, 32},          // batch: frames between occupancy refreshes
    });

int main(int argc, char** argv) {
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);

    // No E2H bridge: H2E injects into a STOCK router's receiver channel (both measured at ~290 MB/s).
    bb::the_mesh() = MeshDevice::create(MeshDeviceConfig(std::nullopt));

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    bb::the_mesh()->close();
    bb::the_mesh().reset();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    return 0;
}
