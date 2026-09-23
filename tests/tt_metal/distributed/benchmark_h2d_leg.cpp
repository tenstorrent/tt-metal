// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-device leg alone, under google-benchmark: one rank, no peer, no H2H. This
// host fills the RX rings in place of a peer's RMA, so the only PCIe traffic is the pull.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "hd_socket_test_utils.hpp"

#include "tt_metal/distributed/host_h2d_leg.hpp"
#include "tt_metal/distributed/host_l1_map.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>
#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace dist = tt::tt_metal::distributed;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

constexpr int kDeviceId = 0;

// A stalled run must fail rather than spin: the receiver kernel exits only once it has seen
// `iters` frames, so Finish() is the other unbounded wait.
constexpr auto kStall = std::chrono::seconds(30);

// The swept axes. A single-element list pins one: `cores` must stay pinned because the
// region maps once per process and refuses a second, differently-sized reservation.
const std::vector<int64_t> kPageSizes = {16384};
const std::vector<int64_t> kCores = {4};
const std::vector<int64_t> kRingPages = {8};
const std::vector<int64_t> kIterations = {20000};
const std::vector<int64_t> kWarmupPct = {10};
const std::vector<int64_t> kVerify = {0};

// SkipWithError marks the report but not the exit status, and analyze_hd_sockets.py drops
// errored rows -- so a failed run would read as no data. main() returns this instead.
bool g_run_failed = false;

void fail(benchmark::State& state, const std::string& why) {
    g_run_failed = true;
    state.SkipWithError(why);
}

// ---------------------------------------------------------------------------------------
// Metrics. Copied from benchmark_hd_sockets.cpp:159-168 and :194-215 and :518-528, which is
// not to be modified; the symbols there are file-local.
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

// The us variant, not the cycles one: this leg's round trip is stamped on the host clock,
// so cycles_per_us only back-fills the cycle columns the CSV header carries.
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

void set_latency_counters(benchmark::State& state, const LatencySummary& s, uint64_t num_iterations) {
    state.counters["num_iterations"] = static_cast<double>(num_iterations);
    state.counters["avg_us"] = s.avg_us;
    state.counters["min_us"] = s.min_us;
    state.counters["max_us"] = s.max_us;
    state.counters["p50_us"] = s.p50_us;
    state.counters["p99_us"] = s.p99_us;
    state.counters["avg_cycles"] = s.avg_cycles;
    state.counters["min_cycles"] = static_cast<double>(s.min_cycles);
    state.counters["max_cycles"] = static_cast<double>(s.max_cycles);
}

// Pre-registered so the CSV header carries every column even when a case is skipped, which
// keeps the shape stable across a sweep.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["frames"] = 0;
    state.counters["bad_frames"] = 0;
    set_latency_counters(state, LatencySummary{}, 0);
}

double us_since(std::chrono::steady_clock::time_point t) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
    return static_cast<double>(ns) / 1e3;
}

// ---------------------------------------------------------------------------------------
// One device bringup per process. Fixture::SetUp runs once per arg case, so a SetUp that
// constructed the mesh would re-pay bringup at every point of a sweep.
// ---------------------------------------------------------------------------------------

struct DeviceFixture {
    std::shared_ptr<dist::MeshDevice> mesh_device;

    DeviceFixture() : mesh_device(dist::MeshDevice::create_unit_mesh(kDeviceId)) {}
};

DeviceFixture& get_device_fixture() {
    static DeviceFixture fixture;
    return fixture;
}

// What test_kernel_signal.cpp:67-68 expects in every payload word but word 0, derived from
// the trailer's own origin selector.
uint32_t pattern_word(uint32_t core) {
    const uint32_t b = 0x40u + (core & 0x1Fu);
    return b | (b << 8) | (b << 16) | (b << 24);
}

// Per core: how many frames this host has released and how many the device has taken back,
// plus the publish stamp of each live slot.
struct CoreState {
    uint64_t published = 0;
    uint64_t drained = 0;
    std::vector<std::chrono::steady_clock::time_point> at;
};

class H2DLegFixture : public benchmark::Fixture {
public:
    // Everything that can refuse the run lives here, so the body is only the measurement.
    // A skipped state does not enter the loop, which is what makes this the gate.
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<uint32_t>(state.range(0));
        cores_ = static_cast<uint32_t>(state.range(1));
        ring_pages_ = static_cast<uint32_t>(state.range(2));
        iters_ = static_cast<uint32_t>(state.range(3));
        verify_ = state.range(5) != 0;
        page_ = tt_uva_frame_page_size(payload_bytes_);

        // RingAlias refuses an overlay wider than an arena, but only once the sockets are
        // built. Cheaper to say so here, in terms of the knob the caller turned.
        if (static_cast<uint64_t>(ring_pages_) * page_ > kArenaBytes) {
            fail(state,
                 "ring " + std::to_string(ring_pages_) + " x page " + std::to_string(page_) + " B exceeds the " +
                     std::to_string(kArenaBytes >> 10) + " KiB arena; lower ring_pages or page_size");
            return;
        }

        mesh_ = get_device_fixture().mesh_device;
        // The device pulls these pages over PCIe, so a remote chip cannot serve them.
        if (!dist::is_device_coord_mmio_mapped(mesh_, dist::MeshCoordinate(0, 0))) {
            fail(state, "device " + std::to_string(kDeviceId) + " is not MMIO-mapped");
            return;
        }

        IDevice* const device = mesh_->get_devices().front();
        const CoreCoord grid = device->compute_with_storage_grid_size();
        grid_width_ = static_cast<uint32_t>(grid.x);
        grid_height_ = static_cast<uint32_t>(grid.y);
        if (cores_ == 0 || cores_ > grid_width_ * grid_height_ || cores_ > kProvisionedCores) {
            fail(state, "cores " + std::to_string(cores_) + " does not fit this grid");
            return;
        }

        const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        l1_ = L1MapNew::compute(l1_base, static_cast<uint32_t>(device->l1_size_per_core()), payload_bytes_, false);
        if (const std::string e = l1_.fits(payload_bytes_); !e.empty()) {
            fail(state, e);
            return;
        }

        core_list_.clear();
        cores_set_ = CoreRangeSet();
        for (uint32_t i = 0; i < cores_; ++i) {
            const CoreCoord c{i % grid_width_, i / grid_width_};
            core_list_.push_back(c);
            cores_set_ = cores_set_.merge(CoreRangeSet(CoreRange(c, c)));
        }

        total_frames_ = static_cast<uint64_t>(cores_) * iters_;
        // Global, not per core: the window opens when the frame count crosses it, so the
        // figure this divides by is the same one that starts the clock.
        warmup_frames_ = total_frames_ * static_cast<uint64_t>(state.range(4)) / 100;
        cycles_per_us_ = dist::get_cycles_per_us(*mesh_);

        // The leg first, then the pin: H2DLeg MAP_FIXEDs its rings over the RX arenas, and
        // a pin taken before that would go on naming the pages it replaced.
        H2DLeg::Config hc;
        hc.cores = cores_;
        hc.grid_width = grid_width_;
        hc.page_bytes = page_;
        hc.ring_pages = ring_pages_;
        HostRegion& region = HostRegion::storage();
        std::string err;
        try {
            // Maps the region for these cores; the leg overlays its rings onto it below.
            region_base_ = region.reserved_base(cores_);
            hc.alias_region_base = region_base_;
            h2d_ = H2DLeg::create(mesh_, hc, err);
        } catch (const std::exception& ex) {
            fail(state, std::string("host region unavailable: ") + ex.what());
            return;
        }
        if (!h2d_) {
            fail(state, "h2d bringup failed: " + err);
            return;
        }
        try {
            // Aliases the h2d rings with the region reserved above; this is what gets
            // registered with RDMA via MPI_Windows once a peer exists.
            region.provision(
                mesh_, /*chip=*/0, cores_, HostTopology{0, 1, 1}, HostRegion::Grid{grid_width_, grid_height_});
        } catch (const std::exception& ex) {
            fail(state, std::string("host region unavailable: ") + ex.what());
            return;
        }
        if (const std::string e = region.verify_header(); !e.empty()) {
            fail(state, "region header check failed: " + e);
            return;
        }
        fill_rings();
    }

    // Unpin BEFORE the leg's destructor puts anonymous pages back over the arenas: the pin
    // must not still name the pages being swapped out. Runs on paths a return would skip.
    void TearDown(benchmark::State& state) override {
        (void)state;
        HostRegion& region = HostRegion::storage();
        if (region.is_provisioned()) {
            region.release();
        }
        h2d_.reset();
        mesh_.reset();
        region_base_ = nullptr;
    }

protected:
    // Filled ONCE, not per frame: the device only reads these pages, so republishing a slot
    // needs no rewrite and the timed loop carries no host memcpy.
    void fill_rings() const {
        for (uint32_t c = 0; c < cores_; ++c) {
            for (uint32_t s = 0; s < ring_pages_; ++s) {
                uint8_t* const slot = region_base_ + rx_slot_offset(c, s, page_);
                // Whole page, so the gap between the payload and the trailer is not left
                // carrying the arena's 0xA5 fill.
                std::memset(slot, 0, page_);
                // Patterned under verify because the kernel compares every payload word
                // against the origin's own byte. Word 0 is restamped per frame, in publish.
                if (verify_) {
                    uint32_t* const w = reinterpret_cast<uint32_t*>(slot);
                    const uint32_t want = pattern_word(c);
                    for (uint32_t k = 1; k < payload_bytes_ / sizeof(uint32_t); ++k) {
                        w[k] = want;
                    }
                }
                FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(slot + payload_bytes_);
                t->guard = tt_uva_frame_guard(kFrameVersion);
                t->length = payload_bytes_;
                t->origin = tt_uva_t6_global_selector(0, 0, c, 1);
                // The kernel exits on this word reaching `iters`, so every frame must carry
                // the ADD-1. apply_signal() is the only thing here that reads the trailer.
                t->sig_off = l1_.dest_word_addr - l1_.l1_base;
                t->sig_val = 1;
                t->sig_op = kSignalAdd;
            }
        }
    }

    std::shared_ptr<dist::MeshDevice> mesh_;
    std::unique_ptr<H2DLeg> h2d_;
    L1MapNew l1_{};
    CoreRangeSet cores_set_;
    std::vector<CoreCoord> core_list_;
    uint8_t* region_base_ = nullptr;
    uint32_t payload_bytes_ = 0;
    uint32_t page_ = 0;
    uint32_t cores_ = 0;
    uint32_t ring_pages_ = 0;
    uint32_t iters_ = 0;
    uint32_t grid_width_ = 0;
    uint32_t grid_height_ = 0;
    uint64_t total_frames_ = 0;
    uint64_t warmup_frames_ = 0;
    double cycles_per_us_ = 0.0;
    bool verify_ = false;
};

BENCHMARK_DEFINE_F(H2DLegFixture, Bandwidth)(benchmark::State& state) {
    init_counters(state);

    for ([[maybe_unused]] auto _ : state) {
        Program program = CreateProgram();
        auto receiver = CreateKernel(
            program,
            TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_signal.cpp",
            cores_set_,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_1,
                .compile_args = {
                    l1_.deliver_addr,
                    page_,
                    l1_.dest_word_addr,
                    iters_,
                    l1_.l1_base,
                    l1_.l1_size,
                    payload_bytes_,
                    verify_ ? 1u : 0u,
                    l1_.verify_addr}});
        const std::vector<uint32_t> cfg = h2d_->config_addresses();
        for (uint32_t i = 0; i < cores_; ++i) {
            SetRuntimeArgs(program, receiver, core_list_[i], {cfg[i], 1u});
        }
        dist::execute_program_on_device(*mesh_, dist::MeshCoordinate(0, 0), std::move(program));

        std::vector<CoreState> core(cores_);
        for (auto& s : core) {
            s.at.resize(ring_pages_);
        }
        // Capped: the frame count scales with cores x iters, and an uncapped reserve throws
        // bad_alloc with the region still pinned.
        constexpr uint64_t kMaxSamples = 4u << 20;
        std::vector<double> rt_us;
        rt_us.reserve(static_cast<size_t>(std::min<uint64_t>(total_frames_ - warmup_frames_, kMaxSamples)));
        uint64_t frames = 0;
        bool ok = true;
        std::string run_error;

        auto t0 = std::chrono::steady_clock::now();
        auto deadline = t0 + kStall;

        while (frames < total_frames_ && ok) {
            for (uint32_t c = 0; c < cores_; ++c) {
                while (core[c].published < iters_) {
                    DeliverTask t;
                    t.core = c;
                    t.slot = static_cast<uint32_t>(core[c].published % ring_pages_);
                    t.page_offset = rx_slot_offset(c, t.slot, page_);
                    t.page_bytes = page_;
                    t.length = payload_bytes_;
                    // publish() only advances bytes_sent -- the bytes are already in the
                    // ring, which is why this leg needs no copy and the pre-fill suffices.
                    // Word 0 per frame: publish() already issues a PCIe write, so one
                    // host store beside it is noise, and it makes a stale slot detectable.
                    if (verify_) {
                        *reinterpret_cast<uint32_t*>(region_base_ + t.page_offset) =
                            static_cast<uint32_t>(core[c].published);
                    }
                    if (!h2d_->publish(t)) {
                        break;
                    }
                    core[c].at[t.slot] = std::chrono::steady_clock::now();
                    ++core[c].published;
                }
            }

            for (uint32_t c = 0; c < cores_; ++c) {
                const uint32_t n = h2d_->drained(c);
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t slot = static_cast<uint32_t>(core[c].drained % ring_pages_);
                    ++frames;
                    if (frames == warmup_frames_) {
                        t0 = std::chrono::steady_clock::now();
                    }
                    if (frames > warmup_frames_) {
                        rt_us.push_back(us_since(core[c].at[slot]));
                    }
                    ++core[c].drained;
                }
                if (n != 0) {
                    deadline = std::chrono::steady_clock::now() + kStall;
                }
            }

            if (const std::string e = h2d_->first_error(); !e.empty()) {
                run_error = "h2d: " + e;
                ok = false;
            } else if (std::chrono::steady_clock::now() > deadline) {
                run_error =
                    "stalled at " + std::to_string(frames) + " of " + std::to_string(total_frames_) + " frames";
                ok = false;
            }
        }
        const double window_us = us_since(t0);

        // Only after the rings have drained: the kernel exits once it has seen `iters`
        // frames, so Finish() before that is the other unbounded wait.
        if (ok) {
            Finish(mesh_->mesh_command_queue());
        }

        // result[0] is the kernel's bad count, result[1] what it actually saw. Read even
        // with verify off: nothing else here confirms the device took every frame.
        uint64_t bad_total = 0;
        if (ok) {
            for (uint32_t i = 0; i < cores_; ++i) {
                std::vector<uint64_t> raw(1, 0);
                dist::read_l1_uint64s(
                    *mesh_, dist::MeshCoreCoord(dist::MeshCoordinate(0, 0), core_list_[i]), l1_.verify_addr, raw);
                const uint32_t bad = static_cast<uint32_t>(raw[0] & 0xFFFFFFFFu);
                const uint32_t seen = static_cast<uint32_t>(raw[0] >> 32);
                bad_total += bad;
                if (seen != iters_) {
                    run_error = "core " + std::to_string(i) + " saw " + std::to_string(seen) + " of " +
                                std::to_string(iters_) + " frames";
                    ok = false;
                    break;
                }
                if (bad != 0) {
                    run_error = "core " + std::to_string(i) + " rejected " + std::to_string(bad) + " frames";
                    ok = false;
                    break;
                }
            }
        }

        state.counters["frames"] = static_cast<double>(frames);
        state.counters["bad_frames"] = static_cast<double>(bad_total);
        if (ok && window_us > 0.0) {
            const double gb = static_cast<double>(total_frames_ - warmup_frames_) * payload_bytes_ / 1e9;
            state.counters["throughput_gbps"] = gb / (window_us / 1e6);
            // publish -> drained: this host pokes bytes_sent, the device pulls the page and
            // acks, and this host sees the ack. A full round trip on one clock.
            set_latency_counters(state, summarize_latency_us(rt_us, cycles_per_us_), rt_us.size());
        }
        if (!ok) {
            fail(state, run_error.empty() ? "h2d leg failed" : run_error);
        }
    }
}

BENCHMARK_REGISTER_F(H2DLegFixture, Bandwidth)
    // Named explicitly: a fixture benchmark is otherwise reported as Fixture/Case, and
    // analyze_hd_sockets.py takes the name up to the first slash as the benchmark key.
    ->Name("BM_H2DLegBandwidth")
    ->ArgsProduct({
        kPageSizes,   // page_size
        kCores,       // cores
        kRingPages,   // ring_pages
        kIterations,  // iters
        kWarmupPct,   // warmup_pct
        kVerify,      // verify
    })
    ->ArgNames({"page_size", "cores", "ring_pages", "iters", "warmup_pct", "verify"})
    ->UseRealTime()
    // The run is the measurement, so a repeat would only re-pay the provision, the pin and
    // the ring fill. --benchmark_repetitions still works.
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

}  // namespace

int main(int argc, char** argv) {
    // Before Initialize: the context takes MPI's own argv entries first, and this leg
    // refuses a world it cannot drive on its own.
    mh::DistributedContext::create(argc, argv);
    if (*mh::DistributedContext::get_current_world()->size() != 1) {
        std::fprintf(stderr, "error: this leg is local to one host; run it without mpirun or with `-n 1`\n");
        return 2;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return g_run_failed ? 1 : 0;
}
