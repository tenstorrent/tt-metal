// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The device-to-host leg alone, under google-benchmark: one rank, no peer, no H2H. The sink
// retires on sight, so a slot is freed by this host seeing the frame, not by a transport.
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

#include "tt_metal/distributed/host_d2h_leg.hpp"
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

// A stalled run must fail rather than spin: the kernel parks in socket_barrier when its
// pages go unacked, and Finish() would then never return.
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
// Metrics. Copied from benchmark_hd_sockets.cpp:159-190 and :518-528, which is not to be
// modified; the symbols there are file-local. `prefix` is the one addition -- see below.
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

// Zeroed rather than fatal on an empty input: a benchmark that measured nothing should
// report zeros and let SkipWithError carry the reason, not abort the process.
LatencySummary summarize_latency_cycles(const std::vector<uint64_t>& cycles, double cycles_per_us) {
    if (cycles.empty() || cycles_per_us <= 0.0) {
        return {};
    }
    auto sorted = cycles;
    std::sort(sorted.begin(), sorted.end());
    double avg_c = 0.0;
    for (auto c : cycles) {
        avg_c += static_cast<double>(c);
    }
    avg_c /= static_cast<double>(cycles.size());

    auto to_us = [&](double c) { return c / cycles_per_us; };
    return {
        .avg_us = to_us(avg_c),
        .min_us = to_us(static_cast<double>(sorted.front())),
        .max_us = to_us(static_cast<double>(sorted.back())),
        .p50_us = to_us(static_cast<double>(sorted[sorted.size() / 2])),
        .p99_us = to_us(static_cast<double>(sorted[(sorted.size() * 99) / 100])),
        .avg_cycles = avg_c,
        .min_cycles = sorted.front(),
        .max_cycles = sorted.back(),
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
    set_latency_counters(state, LatencySummary{}, 0);
    set_latency_counters(state, LatencySummary{}, 0, "slot_wait_");
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

// The pattern test_kernel_put.cpp:64-68 writes into every payload word but word 0.
uint32_t pattern_word(uint32_t core) {
    const uint32_t b = 0x40u + (core & 0x1Fu);
    return b | (b << 8) | (b << 16) | (b << 24);
}

class D2HLegFixture : public benchmark::Fixture {
public:
    // Everything that can refuse the run lives here, so the body is only the measurement.
    // A skipped state does not enter the loop, which is what makes this the gate.
    void SetUp(benchmark::State& state) override {
        page_bytes_ = static_cast<uint32_t>(state.range(0));
        cores_ = static_cast<uint32_t>(state.range(1));
        ring_pages_ = static_cast<uint32_t>(state.range(2));
        iters_ = static_cast<uint32_t>(state.range(3));
        verify_ = state.range(5) != 0;

        const uint32_t page = tt_uva_frame_page_size(page_bytes_);
        // RingAlias refuses an overlay wider than an arena, but only once the sockets are
        // built. Cheaper to say so here, in terms of the knob the caller turned.
        if (static_cast<uint64_t>(ring_pages_) * page > kArenaBytes) {
            fail(state,
                 "ring " + std::to_string(ring_pages_) + " x page " + std::to_string(page) + " B exceeds the " +
                     std::to_string(kArenaBytes >> 10) + " KiB arena; lower ring_pages or page_size");
            return;
        }

        mesh_ = get_device_fixture().mesh_device;
        // The leg reaches the device over PCIe, so a remote chip cannot serve it.
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
        l1_ = L1MapNew::compute(l1_base, static_cast<uint32_t>(device->l1_size_per_core()), page_bytes_, false);
        if (const std::string e = l1_.fits(page_bytes_); !e.empty()) {
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

        // 64-bit: iters and warmup_pct are both caller-supplied, and the 32-bit product
        // wraps inside the range this accepts, reporting a wrong rate as a clean pass.
        warmup_iters_ = static_cast<uint32_t>(static_cast<uint64_t>(iters_) * state.range(4) / 100);
        total_frames_ = static_cast<uint64_t>(cores_) * iters_;
        // Per core, then scaled: the kernel stamps steady state at its own warmup_iters, so
        // a differently-rounded host figure would divide the wrong count by that window.
        warmup_frames_ = static_cast<uint64_t>(cores_) * warmup_iters_;
        cycles_per_us_ = dist::get_cycles_per_us(*mesh_);

        // The leg first, then the pin: D2HLeg MAP_FIXEDs its rings over the arenas, and a
        // pin taken before that would go on naming the pages it replaced.
        D2HLeg::Config dc;
        dc.cores = cores_;
        dc.grid_width = grid_width_;
        dc.payload_bytes = page_bytes_;
        dc.ring_pages = ring_pages_;
        dc.consumed_addr = 0;  // no far device, so nothing credits and tt_uva_sync() is unused
        HostRegion& region = HostRegion::storage();
        std::string err;
        try {
            // Maps the region for these cores; the leg overlays its rings onto it below.
            dc.alias_region_base = region.reserved_base(cores_);
            d2h_ = D2HLeg::create(mesh_, dc, err);
        } catch (const std::exception& ex) {
            fail(state, std::string("host region unavailable: ") + ex.what());
            return;
        }
        if (!d2h_) {
            fail(state, "d2h bringup failed: " + err);
            return;
        }
        try {
            // Aliases the d2h rings with the region reserved above; this is what gets
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
        region_base_ = region.base();
    }

    // Unpin BEFORE the leg's destructor puts anonymous pages back over the arenas: the pin
    // must not still name the pages being swapped out. Runs on paths a return would skip.
    void TearDown(benchmark::State& state) override {
        (void)state;
        HostRegion& region = HostRegion::storage();
        if (region.is_provisioned()) {
            region.release();
        }
        d2h_.reset();
        mesh_.reset();
    }

protected:
    // Sampled, not a full compare: this runs inside the drain spin. Word 0 is the kernel's
    // iteration stamp; every other word carries the per-core pattern, so 0xA5 fill fails it.
    std::string check_frame(const SendTask& t, uint32_t expect_iter) const {
        const uint32_t words = t.length / static_cast<uint32_t>(sizeof(uint32_t));
        if (region_base_ == nullptr || words == 0) {
            return "core " + std::to_string(t.core) + " frame " + std::to_string(expect_iter) + " has no payload";
        }
        const uint32_t* const w = reinterpret_cast<const uint32_t*>(region_base_ + t.page_offset);
        if (w[0] != expect_iter) {
            return "core " + std::to_string(t.core) + " frame " + std::to_string(expect_iter) + " stamped " +
                   std::to_string(w[0]);
        }
        const uint32_t want = pattern_word(t.core);
        for (const uint32_t k : {1u, words / 2u, words - 1u}) {
            if (k != 0 && k < words && w[k] != want) {
                return "core " + std::to_string(t.core) + " frame " + std::to_string(expect_iter) + " word " +
                       std::to_string(k) + " is " + std::to_string(w[k]) + ", want " + std::to_string(want);
            }
        }
        return {};
    }

    std::shared_ptr<dist::MeshDevice> mesh_;
    std::unique_ptr<D2HLeg> d2h_;
    L1MapNew l1_{};
    CoreRangeSet cores_set_;
    std::vector<CoreCoord> core_list_;
    const uint8_t* region_base_ = nullptr;
    uint32_t page_bytes_ = 0;
    uint32_t cores_ = 0;
    uint32_t ring_pages_ = 0;
    uint32_t iters_ = 0;
    uint32_t warmup_iters_ = 0;
    uint32_t grid_width_ = 0;
    uint32_t grid_height_ = 0;
    uint64_t total_frames_ = 0;
    uint64_t warmup_frames_ = 0;
    double cycles_per_us_ = 0.0;
    bool verify_ = false;
};

BENCHMARK_DEFINE_F(D2HLegFixture, Bandwidth)(benchmark::State& state) {
    init_counters(state);
    const uint32_t page = tt_uva_frame_page_size(page_bytes_);

    for ([[maybe_unused]] auto _ : state) {
        Program program = CreateProgram();
        // sig_addr 0 selects the unsignalled tt_uva_put, so there is no receiver kernel and
        // nothing on the device waits on anything this host does except the FIFO ack.
        auto sender = CreateKernel(
            program,
            TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_put.cpp",
            cores_set_,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::NOC_0,
                .compile_args = {
                    l1_.stage_addr,
                    page,
                    l1_.payload_addr,
                    page_bytes_,
                    iters_,
                    grid_width_,
                    0u,
                    0u,
                    1u,
                    0u,
                    l1_.l1_base,
                    0u,
                    l1_.verify_addr,
                    warmup_iters_}});
        const std::vector<uint32_t> cfg = d2h_->config_addresses();
        for (uint32_t i = 0; i < cores_; ++i) {
            // Nothing routes on dst here, so it names this host: the bytes go to the
            // socket's own FIFO either way, and the trailer is only read locally.
            SetRuntimeArgs(program, sender, core_list_[i], {cfg[i], tt_uva_t6_global_selector(0, 0, i, 1), 0u});
        }
        dist::execute_program_on_device(*mesh_, dist::MeshCoordinate(0, 0), std::move(program));

        // This host still drains -- nothing moves otherwise -- but it times nothing. Every
        // number reported below was measured on the device and is only carried here.
        std::vector<uint32_t> pending(cores_, 0);
        std::vector<uint64_t> issue_cycles;
        std::vector<uint64_t> stall_cycles;
        // Capped: the frame count scales with cores x iters, and an uncapped reserve throws
        // bad_alloc with the region still pinned.
        constexpr uint64_t kMaxSamples = 4u << 20;
        const size_t samples = static_cast<size_t>(std::min<uint64_t>(total_frames_ - warmup_frames_, kMaxSamples));
        issue_cycles.reserve(samples);
        stall_cycles.reserve(samples);
        uint64_t frames = 0;
        bool ok = true;
        std::string run_error;
        // Per core, not against the global count: poll() drains one core fully before
        // moving on, so a global gate admits a fast core's ramp and drops steady state.
        std::vector<uint32_t> seen(cores_, 0);

        // Hoisted: the lambda captures more than the small buffer holds, so building it
        // inside the loop put an operator new/delete pair on every pass of the drain spin.
        const D2HLeg::Sink sink = [&](const SendTask& t) {
            const uint32_t i = seen[t.core]++;
            if (verify_ && ok) {
                if (const std::string e = check_frame(t, i); !e.empty()) {
                    run_error = e;
                    ok = false;
                    return false;
                }
            }
            ++pending[t.core];
            ++frames;
            // Matches the kernel's own `i == warmup_iters` stamp.
            if (i >= warmup_iters_) {
                issue_cycles.push_back(tt_uva_frame_elapsed_issue(t.elapsed));
                stall_cycles.push_back(tt_uva_frame_elapsed_stall(t.elapsed));
            }
            return true;
        };

        auto deadline = std::chrono::steady_clock::now() + kStall;

        while (frames < total_frames_ && ok) {
            d2h_->poll(sink);

            // Batched per pass, not per frame: each retire is a PCIe write to the device,
            // and one per frame would put host overhead inside the number being measured.
            for (uint32_t c = 0; c < cores_; ++c) {
                if (pending[c] != 0) {
                    d2h_->retire(c, pending[c]);
                    pending[c] = 0;
                    deadline = std::chrono::steady_clock::now() + kStall;
                }
            }
            if (const std::string e = d2h_->first_error(); !e.empty()) {
                run_error = "d2h: " + e;
                ok = false;
            } else if (std::chrono::steady_clock::now() > deadline) {
                run_error =
                    "stalled at " + std::to_string(frames) + " of " + std::to_string(total_frames_) + " frames";
                ok = false;
            }
        }

        // Only after the FIFO has drained: the kernel parks in socket_barrier until its
        // pages are acked, so Finish() before that is the other unbounded wait.
        if (ok) {
            Finish(mesh_->mesh_command_queue());
        }

        // The loop window each core timed on its own wall clock. That clock is chip-global,
        // so the earliest begin and the latest end bound one window covering every core.
        uint64_t begin = UINT64_MAX;
        uint64_t end = 0;
        if (ok) {
            for (uint32_t i = 0; i < cores_; ++i) {
                // 7 uint32 words, so the uint64 lanes straddle the fields: read 4 lanes of
                // the 64 B doorbell and split them back. verify_addr is 8-byte aligned.
                std::vector<uint64_t> raw(4, 0);
                dist::read_l1_uint64s(
                    *mesh_, dist::MeshCoreCoord(dist::MeshCoordinate(0, 0), core_list_[i]), l1_.verify_addr, raw);
                std::array<uint32_t, 8> w{};
                std::memcpy(w.data(), raw.data(), sizeof(w));
                if (w[4] != iters_) {
                    run_error = "core " + std::to_string(i) + " reported " + std::to_string(w[4]) + " of " +
                                std::to_string(iters_) + " iterations";
                    ok = false;
                    break;
                }
                // w[5..6], not w[0..1]: the steady-state stamp, so the ramp is outside.
                begin = std::min(begin, static_cast<uint64_t>(w[5]) | (static_cast<uint64_t>(w[6]) << 32));
                end = std::max(end, static_cast<uint64_t>(w[2]) | (static_cast<uint64_t>(w[3]) << 32));
            }
        }

        state.counters["frames"] = static_cast<double>(frames);
        if (ok && end > begin) {
            const double secs = static_cast<double>(end - begin) / (cycles_per_us_ * 1e6);
            const double gb = static_cast<double>(total_frames_ - warmup_frames_) * page_bytes_ / 1e9;
            state.counters["throughput_gbps"] = gb / secs;
            // Both stamped in stage(): issue is the payload write and its barrier, slot
            // wait is the block in socket_reserve_pages -- this host's turnaround.
            set_latency_counters(state, summarize_latency_cycles(issue_cycles, cycles_per_us_), issue_cycles.size());
            set_latency_counters(
                state, summarize_latency_cycles(stall_cycles, cycles_per_us_), stall_cycles.size(), "slot_wait_");
        }
        if (!ok) {
            fail(state, run_error.empty() ? "d2h leg failed" : run_error);
        }
    }
}

BENCHMARK_REGISTER_F(D2HLegFixture, Bandwidth)
    // Named explicitly: a fixture benchmark is otherwise reported as Fixture/Case, and
    // analyze_hd_sockets.py takes the name up to the first slash as the benchmark key.
    ->Name("BM_D2HLegBandwidth")
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
    // The run is the measurement and the numbers come off the device, so a repeat would
    // only re-pay the provision and the pin. --benchmark_repetitions still works.
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
