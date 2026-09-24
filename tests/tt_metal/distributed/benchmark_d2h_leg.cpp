// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The device-to-host leg alone: one rank, no peer, no H2H.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "tt_metal/distributed/host_uva_frame.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace dist = tt::tt_metal::distributed;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

constexpr int kDeviceId = 0;

// `cores` must stay a single element: the region maps once per process.
const std::vector<int64_t> kPageSizes = {16384};
const std::vector<int64_t> kCores = {4};
const std::vector<int64_t> kRingPages = {8};
const std::vector<int64_t> kIterations = {20000};
const std::vector<int64_t> kWarmupPct = {10};
const std::vector<int64_t> kVerify = {0};

// Fail rather than spin: a kernel parked in socket_barrier makes Finish() unbounded.
constexpr auto kStall = std::chrono::seconds(30);

// SkipWithError does not set the exit status; main() returns this instead.
bool g_run_failed = false;

void fail(benchmark::State& state, const std::string& why) {
    g_run_failed = true;
    state.SkipWithError(why);
}

// Copied from benchmark_hd_sockets.cpp:159-190 and :518-528; those symbols are file-local.
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

LatencySummary summarize_latency_cycles(const std::vector<uint64_t>& cycles, double cycles_per_us) {
    if (cycles.empty() || cycles_per_us <= 0.0) {
        return {};
    }
    auto sorted = cycles;
    std::sort(sorted.begin(), sorted.end());
    double avg_c = 0.0;
    for (const uint64_t c : cycles) {
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

// `prefix` is the one addition: analyze_hd_sockets.py reads the unprefixed set.
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

// Pre-registered so a skipped case keeps the CSV shape.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["frames"] = 0;
    set_latency_counters(state, LatencySummary{}, 0);
    set_latency_counters(state, LatencySummary{}, 0, "slot_wait_");
}

// One bringup per process: Fixture::SetUp runs once per arg case.
struct DeviceFixture {
    std::shared_ptr<dist::MeshDevice> mesh_device;

    DeviceFixture() : mesh_device(dist::MeshDevice::create_unit_mesh(kDeviceId)) {}
};

DeviceFixture& get_device_fixture() {
    static DeviceFixture fixture;
    return fixture;
}

// As test_kernel_put.cpp:64-68 writes it.
uint32_t pattern_word(uint32_t core) {
    const uint32_t b = 0x40u + (core & 0x1Fu);
    return b | (b << 8) | (b << 16) | (b << 24);
}

class D2HLegFixture : public benchmark::Fixture {
public:
    // A skipped state never enters the loop, which is what makes this the gate.
    void SetUp(benchmark::State& state) override {
        page_bytes_ = static_cast<uint32_t>(state.range(0));
        cores_ = static_cast<uint32_t>(state.range(1));
        ring_pages_ = static_cast<uint32_t>(state.range(2));
        iters_ = static_cast<uint32_t>(state.range(3));
        verify_ = state.range(5) != 0;

        const uint32_t page = tt_uva_frame_page_size(page_bytes_);
        if (static_cast<uint64_t>(ring_pages_) * page > kArenaBytes) {
            fail(state,
                 "ring " + std::to_string(ring_pages_) + " x page " + std::to_string(page) + " B exceeds the " +
                     std::to_string(kArenaBytes >> 10) + " KiB arena; lower ring_pages or page_size");
            return;
        }

        mesh_ = get_device_fixture().mesh_device;
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

        // 64-bit: the 32-bit product wraps inside the range this accepts.
        warmup_iters_ = static_cast<uint32_t>(static_cast<uint64_t>(iters_) * state.range(4) / 100);
        total_frames_ = static_cast<uint64_t>(cores_) * iters_;
        // Per core, then scaled: matches the kernel's own warmup_iters stamp.
        warmup_frames_ = static_cast<uint64_t>(cores_) * warmup_iters_;
        cycles_per_us_ = dist::get_cycles_per_us(*mesh_);

        // The leg first, then the pin: D2HLeg MAP_FIXEDs its rings over the arenas.
        D2HLeg::Config dc;
        dc.cores = cores_;
        dc.grid_width = grid_width_;
        dc.payload_bytes = page_bytes_;
        dc.ring_pages = ring_pages_;
        dc.consumed_addr = 0;  // no far device, so nothing credits and tt_uva_sync() is unused
        HostRegion& region = HostRegion::storage();
        std::string err;
        try {
            // Kept for the verify path: reserved_base() is the only accessor for the
            // mapping's base, and the frame offsets the sink reports are relative to it.
            region_base_ = region.reserved_base(cores_);
            dc.alias_region_base = region_base_;
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
            region.provision(
                mesh_,
                /*chip=*/0,
                cores_,
                experimental::HostTopology{0, 1, 1},
                HostRegion::Grid{grid_width_, grid_height_});
        } catch (const std::exception& ex) {
            fail(state, std::string("host region unavailable: ") + ex.what());
            return;
        }
        if (const std::string e = region.verify_header(); !e.empty()) {
            fail(state, "region header check failed: " + e);
            return;
        }
    }

    // Unpin before the leg's destructor puts anonymous pages back over the arenas.
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
    // Sampled, not a full compare: this runs inside the drain spin.
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
    uint8_t* region_base_ = nullptr;
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
        // sig_addr 0 selects the unsignalled tt_uva_put: no receiver kernel.
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
            SetRuntimeArgs(program, sender, core_list_[i], {cfg[i], tt_uva_t6_global_selector(0, 0, i, 1), 0u});
        }
        dist::execute_program_on_device(*mesh_, dist::MeshCoordinate(0, 0), std::move(program));

        std::vector<uint32_t> pending(cores_, 0);
        std::vector<uint64_t> issue_cycles;
        std::vector<uint64_t> stall_cycles;
        // Capped: an uncapped reserve throws bad_alloc with the region still pinned.
        constexpr uint64_t kMaxSamples = 4u << 20;
        const size_t samples = static_cast<size_t>(std::min<uint64_t>(total_frames_ - warmup_frames_, kMaxSamples));
        issue_cycles.reserve(samples);
        stall_cycles.reserve(samples);
        uint64_t frames = 0;
        bool ok = true;
        std::string run_error;
        // Per core: poll() drains one core fully before moving on.
        std::vector<uint32_t> seen(cores_, 0);

        // Hoisted: built inside the loop this allocates on every pass of the drain spin.
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
            if (i >= warmup_iters_) {
                issue_cycles.push_back(tt_uva_frame_elapsed_issue(t.elapsed));
                stall_cycles.push_back(tt_uva_frame_elapsed_stall(t.elapsed));
            }
            return true;
        };

        auto deadline = std::chrono::steady_clock::now() + kStall;

        while (frames < total_frames_ && ok) {
            d2h_->poll(sink);

            // Batched per pass: each retire is a PCIe write.
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

        // How far the run got, not what the drain below sweeps up.
        const uint64_t frames_at_exit = frames;

        // The kernel runs on after every ok=false path and exits only once its last put is
        // retired. Drain -- discarding, the run already failed -- so it can reach that.
        if (!ok) {
            const D2HLeg::Sink discard = [&](const SendTask& t) {
                ++pending[t.core];
                ++frames;
                return true;
            };
            uint64_t drained = frames;
            auto drain_until = std::chrono::steady_clock::now() + kStall;
            while (frames < total_frames_ && std::chrono::steady_clock::now() < drain_until) {
                d2h_->poll(discard);
                for (uint32_t c = 0; c < cores_; ++c) {
                    if (pending[c] != 0) {
                        d2h_->retire(c, pending[c]);
                        pending[c] = 0;
                    }
                }
                // Bounded on progress, not on total time: a slow drain is still a drain.
                if (frames != drained) {
                    drained = frames;
                    drain_until = std::chrono::steady_clock::now() + kStall;
                }
            }
        }

        // Only after the FIFO has drained: Finish() before that is unbounded.
        if (frames >= total_frames_) {
            Finish(mesh_->mesh_command_queue());
        } else {
            // Nothing here can stop the kernel or close the shared device, and returning
            // runs ~D2HLeg and release() under pages it may still write. Leave them mapped.
            std::fprintf(
                stderr,
                "fatal: %s; the sender kernel did not drain within %llds, so the device may still be "
                "writing into the pinned region. Exiting without teardown -- reset the device before "
                "the next run.\n",
                run_error.c_str(),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(kStall).count()));
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }

        // The chip clock is global, so earliest begin and latest end bound one window.
        uint64_t begin = UINT64_MAX;
        uint64_t end = 0;
        if (ok) {
            for (uint32_t i = 0; i < cores_; ++i) {
                // 7 uint32 words straddle the uint64 lanes: read 4 and split them back.
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
                // w[5..6], not w[0..1]: the steady-state stamp.
                begin = std::min(begin, static_cast<uint64_t>(w[5]) | (static_cast<uint64_t>(w[6]) << 32));
                end = std::max(end, static_cast<uint64_t>(w[2]) | (static_cast<uint64_t>(w[3]) << 32));
            }
        }

        state.counters["frames"] = static_cast<double>(frames_at_exit);
        if (ok && end > begin) {
            const double secs = static_cast<double>(end - begin) / (cycles_per_us_ * 1e6);
            const double gb = static_cast<double>(total_frames_ - warmup_frames_) * page_bytes_ / 1e9;
            state.counters["throughput_gbps"] = gb / secs;
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
    // Named explicitly: analyze_hd_sockets.py keys on the name up to the first slash.
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
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

}  // namespace

int main(int argc, char** argv) {
    // Before Initialize: the context takes MPI's own argv entries first.
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
