// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-device leg alone: this host fills the RX rings in place of a peer's RMA.
#include <algorithm>
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
#include "leg_benchmark_common.hpp"

#include "tt_metal/distributed/host_h2d_leg.hpp"
#include "tt_metal/distributed/host_l1_map.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>
#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace dist = tt::tt_metal::distributed;
namespace mh = tt::tt_metal::distributed::multihost;
using namespace leg_bench;

namespace {

constexpr int kDeviceId = 0;

// `cores` must stay a single element: the region maps once per process.
const std::vector<int64_t> kPageSizes = {16384};
const std::vector<int64_t> kCores = {4};
const std::vector<int64_t> kRingPages = {8};
const std::vector<int64_t> kIterations = {20000};
const std::vector<int64_t> kWarmupPct = {10};
const std::vector<int64_t> kVerify = {0};

// Pre-registered so a skipped case keeps the CSV shape.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["frames"] = 0;
    state.counters["bad_frames"] = 0;
    set_latency_counters(state, LatencySummary{}, 0);
}

struct CoreState {
    uint64_t published = 0;
    uint64_t drained = 0;
    std::vector<std::chrono::steady_clock::time_point> at;
};

class H2DLegFixture : public benchmark::Fixture {
public:
    // A skipped state never enters the loop, which is what makes this the gate.
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<uint32_t>(state.range(0));
        cores_ = static_cast<uint32_t>(state.range(1));
        ring_pages_ = static_cast<uint32_t>(state.range(2));
        iters_ = static_cast<uint32_t>(state.range(3));
        verify_ = state.range(5) != 0;
        page_ = tt_uva_frame_page_size(payload_bytes_);

        if (static_cast<uint64_t>(ring_pages_) * page_ > kArenaBytes) {
            fail(state,
                 "ring " + std::to_string(ring_pages_) + " x page " + std::to_string(page_) + " B exceeds the " +
                     std::to_string(kArenaBytes >> 10) + " KiB arena; lower ring_pages or page_size");
            return;
        }

        mesh_ = unit_mesh(kDeviceId);
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
        // Global, not per core: the same figure that starts the clock divides the bytes.
        warmup_frames_ = total_frames_ * static_cast<uint64_t>(state.range(4)) / 100;
        cycles_per_us_ = dist::get_cycles_per_us(*mesh_);

        // The leg first, then the pin: H2DLeg MAP_FIXEDs its rings over the RX arenas.
        H2DLeg::Config hc;
        hc.cores = cores_;
        hc.grid_width = grid_width_;
        hc.page_bytes = page_;
        hc.ring_pages = ring_pages_;
        HostRegion& region = HostRegion::storage();
        std::string err;
        try {
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
        fill_rings();
    }

    // Unpin before the leg's destructor puts anonymous pages back over the arenas.
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
    // Filled once: the device only reads these pages, so the timed loop carries no memcpy.
    void fill_rings() const {
        for (uint32_t c = 0; c < cores_; ++c) {
            for (uint32_t s = 0; s < ring_pages_; ++s) {
                uint8_t* const slot = region_base_ + rx_slot_offset(c, s, page_);
                // Whole page, so the payload-to-trailer gap is not left at 0xA5.
                std::memset(slot, 0, page_);
                // Word 0 is restamped per frame, in the publish loop.
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
                // The kernel exits on this word reaching `iters`, so every frame adds 1.
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
        // Capped: an uncapped reserve throws bad_alloc with the region still pinned.
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
                    // publish() only advances bytes_sent; the bytes are already in the ring.
                    // One host store beside its PCIe write makes a stale slot detectable.
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

        // Only after the rings have drained: Finish() before that is unbounded.
        if (ok) {
            Finish(mesh_->mesh_command_queue());
        }

        // result[0] is the bad count, result[1] what the kernel saw. Read even without verify.
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
            // publish -> drained: a full round trip on one clock.
            set_latency_counters(state, summarize_latency_us(rt_us, cycles_per_us_), rt_us.size());
        }
        if (!ok) {
            fail(state, run_error.empty() ? "h2d leg failed" : run_error);
        }
    }
}

BENCHMARK_REGISTER_F(H2DLegFixture, Bandwidth)
    // Named explicitly: analyze_hd_sockets.py keys on the name up to the first slash.
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
