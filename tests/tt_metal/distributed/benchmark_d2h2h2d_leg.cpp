// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The whole chain: rank 0's Tensix cores -> host -> host -> rank 1's. One poll() drives it.
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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "hd_socket_test_utils.hpp"
#include "leg_benchmark_common.hpp"

#include "tt_metal/distributed/d2h2h2d_socket.hpp"
#include "tt_metal/distributed/host_rdma_window.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace dist = tt::tt_metal::distributed;
namespace mh = tt::tt_metal::distributed::multihost;
using namespace leg_bench;

namespace {

constexpr int kDeviceId = 0;

// Both ranks build this identically, which is what keeps them on the same case list.
const std::vector<int64_t> kPageSizes = {4096, 16384, 65536, 262144};
const std::vector<int64_t> kCores = {1, 2, 4, 8, 16, 32, 64};
// Was never an arg: the socket defaulted to kNumAliasRingSlots == 1, so a volume run paid a
// full host-to-host credit round trip per frame. ring_pages x page must fit one arena.
const std::vector<int64_t> kRingPages = {1, 4, 8};
const std::vector<int64_t> kVolumeMiB = {1024, 4096, 20480};
const std::vector<int64_t> kPctSteady = {0, 10, 25};
const std::vector<int64_t> kVerify = {0, 1};
const std::vector<int64_t> kTiming = {1};

// Both ranks build this identically, so the socket reserves for the sweep's largest case and
// each case provisions only its own prefix -- reserved_base() maps once per process.
const uint32_t kReservedCores = static_cast<uint32_t>(*std::max_element(kCores.begin(), kCores.end()));

// Pre-registered so a skipped case keeps the CSV shape.
void init_counters(benchmark::State& state) {
    state.counters["throughput_gbps"] = 0;
    state.counters["push_gbps"] = 0;
    state.counters["device_bytes_per_cycle"] = 0;
    state.counters["frames"] = 0;
    state.counters["bad_cores"] = 0;
    for (const char* p : {"d2h_issue_", "d2h_stall_", "h2h_put_credit_", "h2d_publish_drained_"}) {
        set_latency_counters(state, LatencySummary{}, 0, p);
    }
}

// POD, so one all_gather puts both halves of the chain in front of the reporting rank.
struct RankReport {
    LatencySummary d2h_issue{};
    LatencySummary d2h_stall{};
    LatencySummary h2h_put_credit{};
    LatencySummary h2d_publish_drained{};
    double gbps = 0.0;
    double device_bytes_per_cycle = 0.0;
    uint64_t d2h_samples = 0;
    uint64_t h2h_samples = 0;
    uint64_t h2d_samples = 0;
    uint64_t frames = 0;
    uint64_t bad_cores = 0;
};

class D2H2H2DFixture : public benchmark::Fixture {
public:
    // Collective: create() overlays the rings, pins the region and builds the window.
    void SetUp(benchmark::State& state) override {
        payload_bytes_ = static_cast<uint32_t>(state.range(0));
        cores_ = static_cast<uint32_t>(state.range(1));
        ring_pages_ = static_cast<uint32_t>(state.range(2));
        const uint64_t volume = static_cast<uint64_t>(state.range(3)) << 20;
        const uint32_t pct_steady = static_cast<uint32_t>(state.range(4));
        verify_ = state.range(5) != 0;
        timing_ = state.range(6) != 0;

        // Both ranks compute this identically, so a refusal here is collective by construction.
        const uint64_t ring_bytes = static_cast<uint64_t>(ring_pages_) * tt_uva_frame_page_size(payload_bytes_);
        if (ring_bytes > kArenaBytes) {
            fail(state,
                 "ring_pages x page (" + std::to_string(ring_bytes) + " B) exceeds the " +
                     std::to_string(kArenaBytes >> 10) + " KiB arena");
            return;
        }

        const mh::ContextPtr world = mh::DistributedContext::get_current_world();
        rank_ = static_cast<uint32_t>(*world->rank());
        ranks_ = static_cast<uint32_t>(*world->size());
        sending_ = rank_ == 0;

        // Padded by pct_steady, so the clock starts once the ramp is behind.
        const uint64_t measured = std::max<uint64_t>(1, volume / (static_cast<uint64_t>(cores_) * payload_bytes_));
        iters_ = static_cast<uint32_t>(measured + measured * pct_steady / 100);
        warmup_iters_ = static_cast<uint32_t>(iters_ - measured);
        msgs_ = static_cast<uint64_t>(cores_) * iters_;
        warmup_msgs_ = static_cast<uint64_t>(cores_) * warmup_iters_;

        mesh_ = split_unit_mesh(kDeviceId);
        IDevice* const device = mesh_->get_devices().front();
        const CoreCoord grid = device->compute_with_storage_grid_size();
        grid_width_ = static_cast<uint32_t>(grid.x);

        D2H2H2DSocket::Config cfg;
        cfg.topo = experimental::HostTopology{rank_, ranks_, 1};
        cfg.chip = 0;
        cfg.cores = cores_;
        cfg.reserved_cores = kReservedCores;
        cfg.grid_width = grid_width_;
        cfg.grid_height = static_cast<uint32_t>(grid.y);
        cfg.payload_bytes = payload_bytes_;
        cfg.ring_pages = ring_pages_;
        cfg.collect_timing = timing_;

        std::string err;
        if (cores_ == 0 || cores_ > grid_width_ * cfg.grid_height || cores_ > kProvisionedCores) {
            err = "cores " + std::to_string(cores_) + " does not fit this grid";
        } else {
            sock_ = D2H2H2DSocket::create(mesh_, device, cfg, err);
        }
        // One agree for the whole bringup: both ranks leave here with the same verdict.
        if (!RdmaWindow::agree(sock_ != nullptr, err)) {
            fail(state, "rank " + std::to_string(rank_) + ": socket bringup failed: " + err);
            return;
        }

        core_list_.clear();
        cores_set_ = CoreRangeSet();
        for (uint32_t i = 0; i < cores_; ++i) {
            const CoreCoord c{i % grid_width_, i / grid_width_};
            core_list_.push_back(c);
            cores_set_ = cores_set_.merge(CoreRangeSet(CoreRange(c, c)));
        }
        cycles_per_us_ = dist::get_cycles_per_us(*mesh_);
    }

    // The socket unwinds the pin and the window in the order they were built.
    void TearDown(benchmark::State& state) override {
        (void)state;
        sock_.reset();
        mesh_.reset();
    }

protected:
    // Both kernels everywhere: the L1 map must be identical, and each side enables its half.
    Program build_program() const {
        const L1MapNew& l1 = sock_->l1();
        Program program = CreateProgram();

        const std::vector<uint32_t> send_cfg = sock_->d2h().config_addresses();
        auto sender = CreateKernel(
            program,
            TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_put.cpp",
            cores_set_,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::NOC_0,
                .compile_args = {
                    l1.stage_addr,
                    sock_->d2h().page_size(),
                    l1.payload_addr,
                    payload_bytes_,
                    sending_ ? iters_ : 0u,
                    grid_width_,
                    rank_,
                    0u,
                    1u,
                    l1.consumed_addr,
                    l1.l1_base,
                    l1.dest_word_addr,
                    // Only the sender writes the stamp block; the signal kernel owns it here.
                    sending_ ? l1.verify_addr : 0u,
                    warmup_iters_}});
        for (uint32_t i = 0; i < cores_; ++i) {
            // Off l1_base, which both hosts share: this is the offset land_one() resolves,
            // so it must name the buffer the receiver kernel reads, not 0.
            SetRuntimeArgs(
                program,
                sender,
                core_list_[i],
                {send_cfg[i], tt_uva_t6_global_selector(1 - rank_, 0, i, 1), l1.deliver_addr - l1.l1_base});
        }

        const std::vector<uint32_t> recv_cfg = sock_->h2d().config_addresses();
        auto receiver = CreateKernel(
            program,
            TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_signal.cpp",
            cores_set_,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_1,
                // dest_word is the app's signal word: ADD-1 per frame, `iters` for the run.
                .compile_args = {
                    l1.deliver_addr,
                    sock_->d2h().page_size(),
                    l1.dest_word_addr,
                    iters_,
                    l1.l1_base,
                    l1.l1_size,
                    payload_bytes_,
                    verify_ ? 1u : 0u,
                    l1.verify_addr}});
        for (uint32_t i = 0; i < cores_; ++i) {
            SetRuntimeArgs(program, receiver, core_list_[i], {recv_cfg[i], sending_ ? 0u : 1u});
        }
        return program;
    }

    std::shared_ptr<dist::MeshDevice> mesh_;
    std::unique_ptr<D2H2H2DSocket> sock_;
    CoreRangeSet cores_set_;
    std::vector<CoreCoord> core_list_;
    uint32_t payload_bytes_ = 0;
    uint32_t cores_ = 0;
    uint32_t ring_pages_ = 0;
    uint32_t iters_ = 0;
    uint32_t warmup_iters_ = 0;
    uint32_t grid_width_ = 0;
    uint32_t rank_ = 0;
    uint32_t ranks_ = 0;
    uint64_t msgs_ = 0;
    uint64_t warmup_msgs_ = 0;
    double cycles_per_us_ = 0.0;
    bool sending_ = false;
    bool verify_ = false;
    bool timing_ = false;
};

BENCHMARK_DEFINE_F(D2H2H2DFixture, Volume)(benchmark::State& state) {
    init_counters(state);

    for ([[maybe_unused]] auto _ : state) {
        const L1MapNew& l1 = sock_->l1();
        dist::MeshWorkload workload;
        workload.add_program(dist::MeshCoordinateRange(mesh_->shape()), build_program());
        // Non-blocking, so the poll loop below can feed the receiver to `iters`.
        EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/false);

        // `retired`, not `sent`: stopping at QUEUED leaves the tail un-put.
        const auto done = [&] {
            const auto& c = sock_->counters();
            return sending_ ? c.retired >= msgs_ : c.drained >= msgs_;
        };
        auto deadline = std::chrono::steady_clock::now() + kStall;
        uint64_t last_moved = 0;
        bool stalled = false;
        bool timing = false;
        std::chrono::steady_clock::time_point t0;

        while (!done() && !sock_->failed()) {
            sock_->poll();
            const auto& p = sock_->counters();
            if (!timing && (sending_ ? p.retired : p.drained) >= warmup_msgs_) {
                timing = true;
                t0 = std::chrono::steady_clock::now();
            }
            const uint64_t moved = p.sent + p.retired + p.received + p.drained;
            if (moved != last_moved) {
                last_moved = moved;
                deadline = std::chrono::steady_clock::now() + kStall;
            } else if (std::chrono::steady_clock::now() > deadline) {
                stalled = true;
                break;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();

        // Before the collectives below: done() is asymmetric, so an unagreed verdict hangs.
        std::string err = stalled ? "rank " + std::to_string(rank_) + ": stalled" : sock_->first_error();
        if (!RdmaWindow::agree(!sock_->failed() && !stalled && done(), err)) {
            fail(state, err.empty() ? "d2h2h2d chain failed" : err);
            return;
        }

        Finish(mesh_->mesh_command_queue());
        if (const std::string be = sock_->barrier(); !be.empty()) {
            fail(state, "barrier: " + be);
            return;
        }

        RankReport local{};
        local.frames = msgs_;
        if (timing) {
            const double secs = std::chrono::duration<double>(t1 - t0).count();
            const double gb = static_cast<double>(msgs_ - warmup_msgs_) * payload_bytes_ / 1e9;
            local.gbps = secs > 0.0 ? gb / secs : 0.0;
        }

        // The sending kernel stamps its loop window into L1, backpressure included.
        if (sending_ && timing_) {
            uint64_t begin = UINT64_MAX;
            uint64_t end = 0;
            for (uint32_t i = 0; i < cores_; ++i) {
                // 7 uint32 words straddle the uint64 lanes: read 4 and split them back.
                std::vector<uint64_t> raw(4, 0);
                dist::read_l1_uint64s(
                    *mesh_, dist::MeshCoreCoord(dist::MeshCoordinate(0, 0), core_list_[i]), l1.verify_addr, raw);
                std::array<uint32_t, 8> w{};
                std::memcpy(w.data(), raw.data(), sizeof(w));
                begin = std::min(begin, static_cast<uint64_t>(w[5]) | (static_cast<uint64_t>(w[6]) << 32));
                end = std::max(end, static_cast<uint64_t>(w[2]) | (static_cast<uint64_t>(w[3]) << 32));
            }
            if (end > begin) {
                local.device_bytes_per_cycle = static_cast<double>(msgs_ - warmup_msgs_) *
                                               static_cast<double>(payload_bytes_) / static_cast<double>(end - begin);
            }
        }

        // result[0] is the corrupt count, result[1] what landed. Read even without verify.
        if (!sending_) {
            for (uint32_t i = 0; i < cores_; ++i) {
                std::vector<uint64_t> raw(1, 0);
                dist::read_l1_uint64s(
                    *mesh_, dist::MeshCoreCoord(dist::MeshCoordinate(0, 0), core_list_[i]), l1.verify_addr, raw);
                const uint32_t bad = static_cast<uint32_t>(raw[0] & 0xFFFFFFFFu);
                const uint32_t landed = static_cast<uint32_t>(raw[0] >> 32);
                if (bad != 0 || landed != iters_) {
                    ++local.bad_cores;
                }
            }
        }

        if (timing_) {
            const D2H2H2DSocket::Timing& t = sock_->timing();
            local.d2h_issue = summarize_latency_cycles(t.d2h_issue_cycles, cycles_per_us_);
            local.d2h_stall = summarize_latency_cycles(t.d2h_stall_cycles, cycles_per_us_);
            // 0.0: this leg is stamped on the host clock with no device in the path, so
            // back-filling cycle columns from the Tensix rate would invent a quantity.
            local.h2h_put_credit = summarize_latency_ns(t.h2h_put_to_credit_ns, 0.0);
            local.h2d_publish_drained = summarize_latency_ns(t.h2d_publish_to_drained_ns, cycles_per_us_);
            local.d2h_samples = t.d2h_issue_cycles.size();
            local.h2h_samples = t.h2h_put_to_credit_ns.size();
            local.h2d_samples = t.h2d_publish_to_drained_ns.size();
        }

        // d2h and h2h are the sender's legs, h2d the receiver's, so one row needs both.
        std::array<RankReport, 2> all{};
        mh::DistributedContext::get_current_world()->all_gather(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&local), sizeof(local)),
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(all.data()), all.size() * sizeof(RankReport)));

        if (rank_ != 0) {
            continue;
        }
        const RankReport& tx = all[0];
        const RankReport& rx = all[1];
        // The receiver's rate is end-to-end: drained means the frame crossed all three legs.
        state.counters["throughput_gbps"] = rx.gbps;
        state.counters["push_gbps"] = tx.gbps;
        state.counters["device_bytes_per_cycle"] = tx.device_bytes_per_cycle;
        state.counters["frames"] = static_cast<double>(tx.frames);
        state.counters["bad_cores"] = static_cast<double>(rx.bad_cores);
        set_latency_counters(state, tx.d2h_issue, tx.d2h_samples, "d2h_issue_");
        set_latency_counters(state, tx.d2h_stall, tx.d2h_samples, "d2h_stall_");
        set_latency_counters(state, tx.h2h_put_credit, tx.h2h_samples, "h2h_put_credit_");
        set_latency_counters(state, rx.h2d_publish_drained, rx.h2d_samples, "h2d_publish_drained_");
        if (rx.bad_cores != 0) {
            fail(state, std::to_string(rx.bad_cores) + " receiving cores did not land the run intact");
        }
    }
}

BENCHMARK_REGISTER_F(D2H2H2DFixture, Volume)
    // Named explicitly: analyze_hd_sockets.py keys on the name up to the first slash.
    ->Name("BM_D2H2H2DVolume")
    ->ArgsProduct({
        kPageSizes,  // page_size
        kCores,      // cores
        kRingPages,  // ring_pages
        kVolumeMiB,  // volume_mib
        kPctSteady,  // pct_steady
        kVerify,     // verify
        kTiming,     // timing
    })
    ->ArgNames({"page_size", "cores", "ring_pages", "volume_mib", "pct_steady", "verify", "timing"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

}  // namespace

int main(int argc, char** argv) {
    // Before Initialize: the context takes MPI's own argv entries first.
    mh::DistributedContext::create(argc, argv);
    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    const uint32_t rank = static_cast<uint32_t>(*world->rank());
    if (*world->size() != 2) {
        if (rank == 0) {
            std::fprintf(stderr, "error: this chain needs exactly 2 ranks; launch with `mpirun -n 2`\n");
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
