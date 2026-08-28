// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: dispatches kernels that emit 10 differently-named DeviceZoneScopedN zones
// (with increasing durations) on ALL 5 RISCs of a small core grid, then closes. It does NOT drive the drainer
// itself -- run it with TT_METAL_STREAMING_PROFILER=1 so the PerfDebugProfiler boots at MeshDevice bring-up
// and captures these zones (that one switch also implies TT_METAL_DEVICE_PROFILER, arming the kernels'
// markers). The Tracy sink is opt-in: add TT_METAL_STREAMING_PROFILER_TRACY=1 to verify with a connected
// tracy-capture. Grid + iteration count are overridable via argv.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>

// --clkprobe only: reads a tile debug register straight over NoC, which is not a public-API operation.
#include <chrono>
#include <thread>
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/tt_cluster.hpp"

// --empty only: registers a stats consumer on the streaming profiler to measure the profiler's own
// per-zone overhead from the captured zones themselves.
#include <algorithm>
#include <mutex>
#include "tools/profiler/perf_debug_consumer.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// RISCV_DEBUG_REG_WALL_CLOCK_L/H. Reading L atomically LATCHES H, so read L then H.
constexpr uint64_t kWallClockL = 0xFFB121F0ULL;
constexpr uint64_t kWallClockH = 0xFFB121F8ULL;

std::string fmt_label(const std::string& what, const CoreCoord& virt) {
    return what + " v(" + std::to_string(virt.x) + "," + std::to_string(virt.y) + ")";
}

uint64_t read_wall_clock(tt::Cluster& cluster, const tt_cxy_pair& target) {
    uint32_t lo = 0, hi = 0;
    cluster.read_reg(&lo, target, kWallClockL);  // latches H
    cluster.read_reg(&hi, target, kWallClockH);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// STEP 1 of the DRAM-core clock-anchor fix: prove RISCV_DEBUG_REG_WALL_CLOCK is readable on a DRAM tile.
// It is documented as a TENSIX debug register, and nothing had confirmed a DRAM tile answers at all. Prints,
// per core: the raw counter, whether it ADVANCES over a known wall-clock interval, the implied MHz (should be
// ~aiclk), and -- the number the fix actually needs -- the DRAM-core-minus-worker offset at a common instant.
//
// Deliberately does NOT boot a drainer: the point is to isolate the register read from everything else, so a
// hang here indicts the read and nothing more.
void clock_probe(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    IDevice* device = mesh_device->get_devices().front();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    const double aiclk = cluster.get_device_aiclk(chip);

    struct Target {
        std::string label;
        tt_cxy_pair core;
    };
    std::vector<Target> targets;

    // Worker (Tensix) reference: the core sync_device_clock() actually uses today, core_virt[0].
    const CoreCoord w = device->virtual_core_from_logical_core(CoreCoord{0, 0}, CoreType::WORKER);
    targets.push_back(Target{fmt_label("WORKER", w), tt_cxy_pair(chip, w)});

    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    for (uint32_t bank = 0; bank < nbanks; bank++) {
        const CoreCoord lg = mesh_device->impl().pick_unused_dram_logical_core(bank);
        const CoreCoord dv = device->virtual_core_from_logical_core(lg, CoreType::DRAM);
        targets.push_back(Target{fmt_label("DRAM bank " + std::to_string(bank), dv), tt_cxy_pair(chip, dv)});
    }

    printf("[clkprobe] chip %u  aiclk %.1f MHz  %u DRAM views\n", chip, aiclk, nbanks);
    printf(
        "[clkprobe] reading RISCV_DEBUG_REG_WALL_CLOCK (0x%llx) on %zu cores\n",
        (unsigned long long)kWallClockL,
        targets.size());
    fflush(stdout);

    // Pass 1: paired reads, worker first then each DRAM core, so the offset is measured against a nearby
    // worker sample rather than one taken seconds earlier.
    std::vector<uint64_t> t0(targets.size()), t1(targets.size());
    for (size_t i = 0; i < targets.size(); i++) {
        printf("[clkprobe]   reading %s ...\n", targets[i].label.c_str());
        fflush(stdout);  // unbuffered: if the read HANGS, the log still names the core it hung on
        t0[i] = read_wall_clock(cluster, targets[i].core);
        printf(
            "[clkprobe]   %s -> 0x%016llx (%llu)\n",
            targets[i].label.c_str(),
            (unsigned long long)t0[i],
            (unsigned long long)t0[i]);
        fflush(stdout);
    }

    constexpr int kSleepMs = 500;
    const auto h0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
    for (size_t i = 0; i < targets.size(); i++) {
        t1[i] = read_wall_clock(cluster, targets[i].core);
    }
    const auto h1 = std::chrono::steady_clock::now();
    const double host_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(h1 - h0).count();

    printf("\n[clkprobe] %-22s %20s %20s %14s %12s\n", "core", "t0 (cycles)", "t1 (cycles)", "delta", "implied MHz");
    for (size_t i = 0; i < targets.size(); i++) {
        const uint64_t d = t1[i] - t0[i];
        printf(
            "[clkprobe] %-22s %20llu %20llu %14llu %12.1f\n",
            targets[i].label.c_str(),
            (unsigned long long)t0[i],
            (unsigned long long)t1[i],
            (unsigned long long)d,
            (double)d / host_ns * 1000.0);
    }
    // THE quantity the host never measured: drisc_wall - tensix_wall at a common instant.
    printf("\n[clkprobe] offset vs WORKER at a common instant (cycles, and ns at aiclk):\n");
    for (size_t i = 1; i < targets.size(); i++) {
        const int64_t off = (int64_t)t0[i] - (int64_t)t0[0];
        printf(
            "[clkprobe]   %-22s %+20lld cycles  %+15.3f ms\n",
            targets[i].label.c_str(),
            (long long)off,
            (double)off / aiclk / 1000.0);
    }
    fflush(stdout);
}

// ---- --empty mode: profiler self-overhead measurement --------------------------------------------
//
// The kernels emit 10 fully unrolled EMPTY zones back-to-back per iteration (ZONE_MODE=2), so the
// captured stream measures the profiler itself. Per lane, sorted by zone start:
//   DURATION = end - start of one empty zone = open's clock read -> close's clock read around nothing
//              (the close's ring room check + the wall-clock read latency);
//   GAP      = next.start - this.end between two adjacent empty zones = the close's post-clock work
//              (sticky check + 3 ring stores + fence + tail publish) + the next open's clock read.
//   DURATION + GAP = the full cost one zone adds to a kernel at max rate.
// Negative gaps are NESTED pairs (the "<RISC>-KERNEL" wrapper zone contains everything) and are
// dropped; the loop back-edge inflates one gap in ten by a couple of cycles, which the median ignores.
struct EmptyZoneStats {
    std::mutex mu;  // register/unregister lifetime only; batches arrive on a single consumer thread
    // Per (dev, lane): (start, duration) of every Zone record seen.
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> lanes;
    double frequency_ghz = 0.0;

    void operator()(const perf_debug::PerfDebugRecordBatch& batch) {
        std::lock_guard<std::mutex> lk(mu);
        if (frequency_ghz == 0.0 && !batch.context->devices.empty()) {
            frequency_ghz = batch.context->devices[0].frequency_ghz;
        }
        for (const auto& r : batch.records) {
            if (r.meta.type != perf_debug::PerfDebugRecType::Zone) {
                continue;
            }
            lanes[(r.meta.dev << 10) | r.meta.lane].push_back({r.data.zone.start, r.data.zone.duration});
        }
    }

    static uint64_t median(std::vector<uint64_t>& v) {
        if (v.empty()) {
            return 0;
        }
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

    void report() {
        std::lock_guard<std::mutex> lk(mu);
        const double ghz = frequency_ghz > 0.0 ? frequency_ghz : 1.35;
        // Aggregate by RISC index (lane % 5): 0 = BRISC, 1 = NCRISC, 2..4 = TRISC0..2.
        static const char* kRisc[5] = {"BRISC ", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
        printf(
            "\n[empty-zone overhead] per-RISC, cycles @ %.4f GHz (median [mean]); duration = in-zone "
            "(room check + clock read), gap = close+reopen (ring stores + publish + clock read)\n",
            ghz);
        printf(
            "[empty-zone overhead] %-7s %8s %22s %22s %22s\n",
            "risc",
            "zones",
            "duration cyc (ns)",
            "gap cyc (ns)",
            "dur+gap cyc (ns)");
        for (uint32_t risc = 0; risc < 5; risc++) {
            std::vector<uint64_t> durs, gaps;
            uint64_t dur_sum = 0, gap_sum = 0;
            for (auto& [key, zones] : lanes) {
                if ((key & 0x3FF) % 5 != risc || zones.size() < 2) {
                    continue;
                }
                std::sort(zones.begin(), zones.end());
                for (size_t i = 0; i < zones.size(); i++) {
                    // The kernel-wrapper zone contains the whole run; its duration would dwarf the
                    // stats. It is exactly the zone whose END is past the NEXT zone's start (nested),
                    // which the gap filter below already identifies -- for durations, drop the single
                    // largest per lane instead (the wrapper), cheap and exact for this workload.
                    durs.push_back(zones[i].second);
                    if (i + 1 < zones.size()) {
                        const uint64_t end = zones[i].first + zones[i].second;
                        if (zones[i + 1].first >= end) {
                            gaps.push_back(zones[i + 1].first - end);
                        }
                    }
                }
                // Drop this lane's wrapper duration (the max).
                if (!durs.empty()) {
                    auto mx = std::max_element(durs.begin(), durs.end());
                    durs.erase(mx);
                }
            }
            if (durs.empty()) {
                continue;
            }
            for (uint64_t d : durs) {
                dur_sum += d;
            }
            for (uint64_t g : gaps) {
                gap_sum += g;
            }
            const uint64_t dmed = median(durs), gmed = median(gaps);
            const double dmean = durs.empty() ? 0.0 : (double)dur_sum / durs.size();
            const double gmean = gaps.empty() ? 0.0 : (double)gap_sum / gaps.size();
            printf(
                "[empty-zone overhead] %-7s %8zu %9llu (%5.1f) [%6.1f] %9llu (%5.1f) [%6.1f] %9llu (%5.1f)\n",
                kRisc[risc],
                durs.size(),
                (unsigned long long)dmed,
                dmed / ghz,
                dmean,
                (unsigned long long)gmed,
                gmed / ghz,
                gmean,
                (unsigned long long)(dmed + gmed),
                (dmed + gmed) / ghz);
        }
        fflush(stdout);
    }
};

}  // namespace

int main(int argc, char** argv) {
    // --delay is the KNEE knob: uniform nop-iterations per zone, the SAME unit as the standalone drain harness
    // --proddelay, so 0 means MAX RATE (no spin) exactly as it does there. Smaller = higher marker rate.
    // Omitting --delay entirely selects the graduated ~1..100 us wall-clock durations, which is the right
    // default for a representative capture -- graduated is a separate MODE, not a magic --delay value.
    uint32_t gx = 2, gy = 2, n_iters = 50, zone_cyc = 0;  // small grid + modest iters keep the run quick
    bool knee_mode = false;                               // set by --delay, including --delay 0
    bool clkprobe = false;                                // --clkprobe 1: read wall clocks and exit, no workload
    uint32_t emit_markers = 0;  // --markers 1: emit the point-marker trio (Flag/Data/Iter) per iteration.
                                // OFF by default: it exercises every wire shape (PP_EVENT has no other
                                // emitter) but costs ~21% of wire volume, which skews knee/onset numbers.
    uint32_t all_devices = 0;   // --all-devices 1: open the FULL system mesh instead of a unit mesh on
                                // device 0. The workload is already a MeshWorkload over the mesh own
                                // shape, so this alone fans it out to every device -- which is what makes
                                // the capture exercise the cross-device wall-clock sync.
    uint32_t empty_mode = 0;    // --empty 1: unrolled EMPTY zones + stats consumer -> profiler self-overhead
                                // --empty 2: same, plus ONE extra wall-clock read pair per zone BODY -- the
                                //            duration delta vs --empty 1 prices read_wall_clock itself
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        uint32_t v = (uint32_t)std::strtoul(argv[i + 1], nullptr, 10);
        if (a == "--clkprobe") {
            clkprobe = v != 0;
        } else if (a == "--gx") {
            gx = v;
        } else if (a == "--gy") {
            gy = v;
        } else if (a == "--iters") {
            n_iters = v;
        } else if (a == "--delay") {
            zone_cyc = v;
            knee_mode = true;  // NOT `zone_cyc != 0`: --delay 0 is a real knee point (max rate)
        } else if (a == "--empty") {
            empty_mode = v;
        } else if (a == "--markers") {
            emit_markers = v;
        } else if (a == "--all-devices") {
            all_devices = v;
        }
    }

    // --empty: register the overhead stats consumer BEFORE device bring-up, so the profiler attaches
    // it at capture start (registration is process-wide and time-independent, but earliest is safest).
    auto empty_stats = std::make_shared<EmptyZoneStats>();
    perf_debug::PerfDebugConsumerHandle empty_handle = 0;
    if (empty_mode != 0) {
        empty_handle = perf_debug::register_consumer(
            "empty-zone-overhead", [empty_stats](const perf_debug::PerfDebugRecordBatch& b) { (*empty_stats)(b); });
    }

    // TT_METAL_SLOW_DISPATCH_MODE=1 takes the whole run off the command queue. Needed by the Tensix-BRISC
    // drainer control (TT_METAL_PERF_DEBUG_DRAIN_TENSIX), which parks a resident program on a worker core --
    // something fast dispatch will not allow. The workload itself is unchanged; only how it is launched is.
    const char* sd = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const bool slow_dispatch = sd != nullptr && *sd != '\0' && *sd != '0';

    // TT_METAL_PERF_DEBUG_NUM_CQS selects the hardware command-queue count. It exists to test whether the
    // DRISC PCIe hang scales with how much dispatch traffic shares the PCIe tile with the drainer's egress:
    // each CQ adds its own completion queue and its own dispatch cores driving that tile. Default 1 matches
    // every measurement taken before this knob existed, so leaving it unset changes nothing.
    const char* nq = std::getenv("TT_METAL_PERF_DEBUG_NUM_CQS");
    const size_t num_cqs = (nq != nullptr && *nq != '\0') ? (size_t)std::strtoul(nq, nullptr, 10) : 1;

    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    if (all_devices != 0) {
        const auto shape = distributed::SystemMesh::instance().shape();
        printf("[perf-debug zones]   --all-devices: opening system mesh %ux%u\n", (unsigned)shape[0], (unsigned)shape[1]);
        mesh_device = distributed::MeshDevice::create(
            distributed::MeshDeviceConfig(shape), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, num_cqs);
    } else {
        mesh_device = distributed::MeshDevice::create_unit_mesh(
            device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, num_cqs);
    }
    if (clkprobe) {
        clock_probe(mesh_device);
        mesh_device->close();
        return 0;
    }
    Program program = CreateProgram();

    // Clamp the requested grid to the device's compute grid; --gx 0 / --gy 0 (or an over-large value)
    // means "use the full grid". Passing a CoreRange past the grid would throw.
    CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (gx == 0 || gx > grid.x) {
        gx = grid.x;
    }
    if (gy == 0 || gy > grid.y) {
        gy = grid.y;
    }
    CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
    std::map<std::string, std::string> defs{
        {"N_ITERS", std::to_string(n_iters) + "u"},
        {"ZONE_MODE", empty_mode >= 2 ? "3" : (empty_mode != 0 ? "2" : (knee_mode ? "1" : "0"))},
        {"EMIT_MARKERS", emit_markers != 0 ? "1" : "0"},
        {"ZONE_CYC", std::to_string(zone_cyc) + "u"}};
    const std::string kdir = "tt_metal/programming_examples/profiler/test_perf_debug_zones/kernels/";

    // BRISC (RISCV_0) + NCRISC (RISCV_1): the data-movement zone kernel (tags BR_/NC_).
    CreateKernel(
        program,
        kdir + "zones_dm.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
    CreateKernel(
        program,
        kdir + "zones_dm.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defs});
    // TRISC0/1/2: the compute zone kernel (tags T0_/T1_/T2_).
    CreateKernel(program, kdir + "zones_compute.cpp", cores, ComputeConfig{.defines = defs});

    // Report the offered load so a knee sweep is self-documenting: 2 markers per zone (START+END), 10 zones
    // per iteration, 5 RISCs per core. Rate uses the ~1.35 GHz boosted aiclk the spin loop counts against.
    const uint32_t lanes = gx * gy * 5;
    const uint64_t markers = (uint64_t)lanes * 10ull * 2ull * (uint64_t)n_iters;
    printf(
        "[perf-debug zones] dispatching %ux%u cores x 5 RISCs x 10 named zones x %u iters\n"
        "[perf-debug zones]   lanes=%u  total markers=%llu  --delay=%u (%s)\n",
        gx,
        gy,
        n_iters,
        lanes,
        (unsigned long long)markers,
        zone_cyc,
        knee_mode ? "uniform nop-spin: knee mode" : "graduated ~1..100us wall-clock");
    if (knee_mode) {
        // Measured (bh-18, 2026-08-19): the volatile nop loop costs exactly 10 cycles/iteration and a zone's
        // fixed cost is ~87 ns -- from 2x2 x 2000-iter device zone windows at delay 15/500/2000
        // (4.0/75.8/298.0 ms), slope 7.407 ns/unit = 10.00 cycles at 1.35 GHz, intercept 86.7 ns.
        const double zone_ns = 86.7 + zone_cyc * 10.0 / 1.35;
        printf(
            "[perf-debug zones]   --delay=%u nop-iterations/zone ~= %.0f ns/zone, ~%.2f Mmarkers/s/lane "
            "unthrottled (10 cyc/iteration + ~87 ns/zone measured; same unit as --proddelay; 0 = max rate)\n",
            zone_cyc,
            zone_ns,
            2000.0 / zone_ns);
    }
    if (slow_dispatch) {
        IDevice* device = mesh_device->get_devices().front();
        detail::CompileProgram(device, program);
        detail::WriteRuntimeArgsToDevice(device, program);
        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    } else {
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
    }
    printf("[perf-debug zones] workload done; closing device.\n");
    mesh_device->close();
    if (empty_mode) {
        // close() detached the consumers (delivery threads joined), so the stats are complete and
        // race-free to read here.
        perf_debug::unregister_consumer(empty_handle);
        empty_stats->report();
    }
    return 0;
}
