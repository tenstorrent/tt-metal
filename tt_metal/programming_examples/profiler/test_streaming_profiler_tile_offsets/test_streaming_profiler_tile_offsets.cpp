// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Whether a session's tile clock offsets hold. On every chip one idle eth core reads every Tensix and eth tile's wall
// clock against its own (kernels/offset_reader.cpp) at checkpoints between phases of idle, low load (one core), high
// load (FPU matmul and NoC traffic on every worker core), di/dt (1 ms bursts of the high load, 1 ms apart), and the
// same under the firmware's AICLK sweep. The Tensix counters halt while the Tensix domain idles and the eth counters
// never do, so a halt anywhere in the session is a lasting step in the Tensix offsets from the next checkpoint on; a
// rate difference is a trend across them. The reader touches no other tile between checkpoints, so it cannot keep a
// clock domain awake. The streaming profiler's pusher and drainer take two idle eth cores; the reader takes a third,
// so the test runs with the profiler on or off. Exits nonzero if any offset moved by a tick or more from the first
// checkpoint.
//
//   test_streaming_profiler_tile_offsets [--scale S] [--no-sweep] [--csv FILE]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {
const std::string kKernels = "tt_metal/programming_examples/profiler/test_streaming_profiler_tile_offsets/kernels/";
constexpr uint32_t kMaxTargets = 256;  // offset_reader.cpp's scratch layout
constexpr uint32_t kOutWords = 5;
constexpr uint32_t kExit = 0xFFFFFFFFu;
constexpr uint32_t kReps = 256;
constexpr uint32_t kLoadScratch = 0x120000;  // 32 KB of worker L1 above anything the load programs allocate
constexpr uint32_t kLoadBytes = 8192;
constexpr uint32_t kPartners = 8;
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kSweepStart = 0x31, kSweepStop = 0x32;  // SMC AISWEEP_START {low, high MHz}, AISWEEP_STOP
constexpr uint32_t kSweepLowMhz = 800, kSweepHighMhz = 1350;

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

struct Target {
    bool tensix;
    CoreCoord logical, virt;
};

struct Reader {
    IDevice* device = nullptr;
    uint32_t chip = 0;
    CoreCoord logical, virt;
    std::vector<Target> targets;
    std::unique_ptr<Program> program;
};

struct Checkpoint {
    std::string phase;
    std::vector<int> aiclk;                     // per reader, MHz
    std::vector<std::vector<int64_t>> whole2;   // per reader and target: 2 * (target wall - reader wall), ticks
    std::vector<std::vector<int32_t>> spread2;  // the quartile spread of the same, ticks
};

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> out(cores.begin(), cores.end());
    std::sort(out.begin(), out.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });
    return out;
}

// The streaming profiler's pusher is the lowest (y, x) idle eth core and its drainer the idle core nearest the pusher
// on the NoC (streaming_profiler_device.cpp); the reader is the highest of the rest.
std::optional<CoreCoord> pick_reader(tt::Cluster& cluster, IDevice* d) {
    const std::vector<CoreCoord> idle = sorted_yx(d->get_inactive_ethernet_cores());
    if (idle.size() < 3) {
        return std::nullopt;
    }
    const auto phys = [&](const CoreCoord& l) {
        return cluster.get_physical_coordinate_from_logical_coordinates(d->id(), l, CoreType::ETH, true);
    };
    const CoreCoord p = phys(idle.front());
    size_t drainer = 1;
    int best = std::numeric_limits<int>::max();
    for (size_t i = 1; i < idle.size(); i++) {
        const CoreCoord q = phys(idle[i]);
        const int hops = std::abs(static_cast<int>(q.x) - static_cast<int>(p.x)) +
                         std::abs(static_cast<int>(q.y) - static_cast<int>(p.y));
        if (hops < best) {
            best = hops;
            drainer = i;
        }
    }
    return idle.size() - 1 != drainer ? idle.back() : idle[idle.size() - 2];
}

uint32_t reader_scratch() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
}

void launch(Reader& r) {
    auto& cluster = MetalContext::instance().get_cluster();
    std::vector<uint32_t> words(32 + kMaxTargets, 0);
    for (size_t i = 0; i < r.targets.size(); i++) {
        words[32 + i] = static_cast<uint32_t>(r.targets[i].virt.y << 16 | r.targets[i].virt.x);
    }
    cluster.write_core(
        words.data(),
        static_cast<uint32_t>(words.size() * sizeof(uint32_t)),
        tt_cxy_pair(r.chip, r.virt),
        reader_scratch());
    r.program = std::make_unique<Program>(CreateProgram());
    const KernelHandle kid = CreateKernel(
        *r.program,
        kKernels + "offset_reader.cpp",
        r.logical,
        EthernetConfig{
            .eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default, .processor = DataMovementProcessor::RISCV_0});
    SetRuntimeArgs(*r.program, kid, r.logical, {reader_scratch(), static_cast<uint32_t>(r.targets.size()), kReps});
    detail::CompileProgram(r.device, *r.program, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(r.device, *r.program, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(r.device, *r.program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
}

// A host-launched kernel leaves its launch slot valid on exit, so the slot goes back to the firmware's initial message.
void release(Reader& r) {
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    cluster.write_core(&kExit, sizeof(kExit), tt_cxy_pair(r.chip, r.virt), reader_scratch() + 64);
    detail::WaitProgramDone(r.device, *r.program, false);
    auto msg = hal.get_dev_msgs_factory(HalProgrammableCoreType::IDLE_ETH).create<dev_msgs::launch_msg_t>();
    cluster.write_core(
        msg.data(),
        static_cast<uint32_t>(msg.size()),
        tt_cxy_pair(r.chip, r.virt),
        hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::LAUNCH));
}

bool take(std::vector<Reader>& readers, uint32_t k, const std::string& phase, Checkpoint& cp) {
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t scratch = reader_scratch();
    cp.phase = phase;
    for (Reader& r : readers) {
        cluster.write_core(&k, sizeof(k), tt_cxy_pair(r.chip, r.virt), scratch + 64);
    }
    for (Reader& r : readers) {
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        uint32_t done = 0;
        do {
            cluster.read_core(&done, sizeof(done), tt_cxy_pair(r.chip, r.virt), scratch + 68);
            if (Clock::now() > deadline) {
                std::printf("[tile offsets] chip %u reader did not finish checkpoint %u\n", r.chip, k);
                return false;
            }
        } while (done != k);
        std::vector<uint32_t> out(kOutWords * r.targets.size());
        cluster.read_core(
            out.data(),
            static_cast<uint32_t>(out.size() * sizeof(uint32_t)),
            tt_cxy_pair(r.chip, r.virt),
            scratch + 128 + 4 * kMaxTargets);
        std::vector<int64_t> w(r.targets.size());
        std::vector<int32_t> s(r.targets.size());
        for (size_t t = 0; t < r.targets.size(); t++) {
            const auto median2 = static_cast<int32_t>(out[kOutWords * t]);
            const auto coarse = static_cast<int64_t>((uint64_t{out[kOutWords * t + 4]} << 32) | out[kOutWords * t + 3]);
            const double turns = (static_cast<double>(coarse) - median2 / 2.0) / 4294967296.0;
            w[t] = median2 + 2 * std::llround(turns) * int64_t{4294967296};
            s[t] = static_cast<int32_t>(out[kOutWords * t + 1]);
        }
        cp.whole2.push_back(std::move(w));
        cp.spread2.push_back(std::move(s));
        cp.aiclk.push_back(cluster.get_device_aiclk(r.chip));
    }
    return true;
}

// Each chip's offsets against the first checkpoint, in ticks: the Tensix tiles' common step (their median), the
// largest Tensix change beyond it, and the largest eth change. Returns the largest of the three over the chips.
double report(const std::vector<Reader>& readers, const Checkpoint& base, const Checkpoint& cp, size_t k) {
    double worst = 0.0;
    for (size_t i = 0; i < readers.size(); i++) {
        const Reader& r = readers[i];
        std::vector<double> tensix;
        for (size_t t = 0; t < r.targets.size(); t++) {
            if (r.targets[t].tensix) {
                tensix.push_back((cp.whole2[i][t] - base.whole2[i][t]) / 2.0);
            }
        }
        std::vector<double> sorted = tensix;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double common = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
        double core_worst = 0.0, eth_worst = 0.0;
        CoreCoord core_at, eth_at;
        std::vector<int32_t> spreads = cp.spread2[i];
        for (size_t t = 0; t < r.targets.size(); t++) {
            const double d = (cp.whole2[i][t] - base.whole2[i][t]) / 2.0;
            if (r.targets[t].tensix && std::abs(d - common) > std::abs(core_worst)) {
                core_worst = d - common;
                core_at = r.targets[t].logical;
            } else if (!r.targets[t].tensix && std::abs(d) > std::abs(eth_worst)) {
                eth_worst = d;
                eth_at = r.targets[t].logical;
            }
        }
        std::nth_element(spreads.begin(), spreads.begin() + spreads.size() / 2, spreads.end());
        std::printf(
            "[tile offsets] cp %2zu %-24s chip %u aiclk %4d: Tensix common %+.1f, per core %+.1f at (%zu,%zu); eth "
            "%+.1f at (%zu,%zu); median quartile spread %.1f ticks\n",
            k,
            cp.phase.c_str(),
            r.chip,
            cp.aiclk[i],
            common,
            core_worst,
            core_at.x,
            core_at.y,
            eth_worst,
            eth_at.x,
            eth_at.y,
            spreads.empty() ? 0.0 : spreads[spreads.size() / 2] / 2.0);
        worst = std::max({worst, std::abs(common), std::abs(core_worst), std::abs(eth_worst)});
    }
    std::fflush(stdout);
    return worst;
}

struct LoadSpec {
    uint32_t bursts = 1, mm_iters = 0, dm_iters = 0, idle_cycles = 0;
    bool one_core = false;
};

std::shared_ptr<distributed::MeshWorkload> make_load(distributed::MeshDevice& mesh, const LoadSpec& s) {
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    const CoreRange range(CoreCoord(0, 0), s.one_core ? CoreCoord(0, 0) : CoreCoord(grid.x - 1, grid.y - 1));
    Program program = CreateProgram();
    for (const tt::CBIndex cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program,
            range,
            CircularBufferConfig(2 * kTileBytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, kTileBytes));
    }
    const KernelHandle compute = CreateKernel(
        program, kKernels + "load_compute.cpp", range, ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
    const KernelHandle dm0 = CreateKernel(
        program,
        kKernels + "load_dm.cpp",
        range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const KernelHandle dm1 = CreateKernel(
        program,
        kKernels + "load_dm.cpp",
        range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            const CoreCoord c(x, y);
            SetRuntimeArgs(program, compute, c, {s.bursts, s.mm_iters, s.idle_cycles});
            std::vector<uint32_t> args = {kLoadScratch, kLoadBytes, s.bursts, s.dm_iters, s.idle_cycles, kPartners};
            for (uint32_t p = 1; p <= kPartners; p++) {
                const CoreCoord v = mesh.worker_core_from_logical_core(
                    CoreCoord((x + 3 * p) % grid.x, (y + 5 * p + (s.one_core ? 1 : 0)) % grid.y));
                args.push_back(static_cast<uint32_t>(v.y << 16 | v.x));
            }
            SetRuntimeArgs(program, dm0, c, args);
            SetRuntimeArgs(program, dm1, c, args);
        }
    }
    auto w = std::make_shared<distributed::MeshWorkload>();
    w->add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return w;
}

double run_once(distributed::MeshCommandQueue& cq, distributed::MeshWorkload& w) {
    const auto t0 = Clock::now();
    distributed::EnqueueMeshWorkload(cq, w, /*blocking=*/false);
    distributed::Finish(cq);
    return seconds_since(t0);
}

// The firmware's AICLK sweep on every chip: a random target in the range every DVFS tick (1 ms).
struct Sweep {
    std::vector<uint32_t> chips;
    bool on = false;
    void set(bool want) {
        if (want == on) {
            return;
        }
        auto& driver = MetalContext::instance().get_cluster().get_driver();
        for (const uint32_t chip : chips) {
            const std::vector<uint32_t> args =
                want ? std::vector<uint32_t>{kSweepLowMhz, kSweepHighMhz} : std::vector<uint32_t>{};
            const int rc = driver->arc_msg(
                static_cast<int>(chip), want ? kSweepStart : kSweepStop, true, args, std::chrono::milliseconds(2000));
            if (rc != 0) {
                std::printf("[tile offsets] chip %u: AICLK sweep %s returned %d\n", chip, want ? "start" : "stop", rc);
            }
        }
        on = want;
    }
    ~Sweep() { set(false); }
};

enum class Kind { Idle, Low, High, Didt };
struct Phase {
    const char* name;
    Kind kind;
    double seconds;
    bool sweep;
};
}  // namespace

int main(int argc, char** argv) {
    double scale = 1.0;
    bool sweep_allowed = true;
    const char* csv_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) {
            scale = std::strtod(argv[++i], nullptr);
        } else if (!std::strcmp(argv[i], "--no-sweep")) {
            sweep_allowed = false;
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    auto mesh = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    auto& cluster = MetalContext::instance().get_cluster();
    std::vector<Reader> readers;
    Sweep sweep;
    for (IDevice* d : mesh->get_devices()) {
        const auto chip = static_cast<uint32_t>(d->id());
        sweep.chips.push_back(chip);
        const std::optional<CoreCoord> at = pick_reader(cluster, d);
        if (!at) {
            std::printf("[tile offsets] chip %u: fewer than three idle eth cores, no reader\n", chip);
            continue;
        }
        Reader r;
        r.device = d;
        r.chip = chip;
        r.logical = *at;
        r.virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, *at, CoreType::ETH);
        const CoreCoord grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord l(x, y);
                r.targets.push_back(Target{
                    true, l, cluster.get_virtual_coordinate_from_logical_coordinates(chip, l, CoreType::WORKER)});
            }
        }
        std::unordered_set<CoreCoord> eth = d->get_inactive_ethernet_cores();
        for (const CoreCoord& l : d->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false)) {
            eth.insert(l);
        }
        for (const CoreCoord& l : sorted_yx(eth)) {
            if (l != *at) {
                r.targets.push_back(
                    Target{false, l, cluster.get_virtual_coordinate_from_logical_coordinates(chip, l, CoreType::ETH)});
            }
        }
        if (r.targets.size() > kMaxTargets) {
            std::printf(
                "[tile offsets] chip %u: %zu targets, the reader holds %u\n", chip, r.targets.size(), kMaxTargets);
            return 1;
        }
        readers.push_back(std::move(r));
    }
    for (Reader& r : readers) {
        launch(r);
        std::printf(
            "[tile offsets] chip %u: reader on eth (%zu,%zu), %zu targets\n",
            r.chip,
            r.logical.x,
            r.logical.y,
            r.targets.size());
    }

    distributed::MeshCommandQueue& cq = mesh->mesh_command_queue();
    const LoadSpec cal_mm{.mm_iters = 2000}, cal_dm{.dm_iters = 200};
    auto w_mm = make_load(*mesh, cal_mm), w_dm = make_load(*mesh, cal_dm);
    run_once(cq, *w_mm);
    run_once(cq, *w_dm);
    const double s_mm = run_once(cq, *w_mm) / cal_mm.mm_iters, s_dm = run_once(cq, *w_dm) / cal_dm.dm_iters;
    const auto iters = [](double seconds, double per) {
        return std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds / per)));
    };
    const uint32_t idle_1ms = static_cast<uint32_t>(cluster.get_device_aiclk(readers.front().chip)) * 1000u;
    auto w_low =
        make_load(*mesh, LoadSpec{.mm_iters = iters(1e-4, s_mm), .dm_iters = iters(1e-4, s_dm), .one_core = true});
    auto w_high = make_load(*mesh, LoadSpec{.mm_iters = iters(0.05, s_mm), .dm_iters = iters(0.05, s_dm)});
    auto w_didt = make_load(
        *mesh,
        LoadSpec{.bursts = 25, .mm_iters = iters(1e-3, s_mm), .dm_iters = iters(1e-3, s_dm), .idle_cycles = idle_1ms});
    std::printf(
        "[tile offsets] %zu chips; load calibrated at %.2f us per matmul block, %.2f us per NoC round; high-load "
        "program %.1f ms\n",
        readers.size(),
        s_mm * 1e6,
        s_dm * 1e6,
        run_once(cq, *w_high) * 1e3);
    std::fflush(stdout);

    const std::vector<Phase> phases = {
        {"start", Kind::Idle, 0, false},
        {"start again", Kind::Idle, 0, false},
        {"idle", Kind::Idle, 5, false},
        {"idle", Kind::Idle, 30, false},
        {"low load", Kind::Low, 30, false},
        {"high load", Kind::High, 30, false},
        {"di/dt", Kind::Didt, 30, false},
        {"sweep idle", Kind::Idle, 10, true},
        {"sweep low load", Kind::Low, 10, true},
        {"sweep high load", Kind::High, 30, true},
        {"sweep di/dt", Kind::Didt, 30, true},
        {"high load", Kind::High, 30, false},
        {"idle", Kind::Idle, 60, false},
    };
    std::vector<Checkpoint> cps;
    double worst = 0.0;
    bool ok = true;
    for (size_t k = 0; k < phases.size() && ok; k++) {
        const Phase& p = phases[k];
        sweep.set(p.sweep && sweep_allowed);
        const auto t0 = Clock::now();
        const double seconds = p.seconds * scale;
        do {
            switch (p.kind) {
                case Kind::Idle: std::this_thread::sleep_for(std::chrono::duration<double>(seconds)); break;
                case Kind::Low:
                    run_once(cq, *w_low);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    break;
                case Kind::High: run_once(cq, *w_high); break;
                case Kind::Didt: run_once(cq, *w_didt); break;
            }
        } while (seconds_since(t0) < seconds);
        const std::string name = fmt::format("{} {:.0f} s", p.name, seconds);
        Checkpoint cp;
        ok = take(readers, static_cast<uint32_t>(k + 1), name, cp);
        if (ok) {
            cps.push_back(std::move(cp));
            worst = std::max(worst, report(readers, cps.front(), cps.back(), k));
        }
    }
    sweep.set(false);
    for (Reader& r : readers) {
        release(r);
    }
    mesh->close();

    if (FILE* f = csv_path != nullptr ? std::fopen(csv_path, "w") : nullptr) {
        std::fprintf(f, "cp,phase,chip,aiclk,tensix,x,y,whole2,spread2\n");
        for (size_t k = 0; k < cps.size(); k++) {
            for (size_t i = 0; i < readers.size(); i++) {
                for (size_t t = 0; t < readers[i].targets.size(); t++) {
                    const Target& g = readers[i].targets[t];
                    std::fprintf(
                        f,
                        "%zu,%s,%u,%d,%d,%zu,%zu,%lld,%d\n",
                        k,
                        cps[k].phase.c_str(),
                        readers[i].chip,
                        cps[k].aiclk[i],
                        g.tensix ? 1 : 0,
                        g.logical.x,
                        g.logical.y,
                        static_cast<long long>(cps[k].whole2[i][t]),
                        cps[k].spread2[i][t]);
                }
            }
        }
        std::fclose(f);
    }
    std::printf(
        "[tile offsets] %s: %zu checkpoints on %zu chips; the largest offset change from the first checkpoint is %.1f "
        "ticks\n",
        ok && worst < 1.0 ? "PASS" : "FAIL",
        cps.size(),
        readers.size(),
        worst);
    return ok && worst < 1.0 ? 0 : 1;
}
