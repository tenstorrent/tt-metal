// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Stress test for torn 64-bit wall-clock reads on Blackhole DRISCs. Every DRISC-usable DRAM endpoint of every
// device in the mesh runs drisc_wall_clock_tear.cpp for --seconds; with --reader a Tensix per DRISC hammers
// that DRISC's WALL_CLOCK_L over the NoC meanwhile. Results are read back from DRISC L1.
//
//   TT_METAL_SLOW_DISPATCH_MODE=1 test_wall_clock_tear [--mesh 4x2] [--seconds 120] [--reader] [--gap N]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kDoneMagic = 0xD0E0'0001u;
constexpr uint32_t kOutWords = 16;
enum Out : uint32_t {
    kDone = 0,
    kStop = 1,
    kItersLo = 2,
    kItersHi = 3,
    kWraps = 4,
    kFwdJumps = 5,
    kBackSteps = 6,
    kMaxFwdLo = 7,
    kMaxFwdHi = 8,
    kFirstPrevLo = 9,
    kFirstPrevHi = 10,
    kFirstCurLo = 11,
    kFirstCurHi = 12,
    kHiOnly = 13,
};
enum Stat : uint32_t { kReadsLo = 0, kReadsHi = 1, kLast = 2, kFirst = 3, kRStop = 4, kRDone = 5, kStatWords = 8 };

struct Target {
    uint32_t chip;
    uint32_t bank;
    uint32_t sub;
    CoreCoord drisc_logical;
    CoreCoord drisc_virtual;
    CoreCoord worker_logical;
    CoreCoord worker_virtual;
    uint64_t read_addr;  // host-side address of the result block on drisc_virtual
};

// Logical endpoint index of the bank's NOC0 worker endpoint: the syseng firmware owns it and no kernel runs there.
uint32_t noc0_endpoint(const metal_SocDescriptor& soc, uint32_t bank) {
    const CoreCoord pref = soc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank), 0);
    const auto& endpoints = soc.dram_bank_endpoint_coords.at(bank);
    for (uint32_t i = 0; i < endpoints.size(); i++) {
        if (endpoints[i] == pref) {
            return i;
        }
    }
    TT_FATAL(false, "bank {} has no NOC0 endpoint", bank);
    return 0;
}

uint64_t u64(const std::vector<uint32_t>& w, uint32_t lo) { return (static_cast<uint64_t>(w[lo + 1]) << 32) | w[lo]; }

}  // namespace

int main(int argc, char** argv) {
    uint32_t rows = 4, cols = 2, seconds = 120, gap = 0, hi_off = 8, spin = 0;
    bool reader = false, tensix = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mesh") && i + 1 < argc) {
            rows = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
            const char* x = strchr(argv[i], 'x');
            cols = x != nullptr ? static_cast<uint32_t>(strtoul(x + 1, nullptr, 10)) : 1;
        } else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
        } else if (!strcmp(argv[i], "--gap") && i + 1 < argc) {
            gap = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
        } else if (!strcmp(argv[i], "--hi_off") && i + 1 < argc) {
            hi_off = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
        } else if (!strcmp(argv[i], "--spin_cycles") && i + 1 < argc) {
            spin = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
        } else if (!strcmp(argv[i], "--tensix")) {
            tensix = true;
        } else if (!strcmp(argv[i], "--reader")) {
            reader = true;
        } else {
            fprintf(stderr, "usage: %s [--mesh RxC] [--seconds S] [--reader] [--gap N] [--hi_off 8|4] [--tensix]\n", argv[0]);
            return 2;
        }
    }

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::MeshShape(rows, cols)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1);
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    TT_FATAL(hal.has_programmable_core_type(HalProgrammableCoreType::DRAM), "no DRISC programmable cores");

    const uint32_t out_l1 = static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED));
    const uint64_t out_noc = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    // High in worker L1, clear of the kernel binaries at the low end.
    const uint32_t scratch_l1 = 0x170000u;
    const uint32_t stat_l1 = scratch_l1 + 64;

    const std::string kdir = "tt_metal/programming_examples/profiler/test_wall_clock_tear/kernels/";
    std::vector<Target> targets;
    std::vector<std::unique_ptr<Program>> programs;
    std::vector<IDevice*> devices = mesh_device->get_devices();
    for (IDevice* device : devices) {
        const uint32_t chip = static_cast<uint32_t>(device->id());
        const auto& soc = cluster.get_soc_desc(chip);
        const CoreCoord grid = device->compute_with_storage_grid_size();
        std::vector<Target> mine;
        if (tensix) {
            for (uint32_t y = 0; y < grid.y; y++) {
                for (uint32_t x = 0; x < grid.x; x++) {
                    Target t{};
                    t.chip = chip;
                    t.bank = x;
                    t.sub = y;
                    t.drisc_logical = CoreCoord{x, y};
                    t.drisc_virtual = device->virtual_core_from_logical_core(t.drisc_logical, CoreType::WORKER);
                    t.worker_logical = t.drisc_logical;
                    t.worker_virtual = t.drisc_virtual;
                    t.read_addr = scratch_l1;
                    mine.push_back(t);
                }
            }
        }
        for (uint32_t bank = 0; !tensix && bank < soc.get_num_dram_views(); bank++) {
            const uint32_t skip = noc0_endpoint(soc, bank);
            const uint32_t n_ep = static_cast<uint32_t>(soc.dram_bank_endpoint_coords.at(bank).size());
            for (uint32_t sub = 0; sub < n_ep; sub++) {
                if (sub == skip) {
                    continue;
                }
                Target t{};
                t.chip = chip;
                t.bank = bank;
                t.sub = sub;
                t.drisc_logical = CoreCoord{bank, sub};
                t.drisc_virtual = device->virtual_core_from_logical_core(t.drisc_logical, CoreType::DRAM);
                const uint32_t k = static_cast<uint32_t>(mine.size());
                t.worker_logical = CoreCoord{k % grid.x, k / grid.x};
                t.worker_virtual = device->virtual_core_from_logical_core(t.worker_logical, CoreType::WORKER);
                t.read_addr = out_noc;
                mine.push_back(t);
            }
        }
        auto program = std::make_unique<Program>(CreateProgram());
        std::set<CoreRange> drisc_ranges, worker_ranges;
        for (const Target& t : mine) {
            drisc_ranges.insert(CoreRange(t.drisc_logical, t.drisc_logical));
            worker_ranges.insert(CoreRange(t.worker_logical, t.worker_logical));
        }
        const KernelHandle kid =
            tensix ? CreateKernel(
                         *program,
                         kdir + "drisc_wall_clock_tear.cpp",
                         CoreRangeSet(drisc_ranges),
                         DataMovementConfig{
                             .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = {gap, hi_off, spin}})
                   : CreateKernel(
                         *program,
                         kdir + "drisc_wall_clock_tear.cpp",
                         CoreRangeSet(drisc_ranges),
                         DramConfig{.noc = NOC::NOC_0, .compile_args = {gap, hi_off, spin}});
        for (const Target& t : mine) {
            SetRuntimeArgs(*program, kid, t.drisc_logical, {tensix ? scratch_l1 : out_l1});
        }
        if (tensix) {
            reader = false;
        }
        if (reader) {
            auto rid = CreateKernel(
                *program,
                kdir + "tensix_wall_clock_reader.cpp",
                CoreRangeSet(worker_ranges),
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
            for (const Target& t : mine) {
                SetRuntimeArgs(
                    *program,
                    rid,
                    t.worker_logical,
                    {static_cast<uint32_t>(t.drisc_virtual.x), static_cast<uint32_t>(t.drisc_virtual.y), scratch_l1, stat_l1});
            }
            std::vector<uint32_t> zero(kStatWords, 0);
            for (const Target& t : mine) {
                cluster.write_core(
                    zero.data(), kStatWords * sizeof(uint32_t), tt_cxy_pair(chip, t.worker_virtual), stat_l1);
            }
        }
        std::vector<uint32_t> zero_out(kOutWords, 0);
        for (const Target& t : mine) {
            cluster.write_core(zero_out.data(), kOutWords * sizeof(uint32_t), tt_cxy_pair(chip, t.drisc_virtual), t.read_addr);
        }
        detail::CompileProgram(device, *program);
        detail::WriteRuntimeArgsToDevice(device, *program);
        detail::LaunchProgram(device, *program, /*wait_until_cores_done=*/false);
        targets.insert(targets.end(), mine.begin(), mine.end());
        programs.push_back(std::move(program));
    }
    printf(
        "[wallclock tear] %zu %s on %zu devices, gap=%u nops, hi_off=%u, spin=%u cycles, reader=%s, %u s\n",
        targets.size(),
        tensix ? "Tensix BRISCs" : "DRISCs",
        devices.size(),
        gap,
        hi_off,
        spin,
        reader ? "yes" : "no",
        seconds);
    fflush(stdout);

    auto read_out = [&](const Target& t) {
        std::vector<uint32_t> w(kOutWords, 0);
        cluster.read_core(w.data(), kOutWords * sizeof(uint32_t), tt_cxy_pair(t.chip, t.drisc_virtual), t.read_addr);
        return w;
    };
    auto read_stat = [&](const Target& t) {
        std::vector<uint32_t> w(kStatWords, 0);
        cluster.read_core(w.data(), kStatWords * sizeof(uint32_t), tt_cxy_pair(t.chip, t.worker_virtual), stat_l1);
        return w;
    };

    const auto t_start = std::chrono::steady_clock::now();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        uint64_t iters = 0, reads = 0;
        uint32_t wraps = 0, fwd = 0, back = 0, hi_only = 0, dead = 0;
        for (const Target& t : targets) {
            const auto w = read_out(t);
            iters += u64(w, kItersLo);
            wraps += w[kWraps];
            fwd += w[kFwdJumps];
            back += w[kBackSteps];
            hi_only += w[kHiOnly];
            dead += u64(w, kItersLo) == 0 ? 1u : 0u;
            if (reader) {
                reads += u64(read_stat(t), kReadsLo);
            }
        }
        printf(
            "[wallclock tear] t=%.0fs iters=%llu wraps=%u fwd=%u back=%u hi_only=%u not_started=%u reader_reads=%llu\n",
            elapsed,
            (unsigned long long)iters,
            wraps,
            fwd,
            back,
            hi_only,
            dead,
            (unsigned long long)reads);
        fflush(stdout);
        if (elapsed >= seconds) {
            break;
        }
    }

    const uint32_t one = 1;
    for (const Target& t : targets) {
        cluster.write_core(&one, sizeof(one), tt_cxy_pair(t.chip, t.drisc_virtual), t.read_addr + kStop * 4);
        if (reader) {
            cluster.write_core(&one, sizeof(one), tt_cxy_pair(t.chip, t.worker_virtual), stat_l1 + kRStop * 4);
        }
    }
    for (size_t i = 0; i < devices.size(); i++) {
        detail::WaitProgramDone(devices[i], *programs[i]);
    }

    uint32_t total_fwd = 0, total_back = 0, total_hi = 0;
    printf("\n%-5s %-4s %-3s %-9s %14s %6s %5s %5s %7s %s\n", "chip", "bank", "ep", "drisc", "iters", "wraps", "fwd", "back", "hi_only", "first anomaly prev -> cur / reader");
    for (const Target& t : targets) {
        const auto w = read_out(t);
        total_fwd += w[kFwdJumps];
        total_back += w[kBackSteps];
        total_hi += w[kHiOnly];
        std::string extra;
        if (w[kFwdJumps] != 0) {
            char buf[96];
            snprintf(
                buf,
                sizeof buf,
                "0x%08x%08x -> 0x%08x%08x max_fwd=%llu",
                w[kFirstPrevHi],
                w[kFirstPrevLo],
                w[kFirstCurHi],
                w[kFirstCurLo],
                (unsigned long long)u64(w, kMaxFwdLo));
            extra += buf;
        }
        if (reader) {
            const auto s = read_stat(t);
            char buf[96];
            snprintf(buf, sizeof buf, " reader reads=%llu first=0x%08x last=0x%08x done=%u", (unsigned long long)u64(s, kReadsLo), s[kFirst], s[kLast], s[kRDone]);
            extra += buf;
        }
        printf(
            "%-5u %-4u %-3u v(%2u,%2u)  %14llu %6u %5u %5u %7u %s%s\n",
            t.chip,
            t.bank,
            t.sub,
            (unsigned)t.drisc_virtual.x,
            (unsigned)t.drisc_virtual.y,
            (unsigned long long)u64(w, kItersLo),
            w[kWraps],
            w[kFwdJumps],
            w[kBackSteps],
            w[kHiOnly],
            w[kDone] == kDoneMagic ? "" : "NOT DONE ",
            extra.c_str());
    }
    printf("\n[wallclock tear] TOTAL fwd=%u back=%u hi_only=%u over %zu DRISCs\n", total_fwd, total_back, total_hi, targets.size());
    mesh_device->close();
    return 0;
}
