// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ERISC load-overlap microbenchmark (fabric_exp/RESULTS.md §17). The router's sending ERISC spends ~15 of the ~28
// cycles of every empty sender-channel visit waiting on two reads it uses at once: the channel's free-slot stream
// register and its completion counter in L1. Reading them one visit early (prefetching) hides that only if the core
// keeps the reads in flight while it does other work, and if other loads issued meanwhile do not wait behind them.
// This measures both on ERISC0 of an active Ethernet core (the router's sending ERISC), fabric off.
//
// Each pattern is an inline-asm loop: loads, K independent ALU instructions, then the instructions using the loads.
// "Load cost" = cycles per block minus the same-shape baseline without loads (BASE2 to BASE5). Reads: S, S2 = stream
// registers; L, L2 = L1 words; D = a word in local data memory (where the stack lives); TX-queue command / status
// and NoC status / command-buffer registers (read only: nothing is sent).
//
// Needs TT_METAL_SLOW_DISPATCH_MODE=1. Environment: ERISC_LOAD_ETH_INDEX (0), ERISC_LOAD_CSV, ERISC_LOAD_DATA_CACHE
// (set: L1 data cache on during the run; default off, as in the router).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/erisc_load_overlap_common.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace erisc_load_overlap;

constexpr const char* kKernel = "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/erisc_load_overlap.cpp";

struct PatternInfo {
    const char* name;
    uint32_t adds;  // non-load instructions besides the filler (BASE2/BASE3 shape)
};

constexpr PatternInfo kPatterns[kNumPatterns] = {
    {"baseline, 2 adds", 2},
    {"S used at once", 2},
    {"L used at once", 2},
    {"S used at once, L used at once (today)", 2},
    {"S, K, use", 2},
    {"L, K, use", 2},
    {"S + L, K, use both (prefetch)", 2},
    {"L + S, K, use both", 2},
    {"S + D, K, use both", 2},
    {"S + S2, K, use both", 2},
    {"L + D, K, use both", 2},
    {"L + L2, K, use both", 2},
    {"S + discarded D load, K, use S", 2},
    {"baseline, 3 adds", 3},
    {"D used at once", 3},
    {"S2 used at once", 3},
    {"S + L, K/2, D used at once, K/2, use", 3},
    {"S + L, K/2, S2 used at once, K/2, use", 3},
    {"S + L + D, K, use all", 3},
    {"S + S2 + L, K, use all", 3},
    {"baseline, 4 adds", 4},
    {"4 stream reads, K, use all", 4},
    {"baseline, 5 adds", 5},
    {"4 stream reads + L, K, use all", 5},
    {"TXQ cmd used at once", 2},
    {"TXQ cmd + status used at once (busy)", 2},
    {"NoC write-sent status used at once", 2},
    {"NoC cmd-buf ready used at once", 2},
    {"TXQ cmd + status, K, use status", 2},
    {"S + TXQ cmd + status, K, use S + status", 2},
    {"S + NoC write-sent, K, use both", 2},
    {"S + TXQ cmd/status + NoC, K, use 3", 3},
    {"TXQ cmd, status into one reg, used at once", 2},
    {"S + TXQ (one reg) + NoC, K, use 3", 3},
    {"4 stream reads + TXQ (one reg), K, use all", 5},
    {"baseline, 7 adds", 7},
    {"4 stream reads, then D, K, use all", 5},
    {"D, then 4 stream reads, K, use all", 5},
    {"4 stream reads, then D D2 D3, K, use all", 7},
    {"D D2 D3, then 4 stream reads, K, use all", 7},
    {"4 stream, K, use; header L (slot in reg) at once", 5},
    {"4 stream + slot DS, K, use; header via DS at once", 5},
    {"slot DS, header via DS used at once, K", 2},
};

}  // namespace

TEST(EriscLoadOverlap, ReadLatencyHiding) {
    if (MetalContext::instance().get_cluster().arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole only";
    }
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "needs TT_METAL_SLOW_DISPATCH_MODE=1";
    }
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    if (MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::DISABLED) {
        mesh->close();
        GTEST_FAIL() << "fabric was enabled; its routers would occupy the ERISCs";
    }
    IDevice* device = mesh->get_devices()[0];
    std::vector<CoreCoord> eth_cores;
    for (const auto& c : device->get_active_ethernet_cores(true)) {
        eth_cores.push_back(c);
    }
    ASSERT_FALSE(eth_cores.empty());
    std::sort(eth_cores.begin(), eth_cores.end());
    const char* idx_env = std::getenv("ERISC_LOAD_ETH_INDEX");
    const uint32_t eth_index = idx_env ? std::stoul(idx_env) : 0;
    ASSERT_LT(eth_index, eth_cores.size());
    const CoreCoord logical = eth_cores[eth_index];
    const CoreCoord virt = mesh->virtual_core_from_logical_core(logical, CoreType::ETH);

    const auto& hal = MetalContext::instance().hal();
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const std::vector<uint32_t> zeros(kResultWords, 0);
    cluster.write_core(zeros.data(), kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    cluster.l1_barrier(device->id());

    // One program per pattern group (code size); each writes only its own patterns' results.
    std::vector<uint32_t> r(kResultWords, 0);
    for (uint32_t group = 0; group < kNumGroups; group++) {
        Program program = CreateProgram();
        std::map<std::string, std::string> defines = {{"ERISC_LOAD_GROUP", std::to_string(group)}};
        if (std::getenv("ERISC_LOAD_DATA_CACHE")) {
            defines["ERISC_LOAD_DATA_CACHE"] = "1";
        }
        const KernelHandle kernel = CreateKernel(
            program,
            kKernel,
            logical,
            EthernetConfig{
                .noc = NOC::NOC_0,
                .processor = DataMovementProcessor::RISCV_0,
                .defines = defines,
                .opt_level = KernelBuildOptLevel::O3});
        SetRuntimeArgs(program, kernel, logical, {base});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
        distributed::Finish(mesh->mesh_command_queue());

        std::vector<uint32_t> g(kResultWords);
        cluster.read_core(g.data(), kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
        ASSERT_EQ(g[kNumPatterns * kNumFill + 1], kDone) << "kernel did not finish (group " << group << ")";
        for (uint32_t p = kGroups[group][0]; p < kGroups[group][1]; p++) {
            for (uint32_t k = 0; k < kNumFill; k++) {
                r[p * kNumFill + k] = g[p * kNumFill + k];
            }
        }
        r[kNumPatterns * kNumFill + 2] = g[kNumPatterns * kNumFill + 2];
        cluster.write_core(zeros.data(), kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
        cluster.l1_barrier(device->id());
    }

    auto per_block = [&](uint32_t p, uint32_t k) { return static_cast<double>(r[p * kNumFill + k]) / kIterations; };
    // Loop overhead per block, from the 2-add baseline: what is left after the filler and the adds.
    auto overhead = [&](uint32_t k) { return per_block(BASE2, k) - kFill[k] - 2.0; };
    auto overhead3 = [&](uint32_t k) { return per_block(BASE3, k) - kFill[k] - 3.0; };
    auto overhead4 = [&](uint32_t k) { return per_block(BASE4, k) - kFill[k] - 4.0; };
    auto overhead5 = [&](uint32_t k) { return per_block(BASE5, k) - kFill[k] - 5.0; };
    auto overhead7 = [&](uint32_t k) { return per_block(BASE7, k) - kFill[k] - 7.0; };
    auto load_cost = [&](uint32_t p, uint32_t k) {
        const uint32_t adds = kPatterns[p].adds;
        const double oh = adds == 7   ? overhead7(k)
                          : adds == 5 ? overhead5(k)
                          : adds == 4 ? overhead4(k)
                          : adds == 3 ? overhead3(k)
                                      : overhead(k);
        return per_block(p, k) - kFill[k] - adds - oh;
    };

    const uint32_t cache_csr = r[kNumPatterns * kNumFill + 2];
    log_info(
        LogTest,
        "ERISC0 on Ethernet core logical ({},{}); cycles per block, {} blocks, best of {}; CSR 0x7c0 at "
        "start {:#x} (bit 3 set: L1 data cache off); data cache during the run: {}",
        logical.x,
        logical.y,
        kIterations,
        kRepeats,
        cache_csr,
        std::getenv("ERISC_LOAD_DATA_CACHE") ? "on" : "off");
    std::string header = fmt::format("{:42s}", "load cost (cycles) at K filler =");
    for (uint32_t k = 0; k < kNumFill; k++) {
        header += fmt::format(" {:5d}", kFill[k]);
    }
    log_info(LogTest, "{}", header);
    std::string oh = fmt::format("{:42s}", "(loop overhead per block)");
    for (uint32_t k = 0; k < kNumFill; k++) {
        oh += fmt::format(" {:5.1f}", overhead(k));
    }
    log_info(LogTest, "{}", oh);
    std::ofstream csv;
    if (const char* path = std::getenv("ERISC_LOAD_CSV")) {
        csv.open(path);
        csv << "pattern,name,fill,cycles_per_block,load_cost\n";
    }
    for (uint32_t p = 0; p < kNumPatterns; p++) {
        std::string line = fmt::format("{:42s}", kPatterns[p].name);
        for (uint32_t k = 0; k < kNumFill; k++) {
            line += fmt::format(" {:5.1f}", load_cost(p, k));
            if (csv.is_open()) {
                csv << p << ",\"" << kPatterns[p].name << "\"," << kFill[k] << "," << per_block(p, k) << ","
                    << load_cost(p, k) << "\n";
            }
        }
        log_info(LogTest, "{}", line);
    }
    mesh->close();
}

// Stack pointer at kernel entry on ERISC0 and ERISC1 of one active Ethernet core (fabric off), and the end of each
// kernel's local-data-memory sections. Each ERISC has 8 KB of local memory at 0xFFB00000; on ERISC0 the base firmware
// keeps the first 0x700 bytes and Metal's kernels run on the stack the base firmware hands them.
TEST(EriscLoadOverlap, StackAtKernelEntry) {
    if (MetalContext::instance().get_cluster().arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole only";
    }
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "needs TT_METAL_SLOW_DISPATCH_MODE=1";
    }
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    ASSERT_EQ(MetalContext::instance().get_fabric_config(), tt_fabric::FabricConfig::DISABLED);
    IDevice* device = mesh->get_devices()[0];
    std::vector<CoreCoord> eth_cores;
    for (const auto& c : device->get_active_ethernet_cores(true)) {
        eth_cores.push_back(c);
    }
    ASSERT_FALSE(eth_cores.empty());
    std::sort(eth_cores.begin(), eth_cores.end());
    const CoreCoord logical = eth_cores[0];
    const CoreCoord virt = mesh->virtual_core_from_logical_core(logical, CoreType::ETH);
    const auto& hal = MetalContext::instance().hal();
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const std::vector<uint32_t> zeros(8, 0);
    cluster.write_core(zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    cluster.l1_barrier(device->id());

    Program program = CreateProgram();
    for (uint32_t risc = 0; risc < 2; risc++) {
        const KernelHandle k = CreateKernel(
            program,
            "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/erisc_stack_probe.cpp",
            logical,
            EthernetConfig{
                .noc = risc == 0 ? NOC::NOC_0 : NOC::NOC_1,
                .processor = risc == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1});
        SetRuntimeArgs(program, k, logical, {base + risc * 16});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
    distributed::Finish(mesh->mesh_command_queue());

    std::vector<uint32_t> r(8);
    cluster.read_core(r.data(), r.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    for (uint32_t risc = 0; risc < 2; risc++) {
        const uint32_t sp = r[risc * 4], ldm_end = r[risc * 4 + 1];
        ASSERT_EQ(r[risc * 4 + 2], 0x5AC0DEu) << "ERISC" << risc << " kernel did not run";
        log_info(
            LogTest,
            "ERISC{} on Ethernet core logical ({},{}): sp at kernel entry {:#x}, this kernel's data end {:#x}",
            risc,
            logical.x,
            logical.y,
            sp,
            ldm_end);
    }
    mesh->close();
}
