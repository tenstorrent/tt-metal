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

#include <tt-metalium/allocator.hpp>
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
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/noc_cmdbuf_overlap_common.hpp"
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/trid_sent_latency_common.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace erisc_load_overlap;

constexpr const char* kKernel = "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/erisc_load_overlap.cpp";

constexpr uint32_t kNoBase = UINT32_MAX;

struct PatternInfo {
    const char* name;
    uint32_t adds;               // non-load instructions besides the filler (BASE2/BASE3 shape)
    uint32_t base = kNoBase;     // send-batch patterns: the same-shape pattern without the loads measured (cost = difference)
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
    {"count read; K; 4 uses (baseline)", 4, HB_BASE},
    {"count read; H0, K, use", 4, HB_BASE},
    {"count read; H0 H1 together, K, use", 4, HB_BASE},
    {"count read; H0 H1 H2 together, K, use", 4, HB_BASE},
    {"count read; H0..H3 together, K, use", 4, HB_BASE},
    {"count read; H0 at once, H1 at once, K", 4, HB_BASE},
    {"count read; H0..H3 each at once, K", 4, HB_BASE},
    {"count read; DS -> H0, K, use", 4, HB_BASE},
    {"count read; DS DS2 -> H0 H1 together, K, use", 4, HB_BASE},
    {"count read; DS -> H0 at once, DS2 -> H1 at once, K", 4, HB_BASE},
    {"count read; 7 TXQ writes, K (baseline)", 2, HW_BASE},
    {"count read; H1, 7 writes, K, use", 2, HW_BASE},
    {"count read; 4 writes, H1, 3 writes, K, use", 2, HW_BASE},
    {"count read; 7 writes, H1, K, use", 2, HW_BASE},
    {"count read; DS2 -> H1, 7 writes, K, use", 2, HW_BASE},
    {"count read; H1 at once, 7 writes, K", 2, HW_BASE},
    {"count read; K; 3 uses (baseline)", 3, HP_BASE},
    {"count read; TXQ T Q + counter, K, use", 3, HP_BASE},
    {"count read; TXQ T Q + counter + H1, K, use", 3, HP_BASE},
    {"count read; lw H0+40, K, use", 4, HB_BASE},
    {"count read; lhu L, K, use", 4, HB_BASE},
    {"count read; lw L, K, use", 4, HB_BASE},
    {"count read; lhu H0+40 at once, K", 4, HB_BASE},
    {"count read; lw H0+40 at once, K", 4, HB_BASE},
    {"count read; lw L at once, K", 4, HB_BASE},
    {"no count read; lw L, K, use", 4, HX_BASE_NC},
    {"no count read; lhu H0+40, K, use", 4, HX_BASE_NC},
    {"no count read; K, 4 uses (baseline)", 4, HX_BASE_NC},
    {"count read; TXQ T Q at once, K", 4, HB_BASE},
    {"count read; H0 + TXQ T Q at once, K", 4, HB_BASE},
    {"count read; (H0 T Q at once) x2, K", 4, HB_BASE},
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
        if (kPatterns[p].base != kNoBase) {
            return per_block(p, k) - per_block(kPatterns[p].base, k);
        }
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

// NoC command buffers on the router's receiving core (fabric_exp/consolidation/V4_STATE.md §17-18). ERISC1 with NOC1, as
// the receiver, writes 64 B to 4 KiB transaction-ID tagged writes to a Tensix core's L1, issued as the receiver's
// local writes. Table 1: a ready read before each write, writes rotating over 1, 2 or 4 command buffers. Table 2: a
// ready mask instead: buffers known ready are used without a read, the mask sampled as a group at each pass's end
// (every 4 writes) and again before a buffer whose bit is clear; fixed rotation or the lowest ready buffer. Table 3: the
// cost of one grouped ready sample of 1 to 4 buffers. Table 4: forward-like pairs (a posted data write and the posted
// credit, an inline write to a stream register of the same core) on one shared buffer, on two buffers fully
// programmed, on two with presets, and with presets and the mask. Bursts of 4 writes from an idle interface (best of
// 64), streams of 64 (best of 8), 32 pairs (best of 8). Fabric off. Environment: ERISC_LOAD_ETH_INDEX (0),
// NOC_CMDBUF_CSV.
TEST(EriscLoadOverlap, NocCmdBufOverlap) {
    namespace nc = noc_cmdbuf_overlap;
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
    const char* idx_env = std::getenv("ERISC_LOAD_ETH_INDEX");
    const uint32_t eth_index = idx_env ? std::stoul(idx_env) : 0;
    ASSERT_LT(eth_index, eth_cores.size());
    const CoreCoord logical = eth_cores[eth_index];
    const CoreCoord virt = mesh->virtual_core_from_logical_core(logical, CoreType::ETH);
    const CoreCoord worker = device->worker_core_from_logical_core(CoreCoord{0, 0});

    const auto& hal = MetalContext::instance().hal();
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t src_base = base + 0x1000;
    ASSERT_LE(nc::kResultWords * sizeof(uint32_t), 0x1000u);
    ASSERT_LE(0x1000u + nc::kSlots * nc::kSlotBytes, eth_size) << "source ring does not fit the Ethernet core's free L1";
    // The allocator's L1 base on the Tensix core: free while no buffer is allocated there.
    const uint32_t dst_base = static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(HalMemType::L1));
    const std::vector<uint32_t> zeros(nc::kResultWords, 0);
    cluster.write_core(zeros.data(), nc::kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    cluster.l1_barrier(device->id());

    // One program per group; each writes only its own tables, its done flag and its timeout word.
    std::vector<uint32_t> r(nc::kResultWords, 0);
    for (uint32_t group = 0; group < nc::kNumGroups; group++) {
        Program program = CreateProgram();
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/noc_cmdbuf_overlap.cpp",
            logical,
            EthernetConfig{
                .noc = NOC::NOC_1,
                .processor = DataMovementProcessor::RISCV_1,
                .defines = {{"NOC_CMDBUF_GROUP", std::to_string(group)}},
                .opt_level = KernelBuildOptLevel::O3});
        SetRuntimeArgs(program, kernel, logical, {base, src_base, worker.x, worker.y, dst_base});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
        distributed::Finish(mesh->mesh_command_queue());
        cluster.read_core(r.data(), nc::kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
        ASSERT_EQ(r[nc::kDoneBase + group], nc::kDone) << "group " << group << ": kernel did not finish";
        const uint32_t f = r[nc::kFailedBase + group];
        ASSERT_EQ(f, 0u) << fmt::format(
            "group {}: a wait timed out in table {}, shape {}, size index {}, variant {}",
            group,
            f >> 24,
            ((f >> 16) & 0xFF) - 1,
            ((f >> 8) & 0xFF) - 1,
            (f & 0xFF) - 1);
    }

    std::ofstream csv;
    if (const char* path = std::getenv("NOC_CMDBUF_CSV")) {
        csv.open(path);
        csv << "table,shape,size,variant,ops,reps,issue_min,issue_mean,ready_cycles_per_op,reads_per_op,first_min,"
               "sent_min,sent_mean,acked_min\n";
    }
    auto emit = [&](uint32_t table, uint32_t shape, uint32_t size, const std::string& variant, uint32_t at,
                    uint32_t ops, uint32_t reps) {
        if (csv.is_open()) {
            csv << table << "," << shape << "," << size << ",\"" << variant << "\"," << ops << "," << reps << ","
                << r[at + nc::ISSUE_MIN] << "," << static_cast<double>(r[at + nc::ISSUE]) / reps << ","
                << static_cast<double>(r[at + nc::TAG_WAIT]) / (reps * ops) << ","
                << static_cast<double>(r[at + nc::READS]) / (reps * ops) << "," << r[at + nc::FIRST_MIN] << ","
                << r[at + nc::SENT_MIN] << "," << static_cast<double>(r[at + nc::SENT]) / reps << ","
                << r[at + nc::ACKED_MIN] << "\n";
        }
    };
    log_info(
        LogTest,
        "ERISC1 / NOC1 on Ethernet core logical ({},{}) -> Tensix ({},{}); cycles",
        logical.x,
        logical.y,
        worker.x,
        worker.y);
    const char* shape_name[] = {"burst of 4", "stream of 64"};
    for (uint32_t shape = 0; shape < nc::kNumShapes; shape++) {
        const uint32_t writes = shape == nc::BURST ? nc::kBurstWrites : nc::kStreamWrites;
        const uint32_t reps = shape == nc::BURST ? nc::kBurstReps : nc::kStreamReps;
        log_info(
            LogTest,
            "Table 1, {}: a ready read per write; buffers 1 / 2 / 4: issue (min) | ready call per write | first write "
            "| all sent (min) | all acked (min)",
            shape_name[shape]);
        for (uint32_t s = 0; s < nc::kNumSizes; s++) {
            std::string issue, wait, first, sent, acked;
            for (uint32_t b = 0; b < nc::kNumBufCounts; b++) {
                const uint32_t at = nc::t1_index(shape, s, b);
                const char* sep = b ? " / " : "";
                issue += fmt::format("{}{:5d}", sep, r[at + nc::ISSUE_MIN]);
                wait += fmt::format("{}{:5.1f}", sep, static_cast<double>(r[at + nc::TAG_WAIT]) / (reps * writes));
                first += fmt::format("{}{:3d}", sep, r[at + nc::FIRST_MIN]);
                sent += fmt::format("{}{:5d}", sep, r[at + nc::SENT_MIN]);
                acked += fmt::format("{}{:5d}", sep, r[at + nc::ACKED_MIN]);
                emit(1, shape, nc::kSizes[s], fmt::format("{} buffers", nc::kBufCounts[b]), at, writes, reps);
            }
            log_info(LogTest, "{:5d} B | {} | {} | {} | {} | {}", nc::kSizes[s], issue, wait, first, sent, acked);
        }
    }
    const char* mask_name[] = {"rotate 1", "rotate 2", "rotate 4", "any of 2", "any of 4"};
    for (uint32_t shape = 0; shape < nc::kNumShapes; shape++) {
        const uint32_t writes = shape == nc::BURST ? nc::kBurstWrites : nc::kStreamWrites;
        const uint32_t reps = shape == nc::BURST ? nc::kBurstReps : nc::kStreamReps;
        log_info(
            LogTest,
            "Table 2, {}: ready mask; rotate 1 / 2 / 4, any of 2 / 4: issue (min) | ready cycles per write | ready "
            "registers read per write | all sent (min)",
            shape_name[shape]);
        for (uint32_t s = 0; s < nc::kNumSizes; s++) {
            std::string issue, wait, reads, sent;
            for (uint32_t v = 0; v < nc::kNumMaskVariants; v++) {
                const uint32_t at = nc::t2_index(shape, s, v);
                const char* sep = v ? " / " : "";
                issue += fmt::format("{}{:5d}", sep, r[at + nc::ISSUE_MIN]);
                wait += fmt::format("{}{:5.1f}", sep, static_cast<double>(r[at + nc::TAG_WAIT]) / (reps * writes));
                reads += fmt::format("{}{:4.2f}", sep, static_cast<double>(r[at + nc::READS]) / (reps * writes));
                sent += fmt::format("{}{:5d}", sep, r[at + nc::SENT_MIN]);
                emit(2, shape, nc::kSizes[s], mask_name[v], at, writes, reps);
            }
            log_info(LogTest, "{:5d} B | {} | {} | {} | {}", nc::kSizes[s], issue, wait, reads, sent);
        }
    }
    std::string samples;
    for (uint32_t k = 0; k < nc::kMaxSample; k++) {
        samples += fmt::format(
            "{}{} buffers {:.1f}", k ? ", " : "", k + 1, static_cast<double>(r[nc::kT3 + k]) / nc::kSampleIters);
        if (csv.is_open()) {
            csv << 3 << ",," << 0 << ",\"sample " << k + 1 << "\",1," << nc::kSampleIters << ","
                << static_cast<double>(r[nc::kT3 + k]) / nc::kSampleIters << ",,,,,,,\n";
        }
    }
    log_info(LogTest, "Table 3: one grouped ready sample (idle buffers, loop included): {}", samples);
    const char* pair_name[] = {"shared", "full", "preset", "preset+mask", "preset+mask, data rotating"};
    log_info(
        LogTest,
        "Table 4, {} forward-like pairs: shared / full / preset / preset+mask / preset+mask rotating: issue per pair "
        "(min) | ready cycles per pair | first pair | all sent per pair (min)",
        nc::kPairs);
    for (uint32_t s = 0; s < nc::kNumPairSizes; s++) {
        std::string issue, wait, first, sent;
        for (uint32_t m = 0; m < nc::kNumPairModes; m++) {
            const uint32_t at = nc::t4_index(s, m);
            const char* sep = m ? " / " : "";
            issue += fmt::format("{}{:5.1f}", sep, static_cast<double>(r[at + nc::ISSUE_MIN]) / nc::kPairs);
            wait += fmt::format(
                "{}{:5.1f}", sep, static_cast<double>(r[at + nc::TAG_WAIT]) / (nc::kPairReps * nc::kPairs));
            first += fmt::format("{}{:3d}", sep, r[at + nc::FIRST_MIN]);
            sent += fmt::format("{}{:5.1f}", sep, static_cast<double>(r[at + nc::SENT_MIN]) / nc::kPairs);
            emit(4, 0, nc::kPairSizes[s], pair_name[m], at, nc::kPairs, nc::kPairReps);
        }
        log_info(LogTest, "{:5d} B | {} | {} | {} | {}", nc::kPairSizes[s], issue, wait, first, sent);
    }
    mesh->close();
}

// Transaction-ID sent latency on the router's receiving core (fabric_exp/consolidation/V4_RECEIVER_SHAPE.md §9-10): how
// long after the receiver issues a packet's local writes the NIU reports every write of its transaction ID sent (the
// router's completion proof). ERISC1 with NOC1, as the receiver, writes to a Tensix core's L1 as the receiver's local
// writes: one write of 64 B, 1 KiB, 2 KiB or 4352 B, or a 4352 B scatter in 2 or 4 chunks, from an idle NIU or right
// behind one 4352 B write of another ID to another Tensix core. Per case 256 repetitions, each from an idle NIU: the
// issue time, and the time from the last command's go to the poll of the ID's outgoing-write count that read 0
// (percentiles). Fabric off. Environment: ERISC_LOAD_ETH_INDEX (0), TRID_SENT_CSV.
TEST(EriscLoadOverlap, TridSentLatency) {
    namespace ts = trid_sent_latency;
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
    const char* idx_env = std::getenv("ERISC_LOAD_ETH_INDEX");
    const uint32_t eth_index = idx_env ? std::stoul(idx_env) : 0;
    ASSERT_LT(eth_index, eth_cores.size());
    const CoreCoord logical = eth_cores[eth_index];
    const CoreCoord virt = mesh->virtual_core_from_logical_core(logical, CoreType::ETH);
    const CoreCoord worker = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const CoreCoord other = device->worker_core_from_logical_core(CoreCoord{1, 0});

    const auto& hal = MetalContext::instance().hal();
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    constexpr uint32_t kResultBytes = 0x4000;
    const uint32_t src_base = base + kResultBytes;
    ASSERT_LE(ts::kResultWords * sizeof(uint32_t), kResultBytes);
    ASSERT_LE(kResultBytes + 2 * ts::kMaxBytes, eth_size) << "source buffers do not fit the Ethernet core's free L1";
    // The allocator's L1 base on the Tensix cores: free while no buffer is allocated there.
    const uint32_t dst_base = static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(HalMemType::L1));
    const std::vector<uint32_t> zeros(ts::kResultWords, 0);
    cluster.write_core(zeros.data(), ts::kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    cluster.l1_barrier(device->id());

    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/trid_sent_latency.cpp",
        logical,
        EthernetConfig{
            .noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1, .opt_level = KernelBuildOptLevel::O3});
    SetRuntimeArgs(program, kernel, logical, {base, src_base, worker.x, worker.y, other.x, other.y, dst_base});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
    distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
    distributed::Finish(mesh->mesh_command_queue());
    std::vector<uint32_t> r(ts::kResultWords, 0);
    cluster.read_core(r.data(), ts::kResultWords * sizeof(uint32_t), tt_cxy_pair(device->id(), virt), base);
    ASSERT_EQ(r[ts::kDone], ts::kDoneValue) << "kernel did not finish";
    ASSERT_EQ(r[ts::kFailed], 0u) << "a wait timed out in case " << r[ts::kFailed] - 1;

    std::ofstream csv;
    if (const char* path = std::getenv("TRID_SENT_CSV")) {
        csv.open(path);
        csv << "context,kind,writes,bytes,issue_min,issue_mean,total_min,polls_mean,sent_min,sent_p10,sent_p50,"
               "sent_p90,sent_p99,sent_max,poll_cost\n";
    }
    const char* kind_name[] = {"1 x 64 B", "1 x 1 KiB", "1 x 2 KiB", "1 x 4352 B", "2 x 2176 B", "4 x 1088 B"};
    const char* context_name[] = {"idle NIU", "behind a 4352 B write"};
    log_info(
        LogTest,
        "ERISC1 / NOC1 on Ethernet core logical ({},{}) -> Tensix ({},{}); cycles; one poll that reads 0 at once: {}",
        logical.x,
        logical.y,
        worker.x,
        worker.y,
        r[ts::kPollCost]);
    for (uint32_t context = 0; context < ts::kNumContexts; context++) {
        log_info(
            LogTest,
            "{}: issue (min / mean) | first command to sent (min) | last go to sent: min / p10 / p50 / p90 / p99 / max "
            "| polls",
            context_name[context]);
        for (uint32_t kind = 0; kind < ts::kNumKinds; kind++) {
            const uint32_t at = ts::case_index(kind, context);
            std::vector<uint32_t> sent(r.begin() + at + ts::kNumFields, r.begin() + at + ts::kNumFields + ts::kReps);
            std::sort(sent.begin(), sent.end());
            auto pct = [&](double p) { return sent[static_cast<size_t>(p * (ts::kReps - 1))]; };
            const double issue_mean = static_cast<double>(r[at + ts::ISSUE_SUM]) / ts::kReps;
            const double polls_mean = static_cast<double>(r[at + ts::POLLS]) / ts::kReps;
            log_info(
                LogTest,
                "{:>11} | {:4d} / {:6.1f} | {:4d} | {:4d} / {:4d} / {:4d} / {:4d} / {:4d} / {:4d} | {:4.1f}",
                kind_name[kind],
                r[at + ts::ISSUE_MIN],
                issue_mean,
                r[at + ts::TOTAL_MIN],
                sent.front(),
                pct(0.10),
                pct(0.50),
                pct(0.90),
                pct(0.99),
                sent.back(),
                polls_mean);
            if (csv.is_open()) {
                csv << context << "," << kind << "," << ts::kKindWrites[kind] << "," << ts::kKindBytes[kind] << ","
                    << r[at + ts::ISSUE_MIN] << "," << issue_mean << "," << r[at + ts::TOTAL_MIN] << "," << polls_mean
                    << "," << sent.front() << "," << pct(0.10) << "," << pct(0.50) << "," << pct(0.90) << ","
                    << pct(0.99) << "," << sent.back() << "," << r[ts::kPollCost] << "\n";
            }
        }
    }
    mesh->close();
}
