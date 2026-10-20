// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC microbenchmark for the fabric DRISC-offload study (fabric_exp/model): what it costs a DRISC to move router
// packets, and what it costs an ERISC to hand packets to it. No fabric runs; one program puts a kernel on a DRISC
// and one on ERISC1 of an active Ethernet core (the receiving router's core), and they step through phases together:
//
//   1. DRISC-local: timer, L1 loads (dependent / four independent), fence + L1 poll, stream register reads (one /
//      four independent), local stream update, NoC command-buffer ready read; timer vs wall clock.
//   2. DRISC -> ERISC, on NoC0 and (if this DRISC's NIU1 is in stream mode) NoC1: back-to-back writes of 64 B and of
//      the payload (non-posted, posted), bare doorbells (inline write to an ERISC stream register), a forwarder's
//      copy (payload write with a transaction id + doorbell), the same copy as a stateful write (destination set
//      once), one write at a time with its ack, and pulls (reads of the payload from ERISC L1).
//   3. ERISC-local (the same loads and stream reads, for a like-for-like DRISC/ERISC ratio), then ERISC -> DRISC:
//      payload writes into DRISC L1, bare doorbells, pushes (payload + doorbell), stateful pushes with transaction
//      ids, one write at a time.
//   4. Ping-pong: ERISC doorbell -> DRISC doorbell back, and the same with a payload each way.
//   5. Pipeline, per staging depth: the ERISC pushes packets into a ring of DRISC staging slots, the DRISC forwards
//      each into the ERISC's landing ring (payload + doorbell) and returns slots once its write has left its L1.
//      Run with stateless issue, then with lean stateful issue (lean_trid_write / lean_doorbell) on both cores.
//
// Costs are cycles per operation from the kernels' own timers (mcycle; DRISC_MB_WALL_CLOCK=1 uses the DRISC's
// wall clock instead, and phase 1 reports the ratio). "issue" = cycles to issue back-to-back operations, which for
// payload writes waits for the previous write to leave (the NoC's rate, not a barrier); "drain" = until every one
// has completed; "alone" = one operation at a time with the NoC idle (the wait for the previous one is not timed):
// what it costs the CPU when the port is free, including one timer read (~7).
//
// Environment: DRISC_MB_PAYLOAD (4096), DRISC_MB_N (64), DRISC_MB_N_RT (32), DRISC_MB_PIPE_PKTS (512),
// DRISC_MB_DEPTHS ("1,2,4,8,12,16"), DRISC_MB_DRISC_INDEX (0: the first Metal DRISC whose NIUs are both in stream
// mode), DRISC_MB_ETH_INDEX (0), DRISC_MB_WALL_CLOCK (0), DRISC_MB_OPT (O3, as the router builds; Metal's kernel hash
// omits it, so clear the kernel cache when changing it), DRISC_MB_CSV (write the raw results there).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
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
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/tt_fabric/drisc_microbench/kernels/drisc_mb_common.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace drisc_mb;

constexpr const char* kKernelDir = "tests/tt_metal/tt_fabric/drisc_microbench/kernels/";

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::stoul(v)) : fallback;
}

std::vector<uint32_t> env_list(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    std::stringstream ss(v ? v : fallback);
    std::vector<uint32_t> out;
    for (std::string item; std::getline(ss, item, ',');) {
        out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

KernelBuildOptLevel env_opt_level() {
    const char* v = std::getenv("DRISC_MB_OPT");
    const std::string level = v ? v : "O3";
    if (level == "O2") {
        return KernelBuildOptLevel::O2;
    }
    if (level == "Os") {
        return KernelBuildOptLevel::Os;
    }
    TT_FATAL(level == "O3", "DRISC_MB_OPT must be O2, O3 or Os, not {}", level);
    return KernelBuildOptLevel::O3;
}

struct Entry {
    uint32_t index;
    std::string name;
    uint32_t bytes;  // bytes moved per op (0: not a data-movement entry)
};

struct Pair {
    uint32_t cycles = 0;
    uint32_t ops = 0;
    double per_op() const { return ops ? static_cast<double>(cycles) / ops : 0.0; }
};

}  // namespace

TEST(DriscMicrobench, RouterHandoffCosts) {
    const auto& hal = MetalContext::instance().hal();
    if (MetalContext::instance().get_cluster().arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole only";
    }
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // Metal enables them on firmware bundle >= 19.12; older bundles run syseng firmware on a DRAM core.
        GTEST_SKIP() << "DRAM programmable cores not enabled (needs firmware bundle >= 19.12.0, see "
                        "tt_metal/llrt/firmware_capability.cpp)";
    }
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        // Fast dispatch has no write offset for DRAM cores (program/dispatch.cpp); DRISC tests run slow dispatch.
        GTEST_SKIP() << "DRISC programs need TT_METAL_SLOW_DISPATCH_MODE=1";
    }

    const uint32_t payload = env_u32("DRISC_MB_PAYLOAD", 4096);
    const uint32_t n = env_u32("DRISC_MB_N", 64);
    const uint32_t n_rt = env_u32("DRISC_MB_N_RT", 32);
    const uint32_t pipe_pkts = env_u32("DRISC_MB_PIPE_PKTS", 512);
    // In dual-ERISC mode Metal requires ERISC1 kernels to use NoC1, as the receiving router does.
    constexpr uint32_t erisc_noc = 1;
    const uint32_t drisc_index = env_u32("DRISC_MB_DRISC_INDEX", 0);
    const uint32_t eth_index = env_u32("DRISC_MB_ETH_INDEX", 0);
    const bool wall_clock = env_u32("DRISC_MB_WALL_CLOCK", 0) != 0;
    const KernelBuildOptLevel opt_level = env_opt_level();
    std::vector<uint32_t> depths = env_list("DRISC_MB_DEPTHS", "1,2,4,8,12,16");
    const std::vector<uint32_t> parts = env_list("DRISC_MB_PARTS", "1,2,3,4");
    auto runs = [&](uint32_t part) { return std::find(parts.begin(), parts.end(), part) != parts.end(); };
    // Part 4 variants 3 and 4 (both DRISCs at once on different NoCs) hung the chip; they run only when asked for.
    const uint32_t two_mask = env_u32("DRISC_MB_TWO_MASK", ((1u << kTwoDriscVariants) - 1) & ~0b11000u);
    const bool two_spin = env_u32("DRISC_MB_TWO_SPIN", 0) != 0;
    ASSERT_LE(payload, kSlotBytes);
    ASSERT_GT(payload, 0u);
    ASSERT_LE(depths.size(), kMaxDepths);

    // Fabric must stay off: its routers take the ERISCs we launch on. We never set a fabric config, but opening a
    // device with fast dispatch to remote chips (or on a Galaxy) turns fabric on for dispatch.
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    if (MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::DISABLED) {
        mesh_device->close();
        GTEST_FAIL() << "fabric was enabled (for dispatch to remote chips?); its routers would occupy the ERISCs. "
                        "Run on an MMIO chip without remote chips, or with TT_METAL_SLOW_DISPATCH_MODE=1";
    }
    IDevice* device = mesh_device->get_devices()[0];
    const ChipId chip = device->id();
    const auto& cluster = MetalContext::instance().get_cluster();
    const metal_SocDescriptor& soc = cluster.get_soc_desc(chip);

    // ERISC: an active Ethernet core.
    std::vector<CoreCoord> eth_cores;
    for (const auto& c : device->get_active_ethernet_cores(true)) {
        eth_cores.push_back(c);
    }
    if (eth_cores.empty()) {
        mesh_device->close();
        GTEST_SKIP() << "no active Ethernet cores";
    }
    std::sort(eth_cores.begin(), eth_cores.end());
    ASSERT_LT(eth_index, eth_cores.size());
    const CoreCoord eth_logical = eth_cores[eth_index];
    const CoreCoord eth_virtual = mesh_device->virtual_core_from_logical_core(eth_logical, CoreType::ETH);
    // Up to two other active Ethernet cores for part 3's forwarding tests: the DRISC writes into their landing rings,
    // as a router forwards to its downstream routers. They run no kernel.
    std::vector<CoreCoord> fwd_logical, fwd_virtual;
    for (uint32_t i = 1; i < eth_cores.size() && fwd_logical.size() < 2; i++) {
        fwd_logical.push_back(eth_cores[(eth_index + i) % eth_cores.size()]);
        fwd_virtual.push_back(mesh_device->virtual_core_from_logical_core(fwd_logical.back(), CoreType::ETH));
    }

    // DRISC: a Metal DRISC whose NIU on the ERISC's NoC is in stream mode (an NIU in NOC2AXI mode would route the
    // ERISC's plain L1 addresses to GDDR). Prefer one with both NIUs in stream mode, so both NoCs can be measured.
    std::vector<std::pair<CoreCoord, uint8_t>> driscs;  // (logical, NOC2AXI mask)
    for (const auto& logical : soc.get_metal_dram_cores(tt::CoordSystem::LOGICAL)) {
        const uint8_t mask = soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical));
        if ((mask & (1u << erisc_noc)) == 0) {
            driscs.emplace_back(logical, mask);
        }
    }
    std::stable_sort(driscs.begin(), driscs.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    ASSERT_LT(drisc_index, driscs.size()) << "no Metal DRISC with NIU" << erisc_noc << " in stream mode";
    const CoreCoord drisc_logical = driscs[drisc_index].first;
    const uint8_t drisc_mask = driscs[drisc_index].second;
    ASSERT_EQ(drisc_mask & 0b1, 0) << "Metal DRISCs keep NIU0 in stream mode";
    const bool drisc_noc1 = (drisc_mask & 0b10) == 0;
    const CoreCoord drisc_virtual = mesh_device->virtual_core_from_logical_core(drisc_logical, CoreType::DRAM);

    // L1: both cores' UNRESERVED regions. DRISC L1 is reached from the host through its tagged NoC address.
    const uint32_t drisc_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t drisc_host_base = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t drisc_size = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    ASSERT_GE(eth_size, kEthL1Bytes);
    const uint32_t drisc_slots = std::min(kMaxDriscSlots, (drisc_size - kBufferOffset) / kSlotBytes);
    ASSERT_GE(drisc_slots, 1u);
    for (uint32_t& d : depths) {
        d = std::clamp(d, 1u, drisc_slots);
    }

    log_info(
        LogTest,
        "DRISC logical ({},{}) virtual ({},{}) NOC2AXI mask {:#x}, {} staging slots of {} B; ERISC1 on Ethernet core "
        "logical ({},{}) virtual ({},{}), NoC{}; payload {} B; DRISC timer {}",
        drisc_logical.x,
        drisc_logical.y,
        drisc_virtual.x,
        drisc_virtual.y,
        drisc_mask,
        drisc_slots,
        kSlotBytes,
        eth_logical.x,
        eth_logical.y,
        eth_virtual.x,
        eth_virtual.y,
        erisc_noc,
        payload,
        wall_clock ? "wall clock" : "mcycle");

    std::vector<uint32_t> pattern(kSlotBytes / sizeof(uint32_t));
    for (uint32_t i = 0; i < pattern.size(); i++) {
        pattern[i] = 0xA5000000u | i;
    }
    cluster.write_core(pattern.data(), kSlotBytes, tt_cxy_pair(chip, eth_virtual), eth_base + kEthSrcOffset);

    std::vector<uint32_t> common_args(ARG_COUNT, 0);
    common_args[ARG_PAYLOAD] = payload;
    common_args[ARG_N] = n;
    common_args[ARG_N_RT] = n_rt;
    common_args[ARG_DRISC_NOC1] = drisc_noc1;
    common_args[ARG_DRISC_SLOTS] = drisc_slots;
    common_args[ARG_PIPE_PKTS] = pipe_pkts;
    common_args[ARG_FWD_ERISCS] = fwd_virtual.size();
    common_args[ARG_TWO_MASK] = two_mask;
    for (uint32_t i = 0; i < fwd_virtual.size(); i++) {
        common_args[ARG_FWD0_X + 2 * i] = fwd_virtual[i].x;
        common_args[ARG_FWD0_Y + 2 * i] = fwd_virtual[i].y;
    }
    common_args[ARG_NUM_DEPTHS] = depths.size();
    for (uint32_t d = 0; d < depths.size(); d++) {
        common_args[ARG_DEPTHS + d] = depths[d];
    }
    std::vector<uint32_t> drisc_args = common_args;
    drisc_args[ARG_MY_BASE] = drisc_base;
    drisc_args[ARG_PEER_X] = eth_virtual.x;
    drisc_args[ARG_PEER_Y] = eth_virtual.y;
    drisc_args[ARG_PEER_BASE] = eth_base;
    std::vector<uint32_t> eth_args = common_args;
    eth_args[ARG_MY_BASE] = eth_base;
    eth_args[ARG_PEER_X] = drisc_virtual.x;
    eth_args[ARG_PEER_Y] = drisc_virtual.y;
    eth_args[ARG_PEER_BASE] = drisc_base;

    auto read_results = [&](const CoreCoord& core, uint64_t addr) {
        std::vector<uint32_t> words(2 * kMaxResults);
        cluster.read_core(words.data(), words.size() * sizeof(uint32_t), tt_cxy_pair(chip, core), addr);
        std::vector<Pair> out(kMaxResults);
        for (uint32_t i = 0; i < kMaxResults; i++) {
            out[i] = {words[2 * i], words[2 * i + 1]};
        }
        return out;
    };

    // The kernels run as three programs (see DRISC_MB_PART in drisc_mb_device.hpp), each starting from zeroed
    // control words, results and landing slot. Part 1 fills every result but the pipelines; part 2 fills the
    // pipelines; part 3 (the DRISC's command buffers and NoC ports) has its own result layout, kept in `dc`.
    std::vector<Pair> dr(kMaxResults), er(kMaxResults), dc(kMaxResults);
    for (uint32_t part = 1; part <= 3; part++) {
        if (!runs(part)) {
            continue;
        }
        const std::vector<uint32_t> zeros(kBufferOffset / sizeof(uint32_t), 0);
        cluster.write_core(zeros.data(), kBufferOffset, tt_cxy_pair(chip, drisc_virtual), drisc_host_base);
        cluster.write_core(zeros.data(), kBufferOffset, tt_cxy_pair(chip, eth_virtual), eth_base);
        const std::vector<uint32_t> landing_zeros(kSlotBytes / sizeof(uint32_t), 0);
        cluster.write_core(
            landing_zeros.data(), kSlotBytes, tt_cxy_pair(chip, eth_virtual), eth_base + kEthLandingOffset);
        for (const CoreCoord& f : fwd_virtual) {
            cluster.write_core(landing_zeros.data(), kSlotBytes, tt_cxy_pair(chip, f), eth_base + kEthLandingOffset);
        }
        cluster.l1_barrier(chip);

        Program program = CreateProgram();
        std::map<std::string, std::string> defines = {{"DRISC_MB_PART", std::to_string(part)}};
        std::map<std::string, std::string> drisc_defines = defines;
        if (wall_clock) {
            drisc_defines["DRISC_MB_WALL_CLOCK"] = "1";
        }
        const KernelHandle drisc_kernel = CreateKernel(
            program,
            std::string(kKernelDir) + "drisc_mb_drisc.cpp",
            drisc_logical,
            DramConfig{.noc = NOC::NOC_0, .defines = drisc_defines, .opt_level = opt_level});
        SetRuntimeArgs(program, drisc_kernel, drisc_logical, drisc_args);
        const KernelHandle eth_kernel = CreateKernel(
            program,
            std::string(kKernelDir) + "drisc_mb_erisc.cpp",
            eth_logical,
            EthernetConfig{
                .noc = NOC::NOC_1,
                .processor = DataMovementProcessor::RISCV_1,
                .defines = defines,
                .opt_level = opt_level});
        SetRuntimeArgs(program, eth_kernel, eth_logical, eth_args);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        log_info(LogTest, "Part {} finished", part);

        const std::vector<Pair> dp = read_results(drisc_virtual, drisc_host_base + kResultsOffset);
        const std::vector<Pair> ep = read_results(eth_virtual, eth_base + kResultsOffset);
        for (uint32_t i = 0; part < 3 && i < kMaxResults; i++) {
            if ((part == 1) == (i < D_PIPE_BASE)) {
                dr[i] = dp[i];
            }
            if ((part == 1) == (i < E_PIPE_BASE)) {
                er[i] = ep[i];
            }
        }
        if (part == 3) {
            dc = dp;
        }
        // Error words hold the phase + 1 of the first wait that timed out; phase 100/101 on the DRISC: its NIU0/NIU1
        // was not in stream mode.
        EXPECT_EQ(dp[D_ERROR].cycles, 0u)
            << "part " << part << ": DRISC kernel gave up in phase " << dp[D_ERROR].cycles - 1;
        EXPECT_EQ(ep[E_ERROR].cycles, 0u)
            << "part " << part << ": ERISC kernel gave up in phase " << ep[E_ERROR].cycles - 1;
        if (part == 3) {
            // The DRISC's copies into the other Ethernet cores carry A's source pattern.
            for (uint32_t f = 0; f < fwd_virtual.size(); f++) {
                std::vector<uint32_t> landed(kSlotBytes / sizeof(uint32_t));
                cluster.read_core(
                    landed.data(), kSlotBytes, tt_cxy_pair(chip, fwd_virtual[f]), eth_base + kEthLandingOffset);
                const uint32_t words = payload / sizeof(uint32_t);
                const bool multicast_ran = f == 0 || drisc_noc1;
                EXPECT_TRUE(!multicast_ran || std::equal(landed.begin(), landed.begin() + words, pattern.begin()))
                    << "Ethernet core " << fwd_logical[f].str() << " did not receive the forwarded pattern";
            }
            continue;
        }
        // Forwarded data must match what the ERISC pushed.
        std::vector<uint32_t> landed(kSlotBytes / sizeof(uint32_t));
        cluster.read_core(landed.data(), kSlotBytes, tt_cxy_pair(chip, eth_virtual), eth_base + kEthLandingOffset);
        const uint32_t words = payload / sizeof(uint32_t);
        EXPECT_TRUE(std::equal(landed.begin(), landed.begin() + words, pattern.begin()))
            << "part " << part << ": the landing slot does not hold the pushed pattern";
    }

    // Part 4: two DRISCs on one Ethernet tile, as when its link carries traffic both ways. DRISC 1 (the one above)
    // pulls from the ERISC's source buffer, DRISC 2 writes into its landing ring. No ERISC kernel runs.
    std::vector<Pair> d4_pull(kMaxResults), d4_write(kMaxResults);
    CoreCoord drisc2_logical;
    const bool two_driscs = driscs.size() > 1 && runs(4);
    if (two_driscs) {
        const auto& [d2_logical, d2_mask] = driscs[(drisc_index + 1) % driscs.size()];
        drisc2_logical = d2_logical;
        const bool d2_noc1 = (d2_mask & 0b10) == 0;
        const CoreCoord d2_virtual = mesh_device->virtual_core_from_logical_core(d2_logical, CoreType::DRAM);
        const std::vector<uint32_t> zeros(kBufferOffset / sizeof(uint32_t), 0);
        cluster.write_core(zeros.data(), kBufferOffset, tt_cxy_pair(chip, drisc_virtual), drisc_host_base);
        cluster.write_core(zeros.data(), kBufferOffset, tt_cxy_pair(chip, d2_virtual), drisc_host_base);
        cluster.l1_barrier(chip);

        auto role_args = [&](uint32_t role, const CoreCoord& peer, bool noc1) {
            std::vector<uint32_t> a = common_args;
            a[ARG_MY_BASE] = drisc_base;
            a[ARG_PEER_X] = peer.x;
            a[ARG_PEER_Y] = peer.y;
            a[ARG_PEER_BASE] = drisc_base;
            a[ARG_DRISC_NOC1] = noc1;
            a[ARG_ROLE] = role;
            a[ARG_TARGET_X] = eth_virtual.x;
            a[ARG_TARGET_Y] = eth_virtual.y;
            a[ARG_TARGET_BASE] = eth_base;
            return a;
        };
        Program program = CreateProgram();
        std::map<std::string, std::string> defines = {{"DRISC_MB_PART", "4"}};
        if (wall_clock) {
            defines["DRISC_MB_WALL_CLOCK"] = "1";
        }
        const std::string src = std::string(kKernelDir) + "drisc_mb_drisc.cpp";
        const KernelHandle k1 = CreateKernel(
            program, src, drisc_logical, DramConfig{.noc = NOC::NOC_0, .defines = defines, .opt_level = opt_level});
        SetRuntimeArgs(program, k1, drisc_logical, role_args(0, d2_virtual, drisc_noc1));
        const KernelHandle k2 = CreateKernel(
            program, src, d2_logical, DramConfig{.noc = NOC::NOC_0, .defines = defines, .opt_level = opt_level});
        SetRuntimeArgs(program, k2, d2_logical, role_args(1, drisc_virtual, d2_noc1));
        if (two_spin) {
            const KernelHandle ke = CreateKernel(
                program,
                std::string(kKernelDir) + "drisc_mb_erisc.cpp",
                eth_logical,
                EthernetConfig{
                    .noc = NOC::NOC_1,
                    .processor = DataMovementProcessor::RISCV_1,
                    .defines = defines,
                    .opt_level = opt_level});
            SetRuntimeArgs(program, ke, eth_logical, eth_args);
        }
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        log_info(LogTest, "Part 4 finished");
        d4_pull = read_results(drisc_virtual, drisc_host_base + kResultsOffset);
        d4_write = read_results(d2_virtual, drisc_host_base + kResultsOffset);
        EXPECT_EQ(d4_pull[D_ERROR].cycles, 0u) << "part 4: DRISC 1 gave up in phase " << d4_pull[D_ERROR].cycles - 1;
        EXPECT_EQ(d4_write[D_ERROR].cycles, 0u) << "part 4: DRISC 2 gave up in phase " << d4_write[D_ERROR].cycles - 1;
    }
    mesh_device->close();

    std::vector<Entry> d_entries = {
        {D_TIMER_READ, "timer read", 0},
        {D_WALL_CLOCK_READ, "wall clock read", 0},
        {D_L1_LOAD_DEP, "L1 load, dependent", 0},
        {D_L1_LOAD_INDEP4, "L1 load, 4 independent (per load)", 0},
        {D_L1_POLL_FENCE, "fence + L1 load (counter poll)", 0},
        {D_STREAM_READ, "stream register read (doorbell poll)", 0},
        {D_STREAM_READ_INDEP4, "stream register read, 4 independent (per read)", 0},
        {D_STREAM_LOCAL_UPDATE, "local stream register update", 0},
        {D_CMD_BUF_READY, "NoC cmd buffer ready read", 0},
    };
    for (uint32_t wn = 0; wn < 2; wn++) {
        if (wn == 1 && !drisc_noc1) {
            continue;
        }
        const uint32_t b = D_NOC_BASE + wn * DN_COUNT;
        const std::string p = "NoC" + std::to_string(wn) + " ";
        d_entries.insert(
            d_entries.end(),
            {
                {b + DN_WR64_ISSUE, p + "write 64 B -> ERISC, issue", 0},
                {b + DN_WR64_DRAIN, p + "write 64 B -> ERISC, drain", 64},
                {b + DN_WR_ISSUE, p + "write payload -> ERISC, issue", 0},
                {b + DN_WR_DRAIN, p + "write payload -> ERISC, drain", payload},
                {b + DN_WR_POSTED_ISSUE, p + "posted write payload, issue", 0},
                {b + DN_WR_POSTED_DRAIN, p + "posted write payload, drain", payload},
                {b + DN_DOORBELL_ISSUE, p + "doorbell -> ERISC stream reg, issue", 0},
                {b + DN_DOORBELL_DRAIN, p + "doorbell -> ERISC stream reg, drain", 0},
                {b + DN_FWD_ISSUE, p + "forward copy (payload + doorbell), issue", 0},
                {b + DN_FWD_DRAIN, p + "forward copy (payload + doorbell), drain", payload},
                {b + DN_SF_FWD_ISSUE, p + "forward copy, stateful + trid, issue", 0},
                {b + DN_SF_FWD_DRAIN, p + "forward copy, stateful + trid, drain", payload},
                {b + DN_SF_FWD_ALONE, p + "forward copy, stateful + trid, alone", 0},
                {b + DN_DOORBELL_ALONE, p + "doorbell -> ERISC stream reg, alone", 0},
                {b + DN_LEAN_FWD_ISSUE, p + "forward copy, lean stateful, issue", 0},
                {b + DN_LEAN_FWD_DRAIN, p + "forward copy, lean stateful, drain", payload},
                {b + DN_LEAN_FWD_ALONE, p + "forward copy, lean stateful, alone", 0},
                {b + DN_WR_RTT, p + "write payload + wait for ack", 0},
                {b + DN_RD_ISSUE, p + "read payload <- ERISC, issue", 0},
                {b + DN_RD_DRAIN, p + "read payload <- ERISC, drain", payload},
                {b + DN_RD_RTT, p + "read payload + wait for data", 0},
            });
    }
    const std::vector<Entry> e_entries = {
        {E_TIMER_READ, "timer read", 0},
        {E_L1_LOAD_DEP, "L1 load, dependent", 0},
        {E_L1_LOAD_INDEP4, "L1 load, 4 independent (per load)", 0},
        {E_STREAM_READ, "stream register read", 0},
        {E_STREAM_READ_INDEP4, "stream register read, 4 independent (per read)", 0},
        {E_WR_ISSUE, "write payload -> DRISC, issue", 0},
        {E_WR_DRAIN, "write payload -> DRISC, drain", payload},
        {E_DOORBELL_ISSUE, "doorbell -> DRISC stream reg, issue", 0},
        {E_DOORBELL_DRAIN, "doorbell -> DRISC stream reg, drain", 0},
        {E_PUSH_ISSUE, "push (payload + doorbell), issue", 0},
        {E_PUSH_DRAIN, "push (payload + doorbell), drain", payload},
        {E_SF_PUSH_ISSUE, "push, stateful + trid, issue", 0},
        {E_SF_PUSH_DRAIN, "push, stateful + trid, drain", payload},
        {E_SF_PUSH_ALONE, "push, stateful + trid, alone", 0},
        {E_SF_WR_ALONE, "write payload, stateful + trid, alone", 0},
        {E_WR_ALONE, "write payload -> DRISC, alone", 0},
        {E_DOORBELL_ALONE, "doorbell -> DRISC stream reg, alone", 0},
        {E_PUSH_ALONE, "push (payload + doorbell), alone", 0},
        {E_LEAN_PUSH_ISSUE, "push, lean stateful, issue", 0},
        {E_LEAN_PUSH_DRAIN, "push, lean stateful, drain", payload},
        {E_LEAN_PUSH_ALONE, "push, lean stateful, alone", 0},
        {E_LEAN_WR_ALONE, "write payload, lean stateful, alone", 0},
        {E_LEAN_DOORBELL_ALONE, "doorbell, lean stateful, alone", 0},
        {E_WR_RTT, "write payload + wait for ack", 0},
        {E_PING_RTT, "ping: doorbell there and back", 0},
        {E_PING_DATA_RTT, "ping: payload + doorbell there and back", 0},
    };

    std::ofstream csv;
    if (const char* path = std::getenv("DRISC_MB_CSV")) {
        csv.open(path);
        csv << "core,measurement,cycles,ops,cycles_per_op\n";
    }
    auto print = [&](const char* core, const std::vector<Entry>& entries, const std::vector<Pair>& res) {
        for (const auto& e : entries) {
            const Pair& r = res[e.index];
            if (e.bytes && r.per_op() > 0) {
                log_info(
                    LogTest,
                    "{:6s} {:50s} {:9.1f} cycles/op  {:6.1f} B/cycle",
                    core,
                    e.name,
                    r.per_op(),
                    e.bytes / r.per_op());
            } else {
                log_info(LogTest, "{:6s} {:50s} {:9.1f} cycles/op", core, e.name, r.per_op());
            }
            if (csv.is_open()) {
                csv << core << ",\"" << e.name << "\"," << r.cycles << "," << r.ops << "," << r.per_op() << "\n";
            }
        }
    };
    if (runs(1)) {
        print("DRISC", d_entries, dr);
        print("ERISC", e_entries, er);
        const Pair calib = dr[D_CALIBRATION];
        log_info(
            LogTest, "DRISC timer ticks per wall-clock tick: {:.3f}", calib.ops ? double(calib.cycles) / calib.ops : 0);
    }

    const char* const mode_names[kPipeModes] = {"stateless", "lean"};
    for (uint32_t m = 0; runs(2) && m < kPipeModes; m++) {
        log_info(
            LogTest,
            "Pipeline, {} issue (ERISC pushes {} packets of {} B through DRISC staging, DRISC forwards them back):",
            mode_names[m],
            pipe_pkts,
            payload);
        log_info(
            LogTest,
            "  depth  cycles/pkt  ERISC busy/pkt  DRISC busy/pkt  DRISC work/pkt  DRISC idle polls/pkt  "
            "ERISC full-ring polls/pkt");
        for (uint32_t d = 0; d < depths.size(); d++) {
            const uint32_t eb = E_PIPE_BASE + (m * kMaxDepths + d) * EP_COUNT;
            const uint32_t db = D_PIPE_BASE + (m * kMaxDepths + d) * DP_COUNT;
            const Pair total = er[eb + EP_TOTAL];
            const Pair e_busy = er[eb + EP_BUSY];
            const Pair waits = er[eb + EP_CREDIT_WAITS];
            const Pair d_busy = dr[db + DP_BUSY];
            const Pair d_work = dr[db + DP_WORK];
            const Pair idle = dr[db + DP_IDLE_POLLS];
            log_info(
                LogTest,
                "  {:5d}  {:10.1f}  {:14.1f}  {:14.1f}  {:14.1f}  {:20.2f}  {:25.2f}",
                depths[d],
                total.per_op(),
                e_busy.per_op(),
                d_busy.per_op(),
                d_work.per_op(),
                idle.per_op(),
                waits.per_op());
            if (csv.is_open()) {
                const std::string tag =
                    std::string("pipeline ") + mode_names[m] + " depth " + std::to_string(depths[d]);
                auto row = [&](const char* core, const char* what, const Pair& r) {
                    csv << core << ",\"" << tag << " " << what << "\"," << r.cycles << "," << r.ops << "," << r.per_op()
                        << "\n";
                };
                row("ERISC", "total", total);
                row("ERISC", "busy", e_busy);
                row("ERISC", "full-ring polls", waits);
                row("DRISC", "busy", d_busy);
                row("DRISC", "work", d_work);
                row("DRISC", "idle polls", idle);
            }
        }
    }

    // Part 3: the DRISC's command buffers and NoC ports.
    auto per_op = [&](uint32_t i) { return dc[i].per_op(); };
    auto rate = [&](uint32_t i) { return per_op(i) > 0 ? payload / per_op(i) : 0.0; };
    auto csv_row = [&](const std::string& what, const Pair& r) {
        if (csv.is_open()) {
            csv << "DRISC,\"" << what << "\"," << r.cycles << "," << r.ops << "," << r.per_op() << "\n";
        }
    };
    if (runs(3)) {
        log_info(
            LogTest,
            "DRISC command buffers and NoC ports ({} B writes into the ERISC's landing ring, back to back):",
            payload);
        for (uint32_t wn = 0; wn < (drisc_noc1 ? 2u : 1u); wn++) {
            const uint32_t b = D_NOC_BASE + wn * DCN_COUNT;
            const std::string p = "NoC" + std::to_string(wn);
            log_info(
                LogTest,
                "  {}: one write with the NoC idle holds its command buffer {:.1f} cycles",
                p,
                per_op(b + DCN_HOLD));
            csv_row(p + " command buffer hold, one write", dc[b + DCN_HOLD]);
            for (uint32_t k = 1; k <= kMaxWriteCmdBufs; k++) {
                const uint32_t e = b + DCN_WR_K + 2 * (k - 1);
                log_info(
                    LogTest,
                    "  {}: {} command buffer(s): issue {:6.1f} cycles/write, drain {:6.1f} ({:5.1f} B/cycle)",
                    p,
                    k,
                    per_op(e),
                    per_op(e + 1),
                    rate(e + 1));
                csv_row(p + " writes over " + std::to_string(k) + " command buffers, issue", dc[e]);
                csv_row(p + " writes over " + std::to_string(k) + " command buffers, drain", dc[e + 1]);
            }
        }
        if (drisc_noc1) {
            for (uint32_t k = 1; k <= 2; k++) {
                const uint32_t e = DC_BOTH + 2 * (k - 1);
                log_info(
                    LogTest,
                    "  NoC0 + NoC1 alternating, {} command buffer(s) each: issue {:6.1f} cycles/write, drain {:6.1f} "
                    "({:5.1f} B/cycle in total)",
                    k,
                    per_op(e),
                    per_op(e + 1),
                    rate(e + 1));
                csv_row("NoC0 + NoC1 writes, " + std::to_string(k) + " command buffers each, issue", dc[e]);
                csv_row("NoC0 + NoC1 writes, " + std::to_string(k) + " command buffers each, drain", dc[e + 1]);
            }
            const char* const duplex[kDuplexVariants] = {
                "pull NoC0 + forward NoC0, 1 write buffer",
                "pull NoC0 + forward NoC0, 2 write buffers",
                "pull NoC0 + forward NoC1, 1 write buffer",
                "pull NoC0 + forward NoC1, 2 write buffers"};
            for (uint32_t v = 0; v < kDuplexVariants; v++) {
                const uint32_t e = DC_DUPLEX + 2 * v;
                log_info(
                    LogTest,
                    "  {}: issue {:6.1f} cycles/pair, drain {:6.1f} ({:5.1f} B/cycle each way)",
                    duplex[v],
                    per_op(e),
                    per_op(e + 1),
                    rate(e + 1));
                csv_row(std::string(duplex[v]) + ", issue per pair", dc[e]);
                csv_row(std::string(duplex[v]) + ", drain per pair", dc[e + 1]);
            }
        }

        if (!fwd_virtual.empty()) {
            log_info(
                LogTest,
                "Pull from ERISC A ({},{}) on NoC0, forward into other Ethernet cores' L1 (B ({},{}){}), per packet:",
                eth_logical.x,
                eth_logical.y,
                fwd_logical[0].x,
                fwd_logical[0].y,
                fwd_logical.size() > 1 ? fmt::format(", C ({},{})", fwd_logical[1].x, fwd_logical[1].y) : "");
            const char* const fwd_names[kFwdVariants] = {
                "copy to B on NoC0 only (reference)",
                "pull + copy to B, NoC0",
                "pull + copy to B on NoC1",
                "pull + copies to B on NoC0, C on NoC1",
                "pull + copies to B and C on NoC1"};
            for (uint32_t v = 0; v < kFwdVariants; v++) {
                const FwdVariant fv = kFwdVariantList[v];
                const uint32_t e = DC_FWD + 2 * v;
                if (dc[e + 1].ops == 0) {
                    continue;  // not run: needs NoC1 or a second Ethernet core
                }
                const uint32_t copies = fv.c_noc >= 0 ? 2 : 1;
                log_info(
                    LogTest,
                    "  {:40s} issue {:6.1f}, drain {:6.1f} cycles/packet: in {:5.1f}, out {:5.1f} B/cycle",
                    fwd_names[v],
                    per_op(e),
                    per_op(e + 1),
                    fv.pull ? rate(e + 1) : 0.0,
                    copies * rate(e + 1));
                csv_row(std::string(fwd_names[v]) + ", issue per packet", dc[e]);
                csv_row(std::string(fwd_names[v]) + ", drain per packet", dc[e + 1]);
            }
        }
    }

    if (two_driscs) {
        log_info(
            LogTest,
            "Two DRISCs on Ethernet core A ({},{}): DRISC 1 ({},{}) pulls its source buffer, DRISC 2 ({},{}) writes "
            "its landing ring; {} packets of {} B each; ERISC {}:",
            eth_logical.x,
            eth_logical.y,
            drisc_logical.x,
            drisc_logical.y,
            drisc2_logical.x,
            drisc2_logical.y,
            kTwoDriscOpsScale * n,
            payload,
            two_spin ? "polling a stream register" : "idle (no kernel)");
        const char* const names[kTwoDriscVariants] = {
            "pull alone, NoC0",
            "write alone, NoC0",
            "both, same NoC (NoC0)",
            "both, pull NoC0 + write NoC1",
            "both, pull NoC1 + write NoC0",
            "DRISC 1 pulls + writes, NoC0",
            "DRISC 1 pulls NoC0 + writes NoC1"};
        for (uint32_t v = 0; v < kTwoDriscVariants; v++) {
            const Pair& pr = d4_pull[D_NOC_BASE + v];
            const Pair& wr = d4_write[D_NOC_BASE + v];
            auto side = [&](const Pair& r) {
                return r.ops ? fmt::format("{:6.1f} cycles/packet ({:5.1f} B/cycle)", r.per_op(), payload / r.per_op())
                             : std::string("-");
            };
            log_info(LogTest, "  {:30s} pull: {:36s} write: {}", names[v], side(pr), side(wr));
            if (csv.is_open()) {
                for (const auto& [what, r] : {std::pair{"pull", pr}, std::pair{"write", wr}}) {
                    if (r.ops) {
                        csv << "DRISC,\"two DRISCs on one Ethernet tile" << (two_spin ? " (ERISC polling), " : ", ")
                            << names[v] << ", " << what << "\"," << r.cycles << "," << r.ops << "," << r.per_op()
                            << "\n";
                    }
                }
            }
        }
    }

    // Like-for-like DRISC/ERISC ratios (the model's alpha), from the same loops on both cores.
    auto ratio = [](const Pair& a, const Pair& b) { return b.per_op() > 0 ? a.per_op() / b.per_op() : 0.0; };
    log_info(
        LogTest,
        "DRISC/ERISC: dependent L1 load {:.2f}, stream register read {:.2f}",
        ratio(dr[D_L1_LOAD_DEP], er[E_L1_LOAD_DEP]),
        ratio(dr[D_STREAM_READ], er[E_STREAM_READ]));

    // Every doorbell must have landed.
    EXPECT_EQ(er[E_CHECK_FWD_DOORBELLS].cycles, er[E_CHECK_FWD_DOORBELLS].ops);
    EXPECT_EQ(er[E_CHECK_BARE_DOORBELLS].cycles, er[E_CHECK_BARE_DOORBELLS].ops);
    EXPECT_EQ(dr[D_CHECK_PUSH_DOORBELLS].cycles, dr[D_CHECK_PUSH_DOORBELLS].ops);
    EXPECT_EQ(dr[D_CHECK_BARE_DOORBELLS].cycles, dr[D_CHECK_BARE_DOORBELLS].ops);
}
