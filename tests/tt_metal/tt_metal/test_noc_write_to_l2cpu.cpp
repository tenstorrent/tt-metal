// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Tensix NOC write bandwidth to L2CPU tile targets.
//
// Scenarios:
//   1. Tensix (0,0) → L2CPU LIM:        NOC target (8,3), local addr 0x0800_0000
//   2. Tensix (0,0) → L2CPU Front Port: NOC target (8,3), local addr 0x4000_3000_0000
//      NOTE: 0x4000_3000_0000 is 42 bits, which overflows the 36-bit NOC local address
//      field. This scenario is skipped (addressed separately via TLB windows if needed).
//   3. Tensix (0,0) → DRAM tile:        Uses get_noc_addr_from_bank_id baseline.
//
// Uses MeshDeviceSingleCardFixture. Run with:
//   TT_METAL_HOME=$PWD TT_METAL_SLOW_DISPATCH_MODE=1 TT_VISIBLE_DEVICES=1 \
//     ./build_Release/test/tt_metal/unit_tests_legacy \
//     --gtest_filter="*NocWriteToL2cpu*"

#include "common/device_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

using namespace tt::tt_metal;

// L2CPU 0 physical NOC coordinates on Blackhole
static constexpr uint32_t L2CPU_NOC_X = 8;
static constexpr uint32_t L2CPU_NOC_Y = 3;

// L2CPU L3 LIM (SRAM) base — accessible immediately at reset, no init required
static constexpr uint32_t L2CPU_LIM_ADDR = 0x08000000u;

// NOC address coordinate encoding: (y << 6) | x  (NOC_ADDR_NODE_ID_BITS = 6)
static inline uint32_t make_noc_xy_enc(uint32_t x, uint32_t y) { return (y << 6) | x; }

// Layout within kernel result buffer
static constexpr uint32_t RESULT_ELAPSED_IDX = 0;   // uint64_t: wall-clock cycles
static constexpr uint32_t RESULT_BYTES_IDX = 2;     // uint64_t: total bytes transferred
static constexpr uint32_t RESULT_SENTINEL_IDX = 4;  // uint64_t: 0xDEAD0001_xxxxxxxx

// Kernel runtime arg indices (must match noc_write_to_l2cpu.cpp)
static constexpr uint32_t ARG_DST_NOC_XY = 0;
static constexpr uint32_t ARG_DST_LOCAL_ADDR_LO = 1;
static constexpr uint32_t ARG_DST_LOCAL_ADDR_HI = 2;
static constexpr uint32_t ARG_L1_SRC_ADDR = 3;
static constexpr uint32_t ARG_TRANSFER_SIZE = 4;
static constexpr uint32_t ARG_NUM_WRITES = 5;
static constexpr uint32_t ARG_L1_RESULT_ADDR = 6;

struct BenchResult {
    uint64_t elapsed_cycles;
    uint64_t total_bytes;
    double mb_per_sec;
};

struct BenchScenario {
    const char* label;
    uint32_t dst_noc_xy;      // (y<<6)|x
    uint64_t dst_local_addr;  // 36-bit local address for NOC target
    uint32_t transfer_size;
    uint32_t num_writes;
};

// Run a single benchmark scenario on core (0,0).
// Returns {elapsed_cycles, total_bytes, MB/s} or zeroes on sentinel mismatch.
static BenchResult run_scenario(IDevice* dev, uint32_t l1_base, const BenchScenario& sc, double clock_mhz) {
    // L1 layout:
    //   l1_base + 0x0000 : source data buffer  (transfer_size bytes, <= 0x1000)
    //   l1_base + 0x2000 : result region        (3 × uint64_t = 24 bytes)
    uint32_t l1_src_addr = l1_base;
    uint32_t l1_result_addr = l1_base + 0x2000;

    CoreCoord core = {0, 0};

    // Pre-clear result region so sentinel check is reliable
    std::vector<uint32_t> zeros(6, 0);
    detail::WriteToDeviceL1(dev, core, l1_result_addr, zeros);

    Program program = CreateProgram();

    KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_write_to_l2cpu.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });

    uint32_t dst_local_addr_lo = static_cast<uint32_t>(sc.dst_local_addr & 0xFFFFFFFFu);
    uint32_t dst_local_addr_hi = static_cast<uint32_t>((sc.dst_local_addr >> 32) & 0xFu);

    SetRuntimeArgs(
        program,
        kernel,
        core,
        {
            sc.dst_noc_xy,
            dst_local_addr_lo,
            dst_local_addr_hi,
            l1_src_addr,
            sc.transfer_size,
            sc.num_writes,
            l1_result_addr,
        });

    detail::LaunchProgram(dev, program);

    // Read back results (6 × uint32_t = 3 × uint64_t)
    std::vector<uint32_t> raw;
    detail::ReadFromDeviceL1(dev, core, l1_result_addr, 6 * sizeof(uint32_t), raw);

    uint64_t elapsed_cycles = ((uint64_t)raw[RESULT_ELAPSED_IDX + 1] << 32) | raw[RESULT_ELAPSED_IDX];
    uint64_t total_bytes = ((uint64_t)raw[RESULT_BYTES_IDX + 1] << 32) | raw[RESULT_BYTES_IDX];
    uint64_t sentinel = ((uint64_t)raw[RESULT_SENTINEL_IDX + 1] << 32) | raw[RESULT_SENTINEL_IDX];
    uint64_t expected_sentinel = 0xDEAD000100000000ULL | (uint64_t)sc.num_writes;

    if (sentinel != expected_sentinel) {
        printf(
            "  [WARN] sentinel mismatch: got 0x%016llx, expected 0x%016llx\n",
            (unsigned long long)sentinel,
            (unsigned long long)expected_sentinel);
        return {0, 0, 0.0};
    }

    // MB/s = (total_bytes / 1e6) / (elapsed_cycles / clock_hz)
    //       = total_bytes * clock_mhz / elapsed_cycles
    double mb_per_sec = (elapsed_cycles > 0)
                            ? (static_cast<double>(total_bytes) * clock_mhz / static_cast<double>(elapsed_cycles))
                            : 0.0;

    return {elapsed_cycles, total_bytes, mb_per_sec};
}

TEST_F(MeshDeviceSingleCardFixture, NocWriteToL2cpu) {
    IDevice* dev = devices_[0]->get_devices()[0];

    uint32_t l1_base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
    double clock_mhz = static_cast<double>(dev->get_clock_rate_mhz());

    printf("\n=== Tensix NOC Write Bandwidth Benchmark ===\n");
    printf("Device clock: %.0f MHz\n", clock_mhz);
    printf("Source core: Tensix (0,0)\n");
    printf("L1 base: 0x%08x\n", l1_base);
    printf("\n");

    // L2CPU LIM scenarios
    // NOC coord for L2CPU tile 0: (8,3) → xy_enc = (3<<6)|8 = 0xC8
    uint32_t l2cpu_xy = make_noc_xy_enc(L2CPU_NOC_X, L2CPU_NOC_Y);

    // Transfer size → num_writes: enough iterations to amortize overhead
    // and measure steady-state bandwidth
    struct SizeConfig {
        uint32_t size;
        uint32_t count;
    };
    // 128 MB total transfer per scenario (= 134217728 bytes / transfer_size)
    static const SizeConfig kSizes[] = {
        {64, 2097152},
        {256, 524288},
        {1024, 131072},
        {4096, 32768},
        {16384, 8192},
    };

    printf("%-40s  %8s  %10s  %10s  %10s\n", "Scenario", "Size", "Writes", "Cycles", "MB/s");
    printf(
        "%-40s  %8s  %10s  %10s  %10s\n",
        "----------------------------------------",
        "--------",
        "----------",
        "----------",
        "----------");

    bool all_ok = true;

    // --- Scenario 1: Tensix → L2CPU LIM (0x0800_0000) ---
    for (const auto& sz : kSizes) {
        if (sz.size > 0x1000) {
            // Result region starts at l1_base+0x2000; keep src buffer below that
            // and within safe L1 range. Skip sizes that would overflow our buffer layout.
            // (16KB at l1_base would overlap result at l1_base+0x2000 with room to spare.)
            // 16KB = 0x4000 > 0x2000, so it would clobber results. Cap at 4KB.
            continue;
        }
        char label[64];
        snprintf(label, sizeof(label), "L2CPU LIM (8,3) 0x%08x  %5uB", L2CPU_LIM_ADDR, sz.size);

        BenchScenario sc{
            label,
            l2cpu_xy,
            (uint64_t)L2CPU_LIM_ADDR,
            sz.size,
            sz.count,
        };

        BenchResult r = run_scenario(dev, l1_base, sc, clock_mhz);
        printf(
            "%-40s  %8u  %10u  %10llu  %10.1f\n",
            label,
            sz.size,
            sz.count,
            (unsigned long long)r.elapsed_cycles,
            r.mb_per_sec);

        if (r.elapsed_cycles == 0) {
            all_ok = false;
        }
    }

    printf("\n");

    // --- Scenario 2: Tensix → DRAM tile (NOC baseline using bank 0 at offset 0x100000) ---
    // Use get_noc_addr_from_bank_id in the kernel. For this we need to pass the DRAM
    // NOC address directly. We'll use DRAM channel 0, bank 0 NOC coords.
    // Get the DRAM core for bank 0 via the device API.
    {
        // Get virtual coord for DRAM channel 0
        CoreCoord dram_logical = dev->logical_core_from_dram_channel(0);
        CoreCoord dram_virtual = dev->virtual_core_from_logical_core(dram_logical, CoreType::DRAM);
        uint32_t dram_noc_xy = dev->get_noc_unicast_encoding(0, dram_virtual);

        // Safe DRAM offset: use 1MB to avoid any reserved region
        uint64_t dram_local_addr = 0x100000u;

        printf(
            "DRAM channel 0 virtual core: (%zu, %zu), noc_xy_enc=0x%04x\n",
            dram_virtual.x,
            dram_virtual.y,
            dram_noc_xy);

        for (const auto& sz : kSizes) {
            if (sz.size > 0x1000) {
                continue;
            }
            char label[64];
            snprintf(label, sizeof(label), "DRAM ch0 (baseline)         %5uB", sz.size);

            BenchScenario sc{
                label,
                dram_noc_xy,
                dram_local_addr,
                sz.size,
                sz.count,
            };

            BenchResult r = run_scenario(dev, l1_base, sc, clock_mhz);
            printf(
                "%-40s  %8u  %10u  %10llu  %10.1f\n",
                label,
                sz.size,
                sz.count,
                (unsigned long long)r.elapsed_cycles,
                r.mb_per_sec);

            if (r.elapsed_cycles == 0) {
                all_ok = false;
            }
        }
    }

    printf("\n");
    printf("Note: L2CPU DDR front port (0x4000_3000_0000) requires 42-bit local address,\n");
    printf("      which exceeds the 36-bit NOC local address field. Use TLB window\n");
    printf("      remapping or direct DRAM tile access for DDR bandwidth measurements.\n\n");

    EXPECT_TRUE(all_ok) << "One or more benchmark scenarios returned invalid sentinel";
}

// Variant that tests larger transfer sizes (up to 16KB) by moving the result
// region further into L1 so source and result don't collide.
TEST_F(MeshDeviceSingleCardFixture, NocWriteToL2cpuLargeSizes) {
    IDevice* dev = devices_[0]->get_devices()[0];

    uint32_t l1_base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
    double clock_mhz = static_cast<double>(dev->get_clock_rate_mhz());

    // Use a layout that accommodates up to 16KB source:
    //   l1_base + 0x0000 : source data  (up to 0x4000 = 16KB)
    //   l1_base + 0x8000 : result region
    uint32_t l1_src_adj = l1_base;
    uint32_t l1_result_adj = l1_base + 0x8000;

    // We'll set up the scenario manually since run_scenario hardcodes the offsets.
    // For this test, inline the benchmark loop directly.

    uint32_t l2cpu_xy = make_noc_xy_enc(L2CPU_NOC_X, L2CPU_NOC_Y);

    printf("\n=== Tensix NOC Write Bandwidth (Large Sizes) ===\n");
    printf("Device clock: %.0f MHz\n\n", clock_mhz);
    printf("%-40s  %8s  %10s  %10s  %10s\n", "Scenario", "Size", "Writes", "Cycles", "MB/s");
    printf(
        "%-40s  %8s  %10s  %10s  %10s\n",
        "----------------------------------------",
        "--------",
        "----------",
        "----------",
        "----------");

    struct SizeConfig {
        uint32_t size;
        uint32_t count;
    };
    // 128 MB total transfer per scenario
    static const SizeConfig kSizes[] = {
        {4096, 32768},
        {16384, 8192},
    };

    bool all_ok = true;
    CoreCoord core = {0, 0};

    for (const auto& sz : kSizes) {
        // Pre-clear result region
        std::vector<uint32_t> zeros(6, 0);
        detail::WriteToDeviceL1(dev, core, l1_result_adj, zeros);

        Program program = CreateProgram();
        KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_write_to_l2cpu.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            });

        SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                l2cpu_xy,
                L2CPU_LIM_ADDR,
                (uint32_t)0,  // dst_local_addr_hi = 0 (fits in 32 bits)
                l1_src_adj,
                sz.size,
                sz.count,
                l1_result_adj,
            });

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> raw;
        detail::ReadFromDeviceL1(dev, core, l1_result_adj, 6 * sizeof(uint32_t), raw);

        uint64_t elapsed_cycles = ((uint64_t)raw[1] << 32) | raw[0];
        uint64_t total_bytes = ((uint64_t)raw[3] << 32) | raw[2];
        uint64_t sentinel = ((uint64_t)raw[5] << 32) | raw[4];
        uint64_t expected_sent = 0xDEAD000100000000ULL | (uint64_t)sz.count;

        double mb_per_sec = (elapsed_cycles > 0)
                                ? (static_cast<double>(total_bytes) * clock_mhz / static_cast<double>(elapsed_cycles))
                                : 0.0;

        char label[64];
        snprintf(label, sizeof(label), "L2CPU LIM (8,3) 0x%08x %5uB", L2CPU_LIM_ADDR, sz.size);
        printf(
            "%-40s  %8u  %10u  %10llu  %10.1f\n",
            label,
            sz.size,
            sz.count,
            (unsigned long long)elapsed_cycles,
            mb_per_sec);

        if (sentinel != expected_sent) {
            printf("  [WARN] sentinel mismatch: got 0x%016llx\n", (unsigned long long)sentinel);
            all_ok = false;
        }
    }

    printf("\n");
    EXPECT_TRUE(all_ok) << "One or more large-size scenarios returned invalid sentinel";
}
