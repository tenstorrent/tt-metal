// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal test: dispatch a dataflow kernel that does noc_async_write
// and dumps all NOC0 command buffer registers to L1 for analysis.
// Used to capture the real register state for L2CPU NIU replay experiments.

#include "common/device_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tt::tt_metal;

// Must match kernel's OFFSETS array
static const char* REG_NAMES[] = {
    "NOC_TARG_ADDR_LO",      // 0x00
    "NOC_TARG_ADDR_MID",     // 0x04
    "NOC_TARG_ADDR_HI",      // 0x08
    "NOC_RET_ADDR_LO",       // 0x0C
    "NOC_RET_ADDR_MID",      // 0x10
    "NOC_RET_ADDR_HI",       // 0x14
    "NOC_PACKET_TAG",        // 0x18
    "NOC_CTRL",              // 0x1C
    "NOC_AT_LEN_BE",         // 0x20
    "NOC_AT_LEN_BE_1",       // 0x24
    "NOC_AT_DATA",           // 0x28
    "NOC_BRCST_EXCLUDE",     // 0x2C
    "NOC_L1_ACC_AT_INSTRN",  // 0x30
    "NOC_SEC_CTRL",          // 0x34
    "NOC_CMD_CTRL",          // 0x40
    "NOC_NODE_ID",           // 0x44
    "NOC_ENDPOINT_ID",       // 0x48
};
static constexpr uint32_t NUM_REGS = 17;
static constexpr uint32_t DUMP_ENTRY_SIZE = (NUM_REGS + 1) * sizeof(uint32_t);  // regs + sentinel

static void print_dump(const char* label, const std::vector<uint32_t>& data, uint32_t offset) {
    printf("\n  === %s ===\n", label);
    printf("  %-4s %-24s %s\n", "Idx", "Register", "Value");
    printf("  %-4s %-24s %s\n", "---", "------------------------", "----------");
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        uint32_t val = data[offset + i];
        if (val != 0) {
            printf("  %-4u %-24s 0x%08x\n", i, REG_NAMES[i], val);
        }
    }
    uint32_t sentinel = data[offset + NUM_REGS];
    printf("  Sentinel: 0x%08x\n", sentinel);
}

TEST_F(MeshDeviceSingleCardFixture, NocRegDump) {
    IDevice* dev = devices_[0]->get_devices()[0];

    // Get legal L1 addresses
    uint32_t l1_base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Layout in L1:
    //   l1_base + 0x0000: source data (64 bytes of known pattern)
    //   l1_base + 0x1000: register dump region (3 dumps × 18 words = 216 bytes)
    uint32_t l1_src_addr = l1_base;
    uint32_t l1_dump_addr = l1_base + 0x1000;
    uint32_t transfer_size = 64;

    // Write known source data to L1
    CoreCoord core = {0, 0};
    std::vector<uint32_t> src_data(transfer_size / sizeof(uint32_t));
    for (uint32_t i = 0; i < src_data.size(); i++) {
        src_data[i] = 0xCAFE0000 + i;
    }
    detail::WriteToDeviceL1(dev, core, l1_src_addr, src_data);

    // Clear dump region
    std::vector<uint32_t> zeros(3 * (NUM_REGS + 1), 0);
    detail::WriteToDeviceL1(dev, core, l1_dump_addr, zeros);

    // Create program + kernel
    Program program = CreateProgram();

    KernelHandle kernel = CreateKernel(
        program,
        // Kernel path relative to TT_METAL_HOME
        "tests/tt_metal/tt_metal/test_kernels/dataflow/noc_write_dump_regs.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });

    // Runtime args: dram_addr_lo, dram_bank_id, l1_src_addr, transfer_size, l1_dump_addr
    // Use DRAM bank 0 at offset 0x100000 (1MB, safe area)
    uint32_t dram_offset = 0x100000;
    uint32_t dram_bank = 0;

    SetRuntimeArgs(program, kernel, core, {dram_offset, dram_bank, l1_src_addr, transfer_size, l1_dump_addr});

    // Launch
    printf("Launching kernel on core (0,0)...\n");
    detail::LaunchProgram(dev, program);
    printf("Kernel complete.\n");

    // Read back register dumps
    uint32_t total_dump_bytes = 3 * DUMP_ENTRY_SIZE;
    std::vector<uint32_t> dump_data;
    detail::ReadFromDeviceL1(dev, core, l1_dump_addr, total_dump_bytes, dump_data);

    // Print all three dumps
    uint32_t entry_words = NUM_REGS + 1;
    print_dump("BEFORE noc_async_write", dump_data, 0);
    print_dump("AFTER noc_async_write (before barrier)", dump_data, entry_words);
    print_dump("AFTER noc_async_write_barrier", dump_data, 2 * entry_words);

    // Print as JSON-like for easy copy
    printf("\n  === VALUES FOR L2CPU REPLAY ===\n");
    printf("  (use the AFTER write / before barrier values)\n");
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        uint32_t val = dump_data[entry_words + i];
        printf("  %s = 0x%08x\n", REG_NAMES[i], val);
    }

    // Verify sentinels
    EXPECT_EQ(dump_data[entry_words - 1], 0xBEF00001u);      // before
    EXPECT_EQ(dump_data[2 * entry_words - 1], 0xDEAD0002u);  // during
    EXPECT_EQ(dump_data[3 * entry_words - 1], 0xAF7E0003u);  // after
}
