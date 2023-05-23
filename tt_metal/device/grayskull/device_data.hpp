#pragma once

#include "common/core_coord.h"
#include <cstdint>
#include <vector>

#ifdef ARCH_WORMHOLE
#error "CANNOT INCLUDE GRAYSKULL AND WORMHOLE."
#endif
#ifndef ARCH_GRAYSKULL
#define ARCH_GRAYSKULL
#endif
struct GRAYSKULL_DEVICE_DATA {
    const std::vector<CoreCoord> DRAM_LOCATIONS = { {1, 6}, {4, 6}, {7, 6}, {10, 6}, {1, 0}, {4, 0}, {7, 0}, {10, 0} };
    const std::vector<CoreCoord> ARC_LOCATIONS = { {0, 2} };
    const std::vector<CoreCoord> PCI_LOCATIONS = { {0, 4} };
    const std::vector<uint32_t> T6_X_LOCATIONS = {12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6 };
    const std::vector<uint32_t> T6_Y_LOCATIONS = {11, 1, 10, 2, 9, 3, 8, 4, 7, 5 };
    const std::vector<uint32_t> HARVESTING_NOC_LOCATIONS = {5, 7, 4, 8, 3, 9, 2, 10, 1, 11 };

    static constexpr uint32_t STATIC_TLB_SIZE = 1024*1024;

    static constexpr CoreCoord BROADCAST_LOCATION = {0, 0};
    static constexpr uint32_t BROADCAST_TLB_INDEX = 0;

    static constexpr uint32_t TLB_COUNT_1M = 156;
    static constexpr uint32_t TLB_COUNT_2M = 10;
    static constexpr uint32_t TLB_COUNT_16M = 20;

    static constexpr uint32_t TLB_BASE_1M = 0;
    static constexpr uint32_t TLB_BASE_2M = TLB_COUNT_1M * (1 << 20);
    static constexpr uint32_t TLB_BASE_16M = TLB_BASE_2M + TLB_COUNT_2M * (1 << 21);

    static constexpr uint32_t TLB_BASE_INDEX_1M = 0;
    static constexpr uint32_t TLB_BASE_INDEX_2M = TLB_COUNT_1M;
    static constexpr uint32_t TLB_BASE_INDEX_16M = TLB_BASE_INDEX_2M + TLB_COUNT_2M;

    static constexpr uint32_t STATIC_TLB_CFG_ADDR = 0x1fc00000;

    static constexpr uint32_t DYNAMIC_TLB_16M_SIZE = 16 * 1024*1024;
    static constexpr uint32_t DYNAMIC_TLB_16M_CFG_ADDR = STATIC_TLB_CFG_ADDR + (TLB_BASE_INDEX_16M * 8);
    static constexpr uint32_t DYNAMIC_TLB_16M_BASE = TLB_BASE_16M;

    static constexpr uint32_t DYNAMIC_TLB_2M_SIZE = 2 * 1024*1024;
    static constexpr uint32_t DYNAMIC_TLB_2M_CFG_ADDR = STATIC_TLB_CFG_ADDR + (TLB_BASE_INDEX_2M * 8);
    static constexpr uint32_t DYNAMIC_TLB_2M_BASE = TLB_BASE_2M;

    static constexpr uint32_t DYNAMIC_TLB_1M_SIZE = 1 * 1024*1024;
    static constexpr uint32_t DYNAMIC_TLB_1M_CFG_ADDR = STATIC_TLB_CFG_ADDR + (TLB_BASE_INDEX_1M * 8);
    static constexpr uint32_t DYNAMIC_TLB_1M_BASE = TLB_BASE_1M;

    // MEM_*_TLB are for dynamic read/writes to memory, either 16MB (large read/writes) or 2MB (polling). REG_TLB for dynamic writes
    // to registers.   They are aligned with the kernel driver's WC/UC split.  But kernel driver uses different TLB's for these.
    static constexpr unsigned int REG_TLB                   = TLB_BASE_INDEX_16M + 18;
    static constexpr unsigned int MEM_LARGE_WRITE_TLB       = TLB_BASE_INDEX_16M + 17;
    static constexpr unsigned int MEM_LARGE_READ_TLB        = TLB_BASE_INDEX_16M + 0;
    static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB  = TLB_BASE_INDEX_2M + 1;

    static constexpr uint32_t DRAM_CHANNEL_0_X = 1;
    static constexpr uint32_t DRAM_CHANNEL_0_Y = 0;
    static constexpr uint32_t DRAM_CHANNEL_0_PEER2PEER_REGION_START = 0x30000000; // This is the last 256MB of DRAM

    static constexpr uint32_t GRID_SIZE_X = 13;
    static constexpr uint32_t GRID_SIZE_Y = 12;

    static constexpr uint32_t AXI_RESET_OFFSET = 0x1FF30000;
    static constexpr uint32_t ARC_RESET_SCRATCH_OFFSET = AXI_RESET_OFFSET + 0x0060;
    static constexpr uint32_t ARC_RESET_ARC_MISC_CNTL_OFFSET = AXI_RESET_OFFSET + 0x0100;

    static constexpr uint32_t ARC_CSM_OFFSET = 0x1FE80000;
    static constexpr uint32_t ARC_CSM_MAILBOX_OFFSET = ARC_CSM_OFFSET + 0x783BC;
    static constexpr uint32_t ARC_CSM_MAILBOX_SIZE_OFFSET = ARC_CSM_OFFSET + 0x784BC;

    static constexpr uint32_t ARC_CSM_SPI_TABLE_OFFSET = ARC_CSM_OFFSET + 0x78874;
    static constexpr uint32_t ARC_CSM_RowHarvesting_OFFSET = ARC_CSM_OFFSET + 0x7836C;

    static constexpr uint32_t MSG_TYPE_SETUP_IATU_FOR_PEER_TO_PEER = 0x97;

    static constexpr uint32_t TENSIX_SOFT_RESET_ADDR = 0xFFB121B0;

    static constexpr uint32_t RISCV_RESET_DEASSERT[] = { 0xff3ff3ff, 0xf3ff3ff3, 0x3ff3ff3f, 0xff3ff3ff, 0x3ff3, 0, 0, 0 };
};

static const auto DEVICE_DATA = GRAYSKULL_DEVICE_DATA();
