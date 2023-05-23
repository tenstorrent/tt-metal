#pragma once

#include "common/core_coord.h"
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>

#ifdef ARCH_GRAYSKULL
#error "CANNOT INCLUDE GRAYSKULL AND WORMHOLE."
#endif
#ifndef ARCH_WORMHOLE
#define ARCH_WORMHOLE
#endif
struct WORMHOLE_DEVICE_DATA {
    const std::vector<CoreCoord> DRAM_LOCATIONS = {
        {0, 0},  {5, 0}, {0, 1},{5, 1}, {5, 2}, {5, 3}, {5, 4}, {0, 5},  {5, 5}, {0, 6},  {5, 6}, {0, 7},  {5, 7}, {5, 8}, {5, 9}, {5, 10},{0, 11}, {5, 11}
    };
    const std::vector<CoreCoord> ARC_LOCATIONS = { {0, 2} };
    const std::vector<CoreCoord> PCI_LOCATIONS = { {0, 4} };
    const std::vector<CoreCoord> ETH_LOCATIONS = {
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {1, 6}, {2, 6}, {3, 6}, {4, 6}, {6, 6}, {7, 6}, {8, 6}, {9, 6}
    };
    const std::vector<uint32_t> T6_X_LOCATIONS = {1, 2, 3, 4, 6, 7, 8, 9};
    const std::vector<uint32_t> T6_Y_LOCATIONS = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};
    const std::vector<uint32_t> HARVESTING_NOC_LOCATIONS = {11, 1, 10, 2, 9, 3, 8, 4, 7, 5 };

    static constexpr uint32_t STATIC_TLB_SIZE = 1024*1024;

    static constexpr CoreCoord BROADCAST_LOCATION = {0, 0};
    static constexpr uint32_t BROADCAST_TLB_INDEX = 0;
    static constexpr uint32_t STATIC_TLB_CFG_ADDR = 0x1fc00000;

    static constexpr uint32_t TLB_COUNT_1M = 156;
    static constexpr uint32_t TLB_COUNT_2M = 10;
    static constexpr uint32_t TLB_COUNT_16M = 20;

    static constexpr uint32_t TLB_BASE_1M = 0;
    static constexpr uint32_t TLB_BASE_2M = TLB_COUNT_1M * (1 << 20);
    static constexpr uint32_t TLB_BASE_16M = TLB_BASE_2M + TLB_COUNT_2M * (1 << 21);

    static constexpr uint32_t TLB_BASE_INDEX_1M = 0;
    static constexpr uint32_t TLB_BASE_INDEX_2M = TLB_COUNT_1M;
    static constexpr uint32_t TLB_BASE_INDEX_16M = TLB_BASE_INDEX_2M + TLB_COUNT_2M;

    static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;

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
    static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = MEM_LARGE_READ_TLB + 1;
    static constexpr uint32_t INTERNAL_TLB_INDEX = DYNAMIC_TLB_BASE_INDEX + DYNAMIC_TLB_COUNT; // pcie_write_xy and similar
    static constexpr uint32_t DRAM_CHANNEL_0_X = 0;
    static constexpr uint32_t DRAM_CHANNEL_0_Y = 0;
    static constexpr uint32_t DRAM_CHANNEL_0_PEER2PEER_REGION_START = 0x30000000; // This is the last 256MB of DRAM

    static constexpr uint32_t GRID_SIZE_X = 10;
    static constexpr uint32_t GRID_SIZE_Y = 12;

    static constexpr uint32_t AXI_RESET_OFFSET = 0x1FF30000;
    static constexpr uint32_t ARC_RESET_SCRATCH_OFFSET = AXI_RESET_OFFSET + 0x0060;
    static constexpr uint32_t ARC_RESET_ARC_MISC_CNTL_OFFSET = AXI_RESET_OFFSET + 0x0100;

    static constexpr uint32_t ARC_CSM_OFFSET = 0x1FE80000;
    static constexpr uint32_t ARC_CSM_MAILBOX_OFFSET = ARC_CSM_OFFSET + 0x783C4;
    static constexpr uint32_t ARC_CSM_MAILBOX_SIZE_OFFSET = ARC_CSM_OFFSET + 0x784C4;

    static constexpr uint32_t ARC_CSM_SPI_TABLE_OFFSET = ARC_CSM_OFFSET + 0x78874;
    static constexpr uint32_t ARC_CSM_RowHarvesting_OFFSET = ARC_CSM_OFFSET + 0x78E7C;

    static constexpr uint32_t TENSIX_SOFT_RESET_ADDR = 0xFFB121B0;

    static constexpr uint32_t MSG_TYPE_SETUP_IATU_FOR_PEER_TO_PEER = 0x97;

    static constexpr uint32_t RISCV_RESET_DEASSERT[8] = { 0xffffffff, 0xffffffff, 0xffff, 0x0, 0x0, 0x0, 0x0, 0x0 };
};

static const auto DEVICE_DATA = WORMHOLE_DEVICE_DATA();
