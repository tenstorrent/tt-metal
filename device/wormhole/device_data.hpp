#pragma once

#include "common/tt_xy_pair.h"
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>

#ifdef ARCH_GRAYSKULL
#error "CANNOT INCLUDE GRAYSKULL AND WORMHOLE."
#endif
#define ARCH_WORMHOLE

struct WORMHOLE_DEVICE_DATA {
    const std::vector<tt_xy_pair> DRAM_LOCATIONS = {
        {0, 0},  {5, 0}, {0, 1},{5, 1}, {5, 2}, {5, 3}, {5, 4}, {0, 5},  {5, 5}, {0, 6},  {5, 6}, {0, 7},  {5, 7}, {5, 8}, {5, 9}, {5, 10},{0, 11}, {5, 11}
    };
    const std::vector<tt_xy_pair> ARC_LOCATIONS = { {0, 2} };
    const std::vector<tt_xy_pair> PCI_LOCATIONS = { {0, 4} };
    const std::vector<tt_xy_pair> ETH_LOCATIONS = {
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {1, 6}, {2, 6}, {3, 6}, {4, 6}, {6, 6}, {7, 6}, {8, 6}, {9, 6}
    };
    const std::vector<uint32_t> T6_X_LOCATIONS = {1, 2, 3, 4, 6, 7, 8, 9};
    const std::vector<uint32_t> T6_Y_LOCATIONS = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};


    // TODO: verify the BELOW numbers are coorrect for wormhole - the fields were copied from grayskull
    static constexpr uint32_t STATIC_TLB_SIZE = 1024*1024;

    static constexpr tt_xy_pair BROADCAST_LOCATION = {0, 0};

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

    static constexpr uint32_t BROADCAST_TLB_INDEX = TLB_BASE_INDEX_16M; // First 16M TLB
    static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = BROADCAST_TLB_INDEX + 1;
    static constexpr uint32_t DYNAMIC_TLB_COUNT = 20 - 3; // 20 - broadcast TLB - internal TLB - kernel TLB
    static constexpr uint32_t INTERNAL_TLB_INDEX = DYNAMIC_TLB_BASE_INDEX + DYNAMIC_TLB_COUNT; // pcie_write_xy and similar
    static constexpr uint32_t DYNAMIC_TLB_SIZE = 16 * 1024*1024;
    static constexpr uint32_t DYNAMIC_TLB_CFG_ADDR = STATIC_TLB_CFG_ADDR + (BROADCAST_TLB_INDEX * 8);
    static constexpr uint32_t DYNAMIC_TLB_BASE = 0xB000000;

    // MEM_TLB is a 16MB TLB reserved for dynamic writes to memory. REG_TLB ... for dynamic writes to registers.
    // They are aligned with the kernel driver's WC/UC split.
    static constexpr unsigned int MEM_TLB = 17;
    static constexpr unsigned int REG_TLB = 18;

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
    static constexpr uint32_t ARC_CSM_RowHarvesting_OFFSET = ARC_CSM_SPI_TABLE_OFFSET + 376;
    static constexpr uint32_t ARC_CSM_RowHarvesting_en_OFFSET = ARC_CSM_SPI_TABLE_OFFSET + 380;

    static constexpr uint32_t TENSIX_SOFT_RESET_ADDR = 0xFFB121B0;

    static constexpr uint32_t MSG_TYPE_SETUP_IATU_FOR_PEER_TO_PEER = 0x97;

    static constexpr uint32_t RISCV_RESET_DEASSERT[8] = { 0xffffffff, 0xffffffff, 0xffff, 0x0, 0x0, 0x0, 0x0, 0x0 };
};

static const auto DEVICE_DATA = WORMHOLE_DEVICE_DATA();
