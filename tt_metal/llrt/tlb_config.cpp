// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tlb_config.hpp"
#include "device_data.hpp"

#include "third_party/umd/device/blackhole_implementation.h"
#include "third_party/umd/device/grayskull_implementation.h"
#include "third_party/umd/device/wormhole_implementation.h"

namespace ll_api {

namespace grayskull {

static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr unsigned int DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;
static constexpr uint32_t DYNAMIC_TLB_2M_SIZE = 0;
static constexpr uint32_t DYNAMIC_TLB_16M_SIZE = tt::umd::grayskull::DYNAMIC_TLB_16M_SIZE;

int32_t get_static_tlb_index(CoreCoord target) {
    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }
    int flat_index = target.y * DEVICE_DATA.GRID_SIZE_X + target.x;
    if (flat_index == 0) {
        return -1;
    }
    return flat_index;
}

}  // namespace grayskull

namespace wormhole {

static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;
static constexpr uint32_t DYNAMIC_TLB_2M_SIZE = 0;
static constexpr uint32_t DYNAMIC_TLB_16M_SIZE = tt::umd::wormhole::DYNAMIC_TLB_16M_SIZE;

int32_t get_static_tlb_index(CoreCoord target) {
    bool is_eth_location =
        std::find(std::cbegin(DEVICE_DATA.ETH_LOCATIONS), std::cend(DEVICE_DATA.ETH_LOCATIONS), target) !=
        std::cend(DEVICE_DATA.ETH_LOCATIONS);
    bool is_tensix_location =
        std::find(std::cbegin(DEVICE_DATA.T6_X_LOCATIONS), std::cend(DEVICE_DATA.T6_X_LOCATIONS), target.x) !=
            std::cend(DEVICE_DATA.T6_X_LOCATIONS) &&
        std::find(std::cbegin(DEVICE_DATA.T6_Y_LOCATIONS), std::cend(DEVICE_DATA.T6_Y_LOCATIONS), target.y) !=
            std::cend(DEVICE_DATA.T6_Y_LOCATIONS);
    // implementation migrated from wormhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/wormhole.py` from tensix
    // repo (t6py-wormhole-bringup branch)

    // Special handling for DRAM TLBs : return a 2MB TLB pointing to the start of the Epoch Cmd Queue Table
    // The default 1MB TLB is not used for DRAM cores
    // auto DRAM_TLB_IDX = std::find(DEVICE_DATA.DRAM_LOCATIONS.begin(), DEVICE_DATA.DRAM_LOCATIONS.end(), target);
    // if (DRAM_TLB_IDX != DEVICE_DATA.DRAM_LOCATIONS.end()) {
    //     return EPOCH_CMD_QUEUE_TLBS.at(DRAM_TLB_IDX - DEVICE_DATA.DRAM_LOCATIONS.begin());
    // }

    if (is_eth_location) {
        if (target.y == 6) {
            target.y = 1;
        }

        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        int flat_index = target.y * 8 + target.x;
        int tlb_index = flat_index;
        return tlb_index;

    } else if (is_tensix_location) {
        if (target.x >= 5) {
            target.x -= 1;
        }
        target.x -= 1;

        if (target.y >= 6) {
            target.y -= 1;
        }
        target.y -= 1;

        int flat_index = target.y * 8 + target.x;

        // All 80 get single 1MB TLB.
        int tlb_index = DEVICE_DATA.ETH_LOCATIONS.size() + flat_index;

        return tlb_index;
    } else {
        return -1;
    }
}

}  // namespace wormhole

namespace blackhole {

static constexpr uint32_t DYNAMIC_TLB_COUNT = 16;
static constexpr unsigned int MEM_SMALL_READ_WRITE_TLB = DEVICE_DATA.TLB_BASE_INDEX_2M + 1;
static constexpr uint32_t DYNAMIC_TLB_BASE_INDEX = DEVICE_DATA.MEM_LARGE_READ_TLB + 1;
static constexpr uint32_t DYNAMIC_TLB_2M_SIZE = tt::umd::blackhole::DYNAMIC_TLB_2M_SIZE;
static constexpr uint32_t DYNAMIC_TLB_16M_SIZE = 0;

int32_t get_static_tlb_index(CoreCoord target) {
    bool is_eth_location =
        std::find(
            std::cbegin(tt::umd::blackhole::ETH_LOCATIONS), std::cend(tt::umd::blackhole::ETH_LOCATIONS), target) !=
        std::cend(tt::umd::blackhole::ETH_LOCATIONS);
    bool is_tensix_location =
        std::find(
            std::cbegin(tt::umd::blackhole::T6_X_LOCATIONS), std::cend(tt::umd::blackhole::T6_X_LOCATIONS), target.x) !=
            std::cend(tt::umd::blackhole::T6_X_LOCATIONS) &&
        std::find(
            std::cbegin(tt::umd::blackhole::T6_Y_LOCATIONS), std::cend(tt::umd::blackhole::T6_Y_LOCATIONS), target.y) !=
            std::cend(tt::umd::blackhole::T6_Y_LOCATIONS);
    // implementation migrated from blackhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/blackhole.py` from tensix
    // repo (t6py-blackhole-bringup branch)

    // BH worker cores are starting from y=2
    target.y--;
    if (is_eth_location) {
        // TODO: Uplift this
        return -1;
    } else if (is_tensix_location) {
        if (target.x >= 8) {
            target.x -= 2;
        }

        int flat_index = target.y * 14 + target.x;
        int tlb_index = tt::umd::blackhole::ETH_LOCATIONS.size() + flat_index;
        return tlb_index;
    } else {
        return -1;
    }
}

}  // namespace blackhole

void configure_static_tlbs(tt::ARCH arch, chip_id_t mmio_device_id, const metal_SocDescriptor &sdesc, tt_device &device_driver) {
    using get_static_tlb_index_ptr = std::int32_t (*)(tt_xy_pair);
    get_static_tlb_index_ptr get_static_tlb_index;
    uint32_t DYNAMIC_TLB_BASE_INDEX, DYNAMIC_TLB_COUNT, DYNAMIC_TLB_16M_SIZE, DYNAMIC_TLB_2M_SIZE;

    switch (arch) {
        case tt::ARCH::GRAYSKULL:
            get_static_tlb_index = grayskull::get_static_tlb_index;
            DYNAMIC_TLB_BASE_INDEX = grayskull::DYNAMIC_TLB_BASE_INDEX;
            DYNAMIC_TLB_COUNT = grayskull::DYNAMIC_TLB_COUNT;
            DYNAMIC_TLB_16M_SIZE = grayskull::DYNAMIC_TLB_16M_SIZE;
            DYNAMIC_TLB_2M_SIZE = grayskull::DYNAMIC_TLB_2M_SIZE;
            break;
        case tt::ARCH::WORMHOLE:
        case tt::ARCH::WORMHOLE_B0:
            get_static_tlb_index = wormhole::get_static_tlb_index;
            DYNAMIC_TLB_BASE_INDEX = wormhole::DYNAMIC_TLB_BASE_INDEX;
            DYNAMIC_TLB_COUNT = wormhole::DYNAMIC_TLB_COUNT;
            DYNAMIC_TLB_16M_SIZE = wormhole::DYNAMIC_TLB_16M_SIZE;
            DYNAMIC_TLB_2M_SIZE = wormhole::DYNAMIC_TLB_2M_SIZE;
            break;
        case tt::ARCH::BLACKHOLE:
            get_static_tlb_index = blackhole::get_static_tlb_index;
            DYNAMIC_TLB_BASE_INDEX = blackhole::DYNAMIC_TLB_BASE_INDEX;
            DYNAMIC_TLB_COUNT = blackhole::DYNAMIC_TLB_COUNT;
            DYNAMIC_TLB_2M_SIZE = blackhole::DYNAMIC_TLB_2M_SIZE;
            DYNAMIC_TLB_16M_SIZE = blackhole::DYNAMIC_TLB_16M_SIZE;
            break;
        default: TT_THROW("Configuring static TLBs is not supported for {}", tt::get_string(arch));
    }

    auto statically_mapped_cores = sdesc.workers;
    // TODO (#9933): Remove workaround for BH
    if (arch != tt::ARCH::BLACKHOLE) {
        statically_mapped_cores.insert(
            statically_mapped_cores.end(), sdesc.ethernet_cores.begin(), sdesc.ethernet_cores.end());
    }
    std::int32_t address = 0;

    // Setup static TLBs for all worker cores
    for (auto &core : statically_mapped_cores) {
        auto tlb_index = get_static_tlb_index(core);
        // TODO
        // Note: see issue #10107
        // Strict is less performant than Posted, however, metal doesn't presently
        // use this on a perf path and the launch_msg "kernel config" needs to
        // arrive prior to the "go" message during device init and slow dispatch
        // Revisit this when we have a more flexible UMD api
        device_driver.configure_tlb(mmio_device_id, core, tlb_index, address, TLB_DATA::Strict);
    }

    // TODO (#9932): Remove workaround for BH
    if (arch != tt::ARCH::BLACKHOLE) {
        // Setup static TLBs for MMIO mapped data space
        uint64_t peer_dram_offset = DEVICE_DATA.DRAM_CHANNEL_0_PEER2PEER_REGION_START;
        for (uint32_t tlb_id = DYNAMIC_TLB_BASE_INDEX; tlb_id < DYNAMIC_TLB_BASE_INDEX + DYNAMIC_TLB_COUNT; tlb_id++) {
            device_driver.configure_tlb(
                mmio_device_id, CoreCoord(DEVICE_DATA.DRAM_CHANNEL_0_X, DEVICE_DATA.DRAM_CHANNEL_0_Y), tlb_id, peer_dram_offset);
            // Align address space of 16MB TLB to 16MB boundary
            peer_dram_offset += DYNAMIC_TLB_16M_SIZE;
        }
    }
    device_driver.setup_core_to_tlb_map([get_static_tlb_index](CoreCoord core) { return get_static_tlb_index(core); });
}

std::unordered_map<std::string, std::int32_t> get_dynamic_tlb_config() {
    std::unordered_map<std::string, std::int32_t> dynamic_tlb_config;
    dynamic_tlb_config["REG_TLB"] = DEVICE_DATA.REG_TLB;
    return dynamic_tlb_config;
}

}  // namespace ll_api
