#include <cstdint>
#include <optional>
#include "impl_device.hpp"
#include "device/tt_silicon_driver_common.hpp"
#include "common/core_coord.h"
#include "device/tt_device.h"
#include "device_data.hpp"

std::int32_t tt_SiliconDevice::get_static_tlb_index(CoreCoord target) {
    bool is_eth_location = std::find(std::cbegin(DEVICE_DATA.ETH_LOCATIONS), std::cend(DEVICE_DATA.ETH_LOCATIONS), target) != std::cend(DEVICE_DATA.ETH_LOCATIONS);
    bool is_tensix_location = std::find(std::cbegin(DEVICE_DATA.T6_X_LOCATIONS), std::cend(DEVICE_DATA.T6_X_LOCATIONS), target.x) != std::cend(DEVICE_DATA.T6_X_LOCATIONS) &&
                              std::find(std::cbegin(DEVICE_DATA.T6_Y_LOCATIONS), std::cend(DEVICE_DATA.T6_Y_LOCATIONS), target.y) != std::cend(DEVICE_DATA.T6_Y_LOCATIONS);
    // implementation migrated from wormhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/wormhole.py` from tensix repo (t6py-wormhole-bringup branch)
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

        const int num_locations_with_1MB_TLBs = 70;
        // First 70 get a pair of 1MB TLBs
        // Last 10 get a single 2MB TLB
        bool has_pair_of_1MB_TLB_entry = flat_index < num_locations_with_1MB_TLBs;
        int tlb_index = has_pair_of_1MB_TLB_entry ?
                            2 * flat_index + DEVICE_DATA.ETH_LOCATIONS.size() :
                            156 + flat_index - 70;
        return tlb_index;
    } else {
        return -1;
    }
}

std::optional<std::tuple<std::uint32_t, std::uint32_t>> tt_SiliconDevice::describe_tlb(std::int32_t tlb_index) {
    std::uint32_t TLB_COUNT_1M = 156;
    std::uint32_t TLB_COUNT_2M = 10;
    std::uint32_t TLB_COUNT_16M = 20;

    std::uint32_t TLB_BASE_1M = 0;
    std::uint32_t TLB_BASE_2M = TLB_COUNT_1M * (1 << 20);
    std::uint32_t TLB_BASE_16M = TLB_BASE_2M + TLB_COUNT_2M * (1 << 21);

    if (tlb_index < 0) { return std::nullopt; }

    if (tlb_index >= 0 && tlb_index < TLB_COUNT_1M) {
        std::uint32_t size = 1 << 20;
        std::uint32_t tlb_size;
        if (tlb_index >= 16 && tlb_index % 2 == 0) {
            // We have setup the even tlbs after 16 to take a 2M space of the bar.
            tlb_size = 1 << 21;
        } else {
            tlb_size = 1 << 20;
        }
        return std::tuple(TLB_BASE_1M + size * tlb_index, tlb_size);
    } else if(tlb_index >= 0 && tlb_index < TLB_COUNT_1M + TLB_COUNT_2M) {
        auto tlb_offset = tlb_index - TLB_COUNT_1M;
        auto size = 1 << 21;
        return std::tuple(TLB_BASE_2M + tlb_offset * size, size);
    } else if (tlb_index >= 0 and tlb_index < TLB_COUNT_1M + TLB_COUNT_2M + TLB_COUNT_16M) {
        auto tlb_offset = tlb_index - (TLB_COUNT_1M + TLB_COUNT_2M);
        auto size = 1 << 24;
        return std::tuple(TLB_BASE_16M + tlb_offset * size, size);
    }

    return std::nullopt;
}

std::optional<std::uint64_t> tt_SiliconDevice::get_tlb_data(std::uint32_t tlb_index, TLB_DATA data) {
    std::uint32_t TLB_COUNT_1M = 156;
    std::uint32_t TLB_COUNT_2M = 10;
    std::uint32_t TLB_COUNT_16M = 20;

    if (tlb_index < TLB_COUNT_1M) {
        return data.apply_offset(TLB_1M_OFFSET);
    } else if (tlb_index < TLB_COUNT_1M + TLB_COUNT_2M) {
        return data.apply_offset(TLB_2M_OFFSET);
    } else if (tlb_index < TLB_COUNT_1M + TLB_COUNT_2M + TLB_COUNT_16M) {
        return data.apply_offset(TLB_16M_OFFSET);
    }

    return std::nullopt;
}
