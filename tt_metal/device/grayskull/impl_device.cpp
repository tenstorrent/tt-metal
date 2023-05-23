#include "impl_device.hpp"
#include "common/core_coord.h"
#include "device/tt_device.h"
#include "device_data.hpp"

std::int32_t tt_SiliconDevice::get_static_tlb_index(CoreCoord target) {
  int flat_index = target.y * DEVICE_DATA.GRID_SIZE_X + target.x;
  if (flat_index == 0) {
      return -1;
  }
  return flat_index;
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
        return std::tuple(TLB_BASE_1M + size * tlb_index, size);
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
    if (tlb_index < DEVICE_DATA.TLB_COUNT_1M) {
        return data.apply_offset(TLB_1M_OFFSET);
    } else if (tlb_index < DEVICE_DATA.TLB_COUNT_1M + DEVICE_DATA.TLB_COUNT_2M) {
        return data.apply_offset(TLB_2M_OFFSET);
    } else if (tlb_index < DEVICE_DATA.TLB_COUNT_1M + DEVICE_DATA.TLB_COUNT_2M + DEVICE_DATA.TLB_COUNT_16M) {
        return data.apply_offset(TLB_16M_OFFSET);
    }

    return std::nullopt;
}
