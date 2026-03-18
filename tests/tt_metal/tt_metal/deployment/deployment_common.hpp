#ifndef _DEPLOYMENT_COMMON_H
#define _DEPLOYMENT_COMMON_H

#include "tt_metal/api/tt-metalium/hal.hpp"
#include "command_queue_fixture.hpp"

struct l1_allocator {
    uint32_t start;
    uint32_t end;
};

#define ROUND_UP(x, a) ((((x) + (a) - 1) / (a)) * (a))
#define ROUND_DOWN(x, a) (((x) / (a)) * (a))

#define ALIGNMENT (tt::tt_metal::hal::get_l1_alignment())

static inline struct l1_allocator new_tensix_allocator() {
    using namespace tt::tt_metal;

    uint32_t start = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    return (struct l1_allocator){
        .start = ROUND_UP(start, ALIGNMENT),
        .end = ROUND_DOWN(end, ALIGNMENT),
    };
}

static inline struct l1_allocator new_erisc_allocator() {
    using namespace tt::tt_metal;

    uint32_t start =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    return (struct l1_allocator){
        .start = ROUND_UP(start, ALIGNMENT),
        .end = ROUND_DOWN(end, ALIGNMENT),
    };
}

static inline uint32_t l1_alloc(struct l1_allocator* alloc, uint32_t size) {
    size = ROUND_UP(size, ALIGNMENT);

    TT_FATAL(alloc->start + size <= alloc->end, "Couldn't allocate in L1");

    uint32_t ret = alloc->start;
    alloc->start += size;

    return ret;
}

uint64_t read_l1_u64(tt::tt_metal::IDevice* const device, const CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->ethernet_core_from_logical_core(core), l1_addr, 2 * sizeof(uint32_t));

    return (uint64_t)delta_vec[0] | ((uint64_t)delta_vec[1] << 32);
}

#endif /* _DEPLOYMENT_COMMON_H */
