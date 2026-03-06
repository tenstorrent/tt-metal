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

    uint32_t start =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);

    uint32_t end =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);

    return (struct l1_allocator){
        .start = ROUND_UP(start, ALIGNMENT),
        .end = ROUND_DOWN(end, ALIGNMENT),
    };
}

static inline struct l1_allocator new_erisc_allocator() {
    using namespace tt::tt_metal;

    uint32_t start =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    uint32_t end =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    return (struct l1_allocator){
        .start = ROUND_UP(start, ALIGNMENT),
        .end = ROUND_DOWN(end, ALIGNMENT),
    };
}

static inline uint32_t l1_alloc(struct l1_allocator& alloc, uint32_t size) {
    size = ROUND_UP(size, ALIGNMENT);

    TT_FATAL(alloc.start + size <= alloc.end, "Couldn't allocate in L1");

    uint32_t ret = alloc.start;
    alloc.start += size;

    return ret;
}

#endif /* _DEPLOYMENT_COMMON_H */
