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

#define ALIGNMENT 64  // TODO

static inline struct l1_allocator new_tensix_allocator() {
    using namespace tt::tt_metal;

    uint32_t start = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    return (struct l1_allocator){
        .start = start,
        .end = end,
    };
}

static inline struct l1_allocator new_erisc_allocator() {
    using namespace tt::tt_metal;

    uint32_t start =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    uint32_t end = start + MetalContext::instance().hal().get_dev_size(
                               HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    return (struct l1_allocator){
        .start = start,
        .end = end,
    };
}

static inline uint32_t l1_alloc(struct l1_allocator* alloc, uint32_t size, uint32_t alignment = ALIGNMENT) {
    uint32_t start = ROUND_UP(alloc->start, alignment);

    TT_FATAL(start + size <= alloc->end, "Couldn't allocate in L1");

    alloc->start = start + size;

    return start;
}

uint64_t read_l1_u64(tt::tt_metal::IDevice* const device, const CoreCoord& core, uint64_t l1_addr) {
    auto delta_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core<uint32_t>(
        device->id(), device->ethernet_core_from_logical_core(core), l1_addr, 2 * sizeof(uint32_t));

    return (uint64_t)delta_vec[0] | ((uint64_t)delta_vec[1] << 32);
}

extern std::atomic_bool g_stop_requested;
extern std::atomic_bool g_stop_message_printed;

void handle_sigint(int);

class SignalGuard {
private:
    sighandler_t prev;
    int signum;

public:
    SignalGuard(int sig, sighandler_t handler) {
        signum = sig;
        prev = signal(sig, handler);
    }
    ~SignalGuard() { signal(signum, prev); }
};

#endif /* _DEPLOYMENT_COMMON_H */
