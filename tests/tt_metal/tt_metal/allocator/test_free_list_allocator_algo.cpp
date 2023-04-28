#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_free_list() {
    bool pass = true;

    constexpr static uint32_t max_size_bytes = 1024;
    constexpr static uint32_t min_allocation_size_bytes = 32;
    constexpr static uint32_t alignment = 32;

    auto free_list_allocator = allocator::FreeList(
        max_size_bytes,
        min_allocation_size_bytes,
        alignment,
        allocator::FreeList::SearchPolicy::FIRST
    );

    bool allocate_bottom_up = true;
    auto addr_0 = free_list_allocator.allocate(32, true);
    pass &= addr_0 == 0;

    auto addr_1 = free_list_allocator.allocate_at_address(64, 32, true);
    pass &= addr_1 == 64;

    auto addr_2 = free_list_allocator.allocate(48, true);
    pass &= addr_2 == 96;

    auto addr_3 = free_list_allocator.allocate(16, true);
    pass &= addr_3 == 32;

    auto addr_4 = free_list_allocator.allocate_at_address(512, 128, true);
    pass &= addr_4 == 512;

    free_list_allocator.deallocate(96); // coalesce with next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_5 = free_list_allocator.allocate_at_address(128, 64, true);
    pass &= addr_5 == 128;

    auto addr_6 = free_list_allocator.allocate(32, true);
    pass &= addr_6 == 96;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(64); // coalesce with prev block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_7 = free_list_allocator.allocate(64, true);
    pass &= addr_7 == 32;

    auto addr_8 = free_list_allocator.allocate(316, true);
    pass &= addr_8 == 192;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(128);
    free_list_allocator.deallocate(96); // coalesce with prev and next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_9 = free_list_allocator.allocate_at_address(64, 96, true);
    pass &= addr_9 == 64;

    free_list_allocator.deallocate(192);
    auto addr_10 = free_list_allocator.allocate_at_address(256, 128, true);
    pass &= addr_10 == 256;

    free_list_allocator.deallocate(0);
    auto addr_11 = free_list_allocator.allocate(28, true);
    pass &= addr_11 == 0;

    auto addr_12 = free_list_allocator.allocate(64, false);
    pass &= addr_12 == 1024;

    auto addr_13 = free_list_allocator.allocate(128, false);
    pass &= addr_13 == 960;

    auto addr_14 = free_list_allocator.allocate_at_address(736, 96, false);
    pass &= addr_14 == 736;

    auto addr_15 = free_list_allocator.allocate(96, false);
    pass &= addr_15 == 832;

    auto addr_16 = free_list_allocator.allocate(96, false);
    pass &= addr_16 == 480;

    // coalesce two blocks growing in opposite directions
    free_list_allocator.deallocate(480);
    free_list_allocator.deallocate(512);
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_17 = free_list_allocator.allocate(224, true);
    pass &= addr_17 == 384;

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        pass &= test_free_list();

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
