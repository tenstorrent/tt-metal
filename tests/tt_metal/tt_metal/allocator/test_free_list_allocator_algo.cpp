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

    auto addr_0 = free_list_allocator.allocate(32);
    pass &= addr_0 == 0;

    auto addr_1 = free_list_allocator.allocate(64, 32);
    pass &= addr_1 == 64;

    auto addr_2 = free_list_allocator.allocate(48);
    pass &= addr_2 == 96;

    auto addr_3 = free_list_allocator.allocate(16);
    pass &= addr_3 == 32;

    auto addr_4 = free_list_allocator.allocate(512, 128);
    pass &= addr_4 == 512;

    free_list_allocator.deallocate(96); // coalesce with next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_5 = free_list_allocator.allocate(128, 64);
    pass &= addr_5 == 128;

    auto addr_6 = free_list_allocator.allocate(32);
    pass &= addr_6 == 96;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(64); // coalesce with prev block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_7 = free_list_allocator.allocate(64);
    pass &= addr_7 == 32;

    auto addr_8 = free_list_allocator.allocate(316);
    pass &= addr_8 == 192;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(128);
    free_list_allocator.deallocate(96); // coalesce with prev and next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_9 = free_list_allocator.allocate(64, 96);
    pass &= addr_9 == 64;

    free_list_allocator.deallocate(192);
    auto addr_10 = free_list_allocator.allocate(256, 128);
    pass &= addr_10 == 256;

    free_list_allocator.deallocate(0);
    auto addr_11 = free_list_allocator.allocate(28);
    pass &= addr_11 == 0;

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
