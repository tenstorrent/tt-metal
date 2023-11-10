// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
        /*offset*/0,
        min_allocation_size_bytes,
        alignment,
        allocator::FreeList::SearchPolicy::FIRST
    );

    bool allocate_bottom_up = true;
    auto addr_0 = free_list_allocator.allocate(32, true);
    TT_FATAL(addr_0.has_value());
    pass &= addr_0 == 0;

    auto addr_1 = free_list_allocator.allocate_at_address(64, 32);
    TT_FATAL(addr_1.has_value());
    pass &= addr_1 == 64;

    auto addr_2 = free_list_allocator.allocate(48, true);
    TT_FATAL(addr_2.has_value());
    pass &= addr_2 == 96;

    auto addr_3 = free_list_allocator.allocate(16, true);
    TT_FATAL(addr_3.has_value());
    pass &= addr_3 == 32;

    auto addr_4 = free_list_allocator.allocate_at_address(512, 128);
    TT_FATAL(addr_4.has_value());
    pass &= addr_4 == 512;

    free_list_allocator.deallocate(96); // coalesce with next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_5 = free_list_allocator.allocate_at_address(128, 64);
    TT_FATAL(addr_5.has_value());
    pass &= addr_5 == 128;

    auto addr_6 = free_list_allocator.allocate(32, true);
    TT_FATAL(addr_6.has_value());
    pass &= addr_6 == 96;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(64); // coalesce with prev block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_7 = free_list_allocator.allocate(64, true);
    TT_FATAL(addr_7.has_value());
    pass &= addr_7 == 32;

    auto addr_8 = free_list_allocator.allocate(316, true);
    TT_FATAL(addr_8.has_value());
    pass &= addr_8 == 192;

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(128);
    free_list_allocator.deallocate(96); // coalesce with prev and next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_9 = free_list_allocator.allocate_at_address(64, 96);
    TT_FATAL(addr_9.has_value());
    pass &= addr_9 == 64;

    free_list_allocator.deallocate(192);
    auto addr_10 = free_list_allocator.allocate_at_address(256, 128);
    TT_FATAL(addr_10.has_value());
    pass &= addr_10 == 256;

    free_list_allocator.deallocate(0);
    auto addr_11 = free_list_allocator.allocate(28, true);
    TT_FATAL(addr_11.has_value());
    pass &= addr_11 == 0;

    auto addr_12 = free_list_allocator.allocate(64, false);
    pass &= addr_12 == 960;

    auto addr_13 = free_list_allocator.allocate(128, false);
    pass &= addr_13 == 832;

    auto addr_14 = free_list_allocator.allocate_at_address(736, 96);
    pass &= addr_14 == 736;

    auto addr_15 = free_list_allocator.allocate(96, false);
    pass &= addr_15 == 640;

    auto addr_16 = free_list_allocator.allocate(96, false);
    pass &= addr_16 == 416;

    free_list_allocator.deallocate(416);
    free_list_allocator.deallocate(512);
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_17 = free_list_allocator.allocate(224, true);
    pass &= addr_17 == 384;

    // Allocate entire region
    free_list_allocator.clear();
    auto addr_18 = free_list_allocator.allocate(max_size_bytes, true);
    TT_FATAL(addr_18.has_value());
    pass &= addr_18 == 0;

    free_list_allocator.deallocate(0);

    auto addr_19 = free_list_allocator.allocate(64, true);
    TT_FATAL(addr_19.has_value());
    pass &= addr_19 == 0;

    auto addr_20 = free_list_allocator.allocate(max_size_bytes - 64, true);
    TT_FATAL(addr_20.has_value());
    pass &= addr_20 == 64;

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

    TT_FATAL(pass);

    return 0;
}
