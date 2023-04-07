#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/memory_manager/memory_manager.hpp"
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
    auto memory_manager = MemoryManager(max_size_bytes);

    auto addr_0 = memory_manager.allocate(32);
    pass &= addr_0 == 0;

    auto addr_1 = memory_manager.reserve(64, 32);
    pass &= addr_1 == 64;

    auto addr_2 = memory_manager.allocate(48);
    pass &= addr_2 == 96;

    auto addr_3 = memory_manager.allocate(16);
    pass &= addr_3 == 32;

    auto addr_4 = memory_manager.reserve(512, 128);
    pass &= addr_4 == 512;

    memory_manager.deallocate(96); // coalesce with next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_5 = memory_manager.reserve(128, 64);
    pass &= addr_5 == 128;

    auto addr_6 = memory_manager.allocate(32);
    pass &= addr_6 == 96;

    memory_manager.deallocate(32);
    memory_manager.deallocate(64); // coalesce with prev block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_7 = memory_manager.allocate(64);
    pass &= addr_7 == 32;

    auto addr_8 = memory_manager.allocate(316);
    pass &= addr_8 == 192;

    memory_manager.deallocate(32);
    memory_manager.deallocate(128);
    memory_manager.deallocate(96); // coalesce with prev and next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    auto addr_9 = memory_manager.reserve(64, 96);
    pass &= addr_9 == 64;

    memory_manager.deallocate(192);
    auto addr_10 = memory_manager.reserve(256, 128);
    pass &= addr_10 == 256;

    memory_manager.deallocate(0);
    auto addr_11 = memory_manager.allocate(28);
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
