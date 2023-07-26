#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {

    uint32_t sender_noc_x            = get_arg_val<uint32_t>(0);
    uint32_t sender_noc_y            = get_arg_val<uint32_t>(1);
    uint32_t num_tiles               = get_arg_val<uint32_t>(2);
    uint32_t sender_semaphore_addr   = get_arg_val<uint32_t>(3);
    uint32_t receiver_semaphore_addr = get_arg_val<uint32_t>(4);
    uint32_t num_repetitions         = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    constexpr uint32_t cb_id            = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    uint32_t block_size_bytes = get_tile_size(cb_id) * block_size_tiles;

    uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_semaphore_addr);

    for (uint32_t j = 0; j < num_repetitions; j++) {
        for (uint32_t i = 0; i<num_tiles ; i += block_size_tiles) {
            cb_reserve_back(cb_id, block_size_tiles);

            // Reset receiver's own semaphore value to INVALID
            noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);

            // Tell sender we're ready -- atomic increment sender's semaphore
            noc_semaphore_inc(sender_semaphore_noc_addr, 1);

            // Wait on receiver's own semaphore value to become VALID (set by sender after it sends the data)
            noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id, block_size_tiles);
        }
    }

}
