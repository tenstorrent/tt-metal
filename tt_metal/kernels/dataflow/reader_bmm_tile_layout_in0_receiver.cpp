#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 mcast args
    uint32_t in0_mcast_sender_noc_y             = get_arg_val<uint32_t>(0);

    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles                = get_compile_time_arg_val(0);
    // in0/in1 common args
    constexpr uint32_t num_blocks                         = get_compile_time_arg_val(1);
    // in0 mcast args
    constexpr uint32_t in0_mcast_sender_noc_x             = get_compile_time_arg_val(2);
    constexpr uint32_t in0_mcast_sender_semaphore_addr    = get_compile_time_arg_val(3);
    constexpr uint32_t in0_mcast_receiver_semaphore_addr  = get_compile_time_arg_val(4);
    // batch args
    constexpr uint32_t batch                              = get_compile_time_arg_val(5);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    volatile uint32_t* in0_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in0_mcast_receiver_semaphore_addr);

    for (uint32_t b = 0; b < batch; b++) {
        for(uint32_t block = 0; block < num_blocks; block++) {
            // Operand 0
            dataflow::cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            dataflow_internal::noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            uint64_t in0_mcast_sender_semaphore_noc_addr = dataflow::get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);
            dataflow_internal::noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            dataflow_internal::noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

            dataflow::cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
    }
}
