#include <cstdint>

#include "llk_3c.h"

#include "tools/profiler/kernel_profiler.hpp"

//#include "debug_print.h"

namespace NAMESPACE {
void MAIN {
    binary_op_specific_init(ELTWISE_OP_CODE);
    binary_op_init_common(0, 1);

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    // PACK(kernel_profiler::mark_time(5));
    // PACK(kernel_profiler::mark_time(6));
    // PACK(kernel_profiler::mark_time(7));
    // PACK(kernel_profiler::mark_time(8));
    // PACK(kernel_profiler::mark_time(9));
    // PACK(kernel_profiler::mark_time(10));
    // PACK(kernel_profiler::mark_time(11));
    // PACK(kernel_profiler::mark_time(12));
    // PACK(kernel_profiler::mark_time(13));
    // PACK(kernel_profiler::mark_time(14));
    // PACK(kernel_profiler::mark_time(15));
    // PACK(kernel_profiler::mark_time(16));

    // UNPACK(kernel_profiler::mark_time(5));
    // UNPACK(kernel_profiler::mark_time(6));
    // UNPACK(kernel_profiler::mark_time(7));
    // UNPACK(kernel_profiler::mark_time(8));
    // UNPACK(kernel_profiler::mark_time(9));
    // UNPACK(kernel_profiler::mark_time(10));
    // UNPACK(kernel_profiler::mark_time(11));
    // UNPACK(kernel_profiler::mark_time(12));
    // UNPACK(kernel_profiler::mark_time(13));
    // UNPACK(kernel_profiler::mark_time(14));
    // UNPACK(kernel_profiler::mark_time(15));
    // UNPACK(kernel_profiler::mark_time(16));

    // MATH(kernel_profiler::mark_time(5));
    // MATH(kernel_profiler::mark_time(6));
    // MATH(kernel_profiler::mark_time(7));
    // MATH(kernel_profiler::mark_time(8));
    // MATH(kernel_profiler::mark_time(9));
    // MATH(kernel_profiler::mark_time(10));
    // MATH(kernel_profiler::mark_time(11));
    // MATH(kernel_profiler::mark_time(12));
    // MATH(kernel_profiler::mark_time(13));
    // MATH(kernel_profiler::mark_time(14));
    // MATH(kernel_profiler::mark_time(15));
    // MATH(kernel_profiler::mark_time(16));

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // PACK( kernel_profiler::mark_time(5) );
        // UNPACK( kernel_profiler::mark_time(5) );
        // MATH( kernel_profiler::mark_time(5) );

        cb_reserve_back(CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
        {

            // PACK( kernel_profiler::mark_time(6) );
            acquire_dst(DstMode::Half);
            // PACK( kernel_profiler::mark_time(7) );
            // MATH( kernel_profiler::mark_time(6) );

            cb_wait_front(CB::c_in0, 1);
            // UNPACK( kernel_profiler::mark_time(6) );
            cb_wait_front(CB::c_in1, 1);
            // UNPACK( kernel_profiler::mark_time(7) );

            ELTWISE_OP(CB::c_in0, CB::c_in1, 0, 0, 0);
            // MATH( kernel_profiler::mark_time(7) );

            pack_tile(0, CB::c_out0);
            // PACK( kernel_profiler::mark_time(8) );

            cb_pop_front(CB::c_in0, 1);
            // UNPACK( kernel_profiler::mark_time(8) );
            cb_pop_front(CB::c_in1, 1);
            // UNPACK( kernel_profiler::mark_time(9) );

            release_dst(DstMode::Half);
            // PACK( kernel_profiler::mark_time(9) );
            // MATH( kernel_profiler::mark_time(8) );
        }

        cb_push_back(CB::c_out0, per_core_block_size);
        // PACK( kernel_profiler::mark_time(10) );
        // UNPACK( kernel_profiler::mark_time(10) );
        // MATH( kernel_profiler::mark_time(9) );
    }
}
}
