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

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {
        PACK( kernel_profiler::mark_time(5) );
        cb_reserve_back(CB::c_out0, per_core_block_size);

        for(uint32_t t = 0; t < per_core_block_size; ++t)
        {
            // PACK( kernel_profiler::mark_time(6) );
            // MATH( kernel_profiler::mark_time(5) );
            acquire_dst(DstMode::Half);
            PACK( kernel_profiler::mark_time(7) );
            MATH( kernel_profiler::mark_time(6) );

            UNPACK( kernel_profiler::mark_time(5) );
            cb_wait_front(CB::c_in0, 1);
            UNPACK( kernel_profiler::mark_time(6) );
            cb_wait_front(CB::c_in1, 1);
            UNPACK( kernel_profiler::mark_time(7) );

            // ELTWISE_OP is passed in via add_define
            // MATH( kernel_profiler::mark_time(7) );
            ELTWISE_OP(CB::c_in0, CB::c_in1, 0, 0, 0);
            MATH( kernel_profiler::mark_time(8) );
            // PACK( kernel_profiler::mark_time(8) );
            pack_tile(0, CB::c_out0);
            PACK( kernel_profiler::mark_time(9) );

            UNPACK( kernel_profiler::mark_time(8) );
            cb_pop_front(CB::c_in0, 1);
            UNPACK( kernel_profiler::mark_time(9) );
            cb_pop_front(CB::c_in1, 1);
            UNPACK( kernel_profiler::mark_time(10) );

            // PACK( kernel_profiler::mark_time(10) );
            // MATH( kernel_profiler::mark_time(9) );
            release_dst(DstMode::Half);
            PACK( kernel_profiler::mark_time(11) );
            MATH( kernel_profiler::mark_time(10) );
        }

        cb_push_back(CB::c_out0, per_core_block_size);
        // PACK( kernel_profiler::mark_time(12) );
    }
}
}
