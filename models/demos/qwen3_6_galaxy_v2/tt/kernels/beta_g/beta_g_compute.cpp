// beta_g_compute
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
inline uint32_t float_to_bits(const float f) {
    uint32_t r;
    __builtin_memcpy(&r, &f, sizeof(r));
    return r;
}
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
void kernel_main() {
    int32_t v1 = 1;
    size_t v2 = 1;
    size_t v3 = 0;
    size_t v4 = 2;
    experimental::CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    for (size_t i5 = v3; i5 < v4; i5 += v2) {
        for (size_t j6 = v3; j6 < v4; j6 += v2) {
            cb_ctarg_0.wait_front(v1);
            cb_ctarg_1.wait_front(v1);
            cb_ctarg_2.wait_front(v1);
            cb_ctarg_3.wait_front(v1);
            cb_ctarg_4.wait_front(v1);
            cb_ctarg_5.reserve_back(v1);
            cb_ctarg_6.reserve_back(v1);
            init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(5));
            tile_regs_acquire();
            copy_tile_init(get_compile_time_arg_val(0));
            copy_tile(get_compile_time_arg_val(0), v3, v3);
            sigmoid_tile_init();
            sigmoid_tile(v3);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(v3, get_compile_time_arg_val(5), v3);
            tile_regs_release();
            binary_op_init_common(
                get_compile_time_arg_val(1), get_compile_time_arg_val(2), get_compile_time_arg_val(6));
            tile_regs_acquire();
            copy_tile_init(get_compile_time_arg_val(3));
            copy_tile(get_compile_time_arg_val(3), v3, v3);
            copy_tile_init(get_compile_time_arg_val(4));
            copy_tile(get_compile_time_arg_val(4), v3, v4);
            add_tiles_init(get_compile_time_arg_val(1), get_compile_time_arg_val(2));
            add_tiles(get_compile_time_arg_val(1), get_compile_time_arg_val(2), v3, v3, v2);
            exp_tile_init();
            exp_tile(v3);
            exp_tile(v2);
            negative_tile_init();
            negative_tile(v3);
            add_binary_tile_init();
            add_binary_tile(v4, v2, v2);
            log_tile_init();
            log_tile(v2);
            mul_binary_tile_init();
            mul_binary_tile(v3, v2, v3);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(v3, get_compile_time_arg_val(6), v3);
            tile_regs_release();
            cb_ctarg_6.push_back(v1);
            cb_ctarg_5.push_back(v1);
            cb_ctarg_4.pop_front(v1);
            cb_ctarg_3.pop_front(v1);
            cb_ctarg_2.pop_front(v1);
            cb_ctarg_1.pop_front(v1);
            cb_ctarg_0.pop_front(v1);
        }
    }
    return;
}
