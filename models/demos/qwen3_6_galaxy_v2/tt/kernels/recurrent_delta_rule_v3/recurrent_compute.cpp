// recurrent_compute
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
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
    int32_t v1 = 0;
    float v2 = 0.0e+00f;
    int32_t v3 = 1;
    size_t v4 = 2;
    size_t v5 = 1;
    size_t v6 = 0;
    size_t v7 = 4;
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_9(get_compile_time_arg_val(9));
    experimental::CircularBuffer cb_ctarg_7(get_compile_time_arg_val(7));
    experimental::CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
    experimental::CircularBuffer cb_ctarg_8(get_compile_time_arg_val(8));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    cb_ctarg_4.wait_front(v3);
    cb_ctarg_5.wait_front(v3);
    cb_ctarg_9.reserve_back(v3);
    init_sfpu(get_compile_time_arg_val(9), get_compile_time_arg_val(9));
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(v6, v2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v6, get_compile_time_arg_val(9), v6);
    tile_regs_release();
    cb_ctarg_9.push_back(v3);
    for (size_t i8 = v6; i8 < v7; i8 += v5) {
        cb_ctarg_0.wait_front(v3);
        cb_ctarg_2.wait_front(v3);
        cb_ctarg_3.wait_front(v3);
        cb_ctarg_6.reserve_back(v3);
        mm_block_init(
            get_compile_time_arg_val(2), get_compile_time_arg_val(3), get_compile_time_arg_val(6), v1, v3, v3, v3);
        tile_regs_acquire();
        copy_tile_init(get_compile_time_arg_val(5));
        copy_tile(get_compile_time_arg_val(5), v6, v5);
        mm_block_init_short(get_compile_time_arg_val(2), get_compile_time_arg_val(3), v1, v3, v3, v3);
        matmul_block(get_compile_time_arg_val(2), get_compile_time_arg_val(3), v6, v6, v6, v1, v3, v3, v3);
        mul_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(4));
        mul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(4), v6, v6, v4);
        mul_binary_tile_init();
        mul_binary_tile(v6, v5, v5);
        add_binary_tile_init();
        add_binary_tile(v5, v4, v6);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(v6, get_compile_time_arg_val(6), v6);
        tile_regs_release();
        cb_ctarg_6.push_back(v3);
        cb_ctarg_8.reserve_back(v3);
        mm_block_init(
            get_compile_time_arg_val(2), get_compile_time_arg_val(3), get_compile_time_arg_val(8), v1, v3, v3, v3);
        tile_regs_acquire();
        copy_tile_init(get_compile_time_arg_val(5));
        copy_tile(get_compile_time_arg_val(5), v6, v5);
        mm_block_init_short(get_compile_time_arg_val(2), get_compile_time_arg_val(3), v1, v3, v3, v3);
        matmul_block(get_compile_time_arg_val(2), get_compile_time_arg_val(3), v6, v6, v6, v1, v3, v3, v3);
        mul_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(4));
        mul_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(4), v6, v6, v4);
        mul_binary_tile_init();
        mul_binary_tile(v6, v5, v5);
        add_binary_tile_init();
        add_binary_tile(v5, v4, v6);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(v6, get_compile_time_arg_val(8), v6);
        tile_regs_release();
        cb_ctarg_8.push_back(v3);
        cb_ctarg_3.pop_front(v3);
        cb_ctarg_2.pop_front(v3);
        cb_ctarg_0.pop_front(v3);
        cb_ctarg_1.wait_front(v3);
        cb_ctarg_8.wait_front(v3);
        cb_ctarg_9.wait_front(v3);
        cb_ctarg_9.reserve_back(v3);
        mm_block_init(
            get_compile_time_arg_val(1), get_compile_time_arg_val(8), get_compile_time_arg_val(9), v1, v3, v3, v3);
        tile_regs_acquire();
        copy_tile_init(get_compile_time_arg_val(9));
        copy_tile(get_compile_time_arg_val(9), v6, v6);
        mm_block_init_short(get_compile_time_arg_val(1), get_compile_time_arg_val(8), v1, v3, v3, v3);
        matmul_block(get_compile_time_arg_val(1), get_compile_time_arg_val(8), v6, v6, v6, v1, v3, v3, v3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(v6, get_compile_time_arg_val(9), v6);
        tile_regs_release();
        cb_ctarg_9.push_back(v3);
        cb_ctarg_9.pop_front(v3);
        cb_ctarg_8.pop_front(v3);
        cb_ctarg_1.pop_front(v3);
    }
    cb_ctarg_9.wait_front(v3);
    cb_ctarg_7.reserve_back(v3);
    init_sfpu(get_compile_time_arg_val(9), get_compile_time_arg_val(7));
    tile_regs_acquire();
    copy_tile_init(get_compile_time_arg_val(9));
    copy_tile(get_compile_time_arg_val(9), v6, v6);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v6, get_compile_time_arg_val(7), v6);
    tile_regs_release();
    cb_ctarg_7.push_back(v3);
    cb_ctarg_9.pop_front(v3);
    cb_ctarg_5.pop_front(v3);
    cb_ctarg_4.pop_front(v3);
    return;
}
