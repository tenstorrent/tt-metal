// Compute_kernel_9
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "llk_defs.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/activations.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/eltwise_unary/bitwise_not.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/clamp.h"
inline uint32_t float_to_bits(float f) {
    uint32_t r;
    __builtin_memcpy(&r, &f, sizeof(r));
    return r;
}
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#include "compute_kernel_api/reduce.h"
void kernel_main() {
    int32_t v1 = 1;
    int32_t v2 = 3;
    size_t v3 = 0;
    cb_reserve_back(get_compile_time_arg_val(3), v2);
    cb_wait_front(get_compile_time_arg_val(0), v1);
    cb_pop_front(get_compile_time_arg_val(0), v1);
    cb_reserve_back(get_compile_time_arg_val(0), v1);
    cb_push_back(get_compile_time_arg_val(0), v1);
    cb_reserve_back(get_compile_time_arg_val(1), v1);
    fill_tile_init();
    unary_op_init_common(get_compile_time_arg_val(1), get_compile_time_arg_val(1));
    float v4 = 0.0e+00f;
    tile_regs_acquire();
    size_t v5 = 0;
    fill_tile(v5, v4);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v5, get_compile_time_arg_val(1), v5);
    tile_regs_release();
    cb_push_back(get_compile_time_arg_val(1), v1);
    cb_wait_front(get_compile_time_arg_val(1), v1);
    DPRINT_UNPACK(DPRINT << "cb_id " << get_compile_time_arg_val(1) << ": { size: "
                         << get_local_cb_interface(0).fifo_size << ", limit: " << get_local_cb_interface(0).fifo_limit
                         << ", page_size: " << get_local_cb_interface(0).fifo_page_size
                         << ", num_pages: " << get_local_cb_interface(0).fifo_num_pages
                         << ", rd_ptr: " << get_local_cb_interface(0).fifo_rd_ptr
                         << ", wr_ptr: " << get_local_cb_interface(0).fifo_wr_ptr
                         << ", wr_tile_ptr: " << get_local_cb_interface(0).fifo_wr_tile_ptr << " }";)
    DPRINT_UNPACK(DPRINT << "cb_idx: " << (uint8_t)get_compile_time_arg_val(1) << " tile_idx: " << 0 << ENDL();)
    DPRINT_UNPACK(DPRINT << "======INPUT======" << ENDL();)
    DPRINT_UNPACK(for (uint16_t r = 0; r < 32; ++r) {)
  DPRINT_UNPACK(  DPRINT << (uint)r << " : ";)
  DPRINT_UNPACK(  for (uint16_t c = 0; c < 32; c+=16) {)
  DPRINT_UNPACK(    DPRINT << " " << TileSlice((uint8_t)get_compile_time_arg_val(1), 0, SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = (uint8_t)1, .w0 = (uint8_t)(c), .w1 = (uint8_t)(c + 16), .ws = (uint8_t)1}, false, true);)
  DPRINT_UNPACK(  })
  DPRINT_UNPACK(  DPRINT << ENDL();)
  DPRINT_UNPACK(
    })
    DPRINT_UNPACK(DPRINT << ENDL();)
    cb_wait_front(get_compile_time_arg_val(0), v1);
    cb_pop_front(get_compile_time_arg_val(0), v1);
    cb_reserve_back(get_compile_time_arg_val(0), v1);
    cb_push_back(get_compile_time_arg_val(0), v1);
    cb_wait_front(get_compile_time_arg_val(0), v1);
    cb_pop_front(get_compile_time_arg_val(0), v1);
    cb_reserve_back(get_compile_time_arg_val(0), v1);
    cb_push_back(get_compile_time_arg_val(0), v1);
    unary_op_init_common(get_compile_time_arg_val(1), get_compile_time_arg_val(3));
    copy_tile_init(get_compile_time_arg_val(1));
    tile_regs_acquire();
    copy_tile(get_compile_time_arg_val(1), v3, v3);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(v3, get_compile_time_arg_val(3), v3);
    tile_regs_release();
    DPRINT_PACK(DPRINT << "cb_id " << get_compile_time_arg_val(3) << ": { size: " << get_local_cb_interface(2).fifo_size
                       << ", limit: " << get_local_cb_interface(2).fifo_limit
                       << ", page_size: " << get_local_cb_interface(2).fifo_page_size
                       << ", num_pages: " << get_local_cb_interface(2).fifo_num_pages << ", rd_ptr: "
                       << get_local_cb_interface(2).fifo_rd_ptr << ", wr_ptr: " << get_local_cb_interface(2).fifo_wr_ptr
                       << ", wr_tile_ptr: " << get_local_cb_interface(2).fifo_wr_tile_ptr << " }";)
    DPRINT_PACK(DPRINT << "cb_idx: " << (uint8_t)get_compile_time_arg_val(3) << " tile_idx: " << 0 << ENDL();)
    DPRINT_PACK(DPRINT << "======OUTPUT======" << ENDL();)
    DPRINT_PACK(for (uint16_t r = 0; r < 32; ++r) {)
  DPRINT_PACK(  DPRINT << (uint)r << " : ";)
  DPRINT_PACK(  for (uint16_t c = 0; c < 32; c+=16) {)
  DPRINT_PACK(    DPRINT << " " << TileSlice((uint8_t)get_compile_time_arg_val(3), 0, SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = (uint8_t)1, .w0 = (uint8_t)(c), .w1 = (uint8_t)(c + 16), .ws = (uint8_t)1}, false, true);)
  DPRINT_PACK(  })
  DPRINT_PACK(  DPRINT << ENDL();)
  DPRINT_PACK(
    })
    DPRINT_PACK(DPRINT << ENDL();)
    return;
}
