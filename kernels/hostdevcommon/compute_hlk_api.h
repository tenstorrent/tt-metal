#pragma once
#include <cstdint>
#include "hostdevcommon/kernel_structs.h"

// Optionally include any defines generated via add_defines() API
#if __has_include("hlk_defines_generated.h")
#include "hlk_defines_generated.h"
#endif

// rk: forcefully remove TT_THROW or TT_ASSERT related, not used for now
// #if defined(HLK_ASSERTIONS_ENABLED) && (HLK_ASSERTIONS_ENABLED == 1)
// #include "common/assert.hpp"
// #else
// #if defined(TT_ASSERT) || defined(TT_THROW)
// FIXME: had to comment this out, it's breaking code that includes common/assert.h + hlks/hlk_api.h
// #error "common/assert.hpp shouldn't have been included down this path"
// #else
// #define TT_ASSERT(condition, ...) ((void)0)
// #define TT_THROW(...) ((void)0)
// #endif
// #endif

  //////////////////////
 // user facing APIs //
//////////////////////

// Compute kernel entry point declaration
#define compute_main(arg)                                                                          hlk_main(tt_core *core_ptr, arg)

#define cb_reserve_back(cb_id, num_tiles)                                                          hlk_wait_for_free_tiles(nullptr, cb_id, num_tiles)
#define cb_wait_front(cb_id, num_tiles)                                                            hlk_wait_tiles(nullptr, cb_id, num_tiles)
#define cb_pop_front(cb_id, num_tiles)                                                             hlk_pop_tiles(nullptr, cb_id, num_tiles)
#define cb_push_back(cb_id, num_tiles)                                                             hlk_push_tiles(nullptr, cb_id, num_tiles)

#define acquire_dst(dst_mode)                                                                      hlk_acquire_dst(nullptr, dst_mode)
#define release_dst(dst_mode)                                                                      hlk_release_dst(nullptr, dst_mode)

#define matmul_tile_init_once(transpose)                                                           hlk_mm_tile_init_once(nullptr, transpose)
#define matmul_tile_init(transpose)                                                                hlk_mm_tile_init(nullptr, transpose)
#define matmul_tile_init_short(face_transpose)                                                     hlk_mm_tile_init_short(nullptr, face_transpose)
#define matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index, transpose)         hlk_mm_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index, transpose)

#define matmul_load_partial_init()                                                                 hlk_load_mm_partial_to_dst_init(nullptr)
#define matmul_load_partial_init_short(face_transpose)                                             hlk_load_mm_partial_to_dst_init_short(nullptr, face_transpose)
#define matmul_load_partial(in_cb_id, in_tile_index, dst_tile_index)                               hlk_load_mm_partial_to_dst(nullptr, in_cb_id, in_tile_index, dst_tile_index)

#define copy_tile_init()                                                                           hlk_copy_tile_to_dst_init(nullptr)
#define copy_tile(in0_cb_id, in0_tile_index, dst_tile_index)                                       hlk_copy_tile_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index)
#define pack_tile(dst_index, cb_id)                                                                hlk_pack_tile_to_stream(nullptr, dst_index, cb_id)

#define add_tiles_init()                                                                           hlk_add_tile_init(nullptr)
#define add_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_add_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define sub_tiles_init()                                                                           hlk_subtract_tile_init(nullptr)
#define sub_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_subtract_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define mul_tiles_init()                                                                           hlk_multiply_tile_init(nullptr)
#define mul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_multiply_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)

#define add_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_add_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define tilize_and_copy(in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)                    hlk_tilize_and_copy_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)
#define untilize_and_copy(in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)                  hlk_untilize_and_copy_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)
#define mul_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_multiply_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define sub_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_subtract_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define exp_tile_init()                                                                            hlk_sfpu_exponential_init(nullptr)
#define exp_tile(dst_tile_index)                                                                   hlk_sfpu_exponential(nullptr, dst_tile_index)
#define gelu_tile_init()                                                                           hlk_sfpu_gelu_init(nullptr)
#define gelu_tile(dst_tile_index)                                                                  hlk_sfpu_gelu(nullptr, dst_tile_index)
#define recip_tile_init()                                                                          hlk_sfpu_reciprocal_init(nullptr)
#define recip_tile(dst_tile_index)                                                                 hlk_sfpu_reciprocal(nullptr, dst_tile_index)
#define sqrt_tile_init()                                                                           hlk_sfpu_sqrt_init(nullptr)
#define sqrt_tile(dst_tile_index)                                                                  hlk_sfpu_sqrt(nullptr, dst_tile_index)
#define reduce_tile(reduce_func, dim, lstream, lindex, dstindex, coeff)                            hlk_reduce_tile(nullptr, reduce_func, dim, lstream, lindex, dstindex, coeff)
#define transpose_wh_tile(in_cb_id, in_tile_index, dst_tile_index)                                 hlk_transpose_xy_tile(nullptr, in_cb_id, in_tile_index, dst_tile_index)

class tt_core;

// IMPORTANT: use "int"s as args instead of enums, this is required for compile-time constant folding in HLKC

void hlk_acquire_dst(tt_core* core_ptr, int dst_mode);
void hlk_release_dst(tt_core* core_ptr, int dst_mode);

// unpack
void hlk_wait_tiles(tt_core* core_ptr, int stream, int num_tiles);

void hlk_pop_tiles(tt_core* core_ptr, int stream, int num_tiles);

// pack
void hlk_wait_for_free_tiles(tt_core* core_ptr, int stream, int num_tiles);
void hlk_push_tiles(tt_core* core_ptr, int stream, int num_tiles);
void hlk_pack_tile_to_stream(tt_core* core_ptr, int dst_index, int stream);
void hlk_pack_relu_tile_to_stream(tt_core* core_ptr, int dst_index, int stream); // could be a flag, but don't want to break legacy api
void hlk_pack_tile_to_stream(tt_core* core_ptr, int dst_index, int stream, int tile_index);

void hlk_get_tile(tt_core* core_ptr, int stream, int index, volatile void* p_tile);
void hlk_release_tile(tt_core* core_ptr, int stream);

void hlk_debug_dump(tt_core* core_ptr, unsigned char *data, int size);


// math
void hlk_copy_tile_to_dst(tt_core* core_ptr, int stream, int index, int dstindex);
void hlk_tilize_and_copy_to_dst(tt_core* core_ptr, int stream, int index, int dstindex, int num_tiles_c);
void hlk_untilize_and_copy_to_dst(tt_core* core_ptr, int stream, int index, int dstindex, int num_tiles_c);
void hlk_load_mm_partial_to_dst(tt_core* core_ptr, int stream, int index, int dstindex);
void hlk_transpose_xy_tile(tt_core* core_ptr, int stream, int index, int dstindex);
void hlk_mm_tile(tt_core* core_ptr, int lstream, int rstream, int lindex, int rindex, int dstindex, int transpose);
void hlk_addbias_tile(tt_core * core_ptr, int stream, int index, int dstindex);
// void hlk_sfpu_tile(tt_core* core_ptr, int sfpu_op, int dstindex);
void hlk_broadcast_tile(tt_core* core_ptr, int dim, int lstream, int lindex, int dstindex);
void hlk_sfpu_exponential(tt_core* core_ptr, int dstindex);
void hlk_sfpu_sqrt(tt_core* core_ptr, int dstindex);
void hlk_sfpu_gelu(tt_core* core_ptr, int dstindex);
void hlk_sfpu_gelu_derivative(tt_core* core_ptr, int dstindex);
void hlk_sfpu_reciprocal(tt_core* core_ptr, int dstindex);
void hlk_sfpu_log(tt_core* core_ptr, int dstindex);
void hlk_sfpu_tanh(tt_core* core_ptr, int dstindex);
void hlk_sfpu_dropout(tt_core* core_ptr, int dstindex, float prob);
void hlk_sfpu_sigmoid(tt_core* core_ptr, int dstindex);
void hlk_add_tile(tt_core* core_ptr, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_add_tile_to_dst(tt_core* core_ptr, int lstream, int lindex, int dstindex, int clear_dst_acc);
void hlk_subtract_tile(tt_core* core_ptr, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_add_tile_bcast(tt_core* core, int dim, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_add_tile_from_dst(tt_core* core_ptr, int rstream, int rindex, int dstindex);
void hlk_subtract_tile_from_dst(tt_core* core_ptr, int rstream, int rindex, int dstindex);
void hlk_multiply_tile_from_dst(tt_core* core_ptr, int rstream, int rindex, int dstindex);
void hlk_add_tile_from_dst_bcast(tt_core* core_ptr, int dim, int rstream, int rindex, int dstindex);
void hlk_subtract_tile_from_dst_bcast(tt_core* core_ptr, int dim, int rstream, int rindex, int dstindex);
void hlk_multiply_tile_from_dst_bcast(tt_core* core_ptr, int dim, int rstream, int rindex, int dstindex);
void hlk_multiply_tile(tt_core* core_ptr, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_multiply_tile_bcast(tt_core* core, int dim, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_subtract_tile_bcast(tt_core* core_ptr, int dim, int lstream, int rstream, int lindex, int rindex, int dstindex);
void hlk_reduce_tile(tt_core* core_ptr, int reduce_func, int dim, int lstream, int lindex, int dstindex, float coefficient);

// placeholder inits
// init_once -> unpacker+packer+math init + hw configure done once at the start of the kernel run
// init -> dynamic unpacker+packer+math init
// init_short -> dynamic unpacker+math init
void hlk_copy_tile_to_dst_init(tt_core* core_ptr);
void hlk_copy_tile_to_dst_init_short(tt_core* core_ptr);
void hlk_copy_tile_to_dst_init_once(tt_core* core_ptr);
void hlk_load_mm_partial_to_dst_init(tt_core* core_ptr);
void hlk_load_mm_partial_to_dst_init_short(tt_core* core_ptr, int within_face_16x16_transpose);
void hlk_mm_tile_init(tt_core* core_ptr, int transpose);
void hlk_mm_tile_init_short(tt_core* core_ptr, int transpose);
void hlk_mm_tile_init_once(tt_core* core_ptr, int transpose);
void hlk_add_tile_init(tt_core* core_ptr);
void hlk_add_tile_init_short(tt_core* core_ptr);
void hlk_add_tile_init_once(tt_core* core_ptr);
void hlk_multiply_tile_init(tt_core* core_ptr);
void hlk_multiply_tile_init_short(tt_core* core_ptr);
void hlk_multiply_tile_init_once(tt_core* core_ptr);
void hlk_subtract_tile_init(tt_core* core_ptr);
void hlk_subtract_tile_init_short(tt_core* core_ptr);
void hlk_subtract_tile_init_once(tt_core* core_ptr);
void hlk_reduce_tile_init(tt_core* core_ptr);
void hlk_reduce_tile_init_short(tt_core* core_ptr);
void hlk_reduce_tile_init_once(tt_core* core_ptr);
void hlk_add_tile_bcast_init(tt_core* core_ptr);
void hlk_add_tile_bcast_init_short(tt_core* core_ptr);
void hlk_add_tile_bcast_init_once(tt_core* core_ptr);
void hlk_multiply_tile_bcast_init(tt_core* core_ptr);
void hlk_multiply_tile_bcast_init_short(tt_core* core_ptr);
void hlk_multiply_tile_bcast_init_once(tt_core* core_ptr);
void hlk_subtract_tile_bcast_init(tt_core* core_ptr);
void hlk_subtract_tile_bcast_init_short(tt_core* core_ptr);
void hlk_subtract_tile_bcast_init_once(tt_core* core_ptr);
void hlk_add_tile_from_dst_init(tt_core* core_ptr);
void hlk_add_tile_from_dst_init_short(tt_core* core_ptr);
void hlk_add_tile_from_dst_init_once(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_init(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_init_short(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_init_once(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_init(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_init_short(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_init_once(tt_core* core_ptr);
void hlk_add_tile_from_dst_bcast_init(tt_core* core_ptr);
void hlk_add_tile_from_dst_bcast_init_short(tt_core* core_ptr);
void hlk_add_tile_from_dst_bcast_init_once(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_bcast_init(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_bcast_init_short(tt_core* core_ptr);
void hlk_subtract_tile_from_dst_bcast_init_once(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_bcast_init(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_bcast_init_short(tt_core* core_ptr);
void hlk_multiply_tile_from_dst_bcast_init_once(tt_core* core_ptr);
void hlk_add_tile_to_dst_init(tt_core* core_ptr);
void hlk_add_tile_to_dst_init_short(tt_core* core_ptr);
void hlk_add_tile_to_dst_init_once(tt_core* core_ptr);
void hlk_sfpu_dropout_init(tt_core* core_ptr, int seed);
void hlk_sfpu_sqrt_init(tt_core* core_ptr);
void hlk_sfpu_exponential_init(tt_core* core_ptr);
void hlk_sfpu_reciprocal_init(tt_core* core_ptr);
void hlk_sfpu_log_init(tt_core* core_ptr);
void hlk_sfpu_gelu_init(tt_core* core_ptr);
void hlk_sfpu_gelu_derivative_init(tt_core* core_ptr);
void hlk_sfpu_sigmoid_init(tt_core* core_ptr);
void hlk_sfpu_max_init(tt_core* core_ptr);
void hlk_sfpu_square_init(tt_core* core_ptr);
void hlk_sfpu_power_init(tt_core* core_ptr);
void hlk_sfpu_sine_init(tt_core* core_ptr);
void hlk_sfpu_cosine_init(tt_core* core_ptr);

void hlk_get_next_op_info(tt_core* core_ptr, op_info_t& op_info_struct);
void hlk_relu_config(tt_core* core_ptr, int config);