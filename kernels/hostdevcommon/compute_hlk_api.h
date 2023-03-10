#pragma once
#include <cstdint>
#include "hostdevcommon/kernel_structs.h"

// Optionally include any defines generated via add_defines() API
#if __has_include("hlk_defines_generated.h")
#include "hlk_defines_generated.h"
#endif

// For usage in HLKs only

using namespace tt;

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

/** @file */

  //////////////////////
 // user facing APIs //
//////////////////////

// Compute kernel entry point declaration
#define compute_main(arg)                                                                          hlk_main(tt_core *core_ptr, arg)

#define cb_reserve_back(cb_id, num_tiles)                                                          hlk_wait_for_free_tiles(nullptr, cb_id, num_tiles)
#define cb_wait_front(cb_id, num_tiles)                                                            hlk_wait_tiles(nullptr, cb_id, num_tiles)
#define cb_pop_front(cb_id, num_tiles)                                                             hlk_pop_tiles(nullptr, cb_id, num_tiles)
#define cb_push_back(cb_id, num_tiles)                                                             hlk_push_tiles(nullptr, cb_id, num_tiles)

/**
 * Acquires an exclusive lock on the internal DST register for the current
 * Tensix core. This register is an array of 16 tiles of 32x32 elements each.
 * If the lock is already acquired, this function will wait until it is
 * released.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * DOX-TODO(Describe meanings of dst_mode values).
 *
 * Return value: None
 *
 * | Argument | Description                                                | Type     | Valid Range                                 | Required |
 * |----------|------------------------------------------------------------|----------|---------------------------------------------|----------|
 * | dst_mode | Specifies how the destination register is going to be used | uint32_t | DstMode::Full, DstMode::Half, DstMode::Tile | True     |
 */
#define acquire_dst(dst_mode)                                                                      hlk_acquire_dst(nullptr, dst_mode)

/**
 * Releases the exclusive lock on the internal DST register for the current
 * Tensix core. This lock had to be previously acquired with acquire_dst. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * DOX-TODO(Describe meanings of dst_mode values).
 *
 * | Argument | Description                                                | Type     | Valid Range                                 | Required |
 * |----------|------------------------------------------------------------|----------|---------------------------------------------|----------|
 * | dst_mode | Specifies how the destination register is going to be used | uint32_t | DstMode::Full, DstMode::Half, DstMode::Tile | True     |
 */
#define release_dst(dst_mode)                                                                      hlk_release_dst(nullptr, dst_mode)

#define matmul_tile_init_once(transpose)                                                           hlk_mm_tile_init_once(nullptr, transpose)
#define matmul_tile_init(transpose)                                                                hlk_mm_tile_init(nullptr, transpose)
#define matmul_tile_init_short(face_transpose)                                                     hlk_mm_tile_init_short(nullptr, face_transpose)

/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The indentifier of the second input circular buffer (CB)                | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
#define matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index, transpose)         hlk_mm_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index, transpose)

#define matmul_load_partial_init()                                                                 hlk_load_mm_partial_to_dst_init(nullptr)
#define matmul_load_partial_init_short(face_transpose)                                             hlk_load_mm_partial_to_dst_init_short(nullptr, face_transpose)

/**
 * Loads a submatrix element of a tile-sized matrix into register DST at a
 * specified index for subsequent use with matmul_tiles. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine.
 *
 * DOX-TODO(AP): needs review/better description.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t | Must be less than the size of the CB                | True     |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 */
#define matmul_load_partial(in_cb_id, in_tile_index, dst_tile_index)                               hlk_load_mm_partial_to_dst(nullptr, in_cb_id, in_tile_index, dst_tile_index)

#define copy_tile_init()                                                                           hlk_copy_tile_to_dst_init(nullptr)

/**
 * Copies a single tile from the specified input CB and writes the result to
 * DST at a specified index. For the in_tile_index to be valid for this call,
 * cb_wait_front(n) had to be previously called to ensure that at least some
 * number n>0 of tiles are available in the input CB. The CB index 0 then
 * references the first tile in the received section of the CB, up to index n-1
 * (in a FIFO order). The DST register buffer must be in acquired state via
 * acquire_dst call. This call is blocking and is only available on the compute
 * engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | Yes      |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | Yes      |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | Yes      |
 * */
#define copy_tile(in0_cb_id, in0_tile_index, dst_tile_index)                                       hlk_copy_tile_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index)

/**
 * Copies a single tile from the DST register buffer at a specified index to a
 * specified CB at a given index. For the out_tile_index to be valid for this
 * call, cb_reserve_back(n) had to be called first to reserve at least some
 * number n>0 of tiles in the output CB. The out_tile_index = 0 then references
 * the first tile in the reserved section of the CB, up to index n-1 that will
 * then be visible to the consumer in the same order after a cb_push_back call.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Operates in tandem with functions cb_reserve_back and cb_push_back.
 *
 * A typical use case is first the producer ensures that there is a number of
 * tiles available in the buffer via cb_reserve_back, then the producer uses
 * the pack_tile call to copy a tile from one of DST slots to a slot in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | src_tile_index | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 * | out_cb_id      | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | out_tile_index | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                | True     |
 */
#define pack_tile(dst_index, cb_id)                                                                hlk_pack_tile_to_stream(nullptr, dst_index, cb_id)

#define add_tiles_init()                                                                           hlk_add_tile_init(nullptr)

/**
 * Performs element-wise addition C=A+B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
#define add_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_add_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define sub_tiles_init()                                                                           hlk_subtract_tile_init(nullptr)

/**
 * Performs element-wise subtraction C=A-B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
#define sub_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_subtract_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)
#define mul_tiles_init()                                                                           hlk_multiply_tile_init(nullptr)

/**
 * Performs element-wise multiplication C=A*B of tiles in two CBs at given
 * indices and writes the result to the DST register at index dst_tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
#define mul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)            hlk_multiply_tile(nullptr, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)

/**
 * This document applies to either one of the 3 broadcast operation variants -
 * *add_tiles_bcast*, *sub_tiles_bcast* and *mul_tiles_bcast*.
 *
 * The description below describes *add_tiles_bcast*, the other 2 operations
 * use the same definition with the corresponding substitution of the math
 * operator.
 *
 * Performs a broadcast-operation *C=A+B* of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Broadcasting semantics are defined as follows:
 *
 * For *dim==Dim::R*, the input in *B* is expected to be a single tile with a
 * filled 0-column and zeros elsewhere.  The result is *C[h, w] = A[h,w] +
 * B[w]*
 *
 * For *dim==Dim::C*, the input in *B* is expected to be a single tile with a
 * filled 0-row, and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[h]*
 *
 * For *dim==Dim::RC*, the input in *B* is expected to be a single tile with a
 * filled single value at location [0,0], and zeros elsewhere.  The result is
 * *C[h, w] = A[h,w] + B[0,0]*
 *
 * Return value: None
 *
 * DOX-TODO(AP): verify that the bcast tile is actually required to be filled
 * with zeros.
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | dim            | Broadcast dimension                                      | uint32_t | One of Dim::R, Dim::C, Dim::RC.                | True     |
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The indentifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
#define add_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_add_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
#define mul_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_multiply_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
#define sub_tiles_bcast(dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index) hlk_subtract_tile_bcast(nullptr, dim, in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index)

/**
 * Converts the input tile from a row-major format to a 4-faces row-major
 * format and copies to the DST register at specified index. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * DOX-TODO(AP): this is a guess at this point, needs a review of correctness.
 *
 * DOX-TODO(AP): needs a definition of 4-faces row-major format.
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B       | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | num_tiles_c    | TODO(AP): need to ask what this does.                   | uint32_t | TODO(AP)                                       | True     |
 */
#define tilize_and_copy(in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)                    hlk_tilize_and_copy_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)

/**
 */
#define untilize_and_copy(in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)                  hlk_untilize_and_copy_to_dst(nullptr, in0_cb_id, in0_tile_index, dst_tile_index, num_tiles_c)

#define exp_tile_init()                                                                            hlk_sfpu_exponential_init(nullptr)

/**
 * Performs element-wise computation of exponential on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define exp_tile(dst_tile_index)                                                                   hlk_sfpu_exponential(nullptr, dst_tile_index)

#define gelu_tile_init()                                                                           hlk_sfpu_gelu_init(nullptr)

/**
 * Performs element-wise computation of GELU activation for each element of a
 * tile in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define gelu_tile(dst_tile_index)                                                                  hlk_sfpu_gelu(nullptr, dst_tile_index)

#define recip_tile_init()                                                                          hlk_sfpu_reciprocal_init(nullptr)

/**
 * Performs element-wise computation of 1/x for each element of a tile in DST
 * register at index tile_index. The DST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define recip_tile(dst_tile_index)                                                                 hlk_sfpu_reciprocal(nullptr, dst_tile_index)

#define sqrt_tile_init()                                                                           hlk_sfpu_sqrt_init(nullptr)

/**
 * Performs element-wise computation of sqrt(x) for each element of a tile in
 * DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define sqrt_tile(dst_tile_index)                                                                  hlk_sfpu_sqrt(nullptr, dst_tile_index)

#define sigmoid_tile_init()                                                                        hlk_sfpu_sigmoid_init(nullptr)

/**
 * Performs element-wise computation of sigmoid(x) for each element of a tile in
 * DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define sigmoid_tile(dst_tile_index)                                                              hlk_sfpu_sigmoid(nullptr, dst_tile_index)

#define log_tile_init()                                                                           hlk_sfpu_log_init(nullptr)

/**
 * Performs element-wise computation of log(x) for each element of a tile in
 * DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define log_tile(dst_tile_index)                                                                  hlk_sfpu_log(nullptr, dst_tile_index)

#define tanh_tile_init()                                                                          hlk_sfpu_tanh_init(nullptr)

/**
 * Performs element-wise computation of tanh(x) for each element of a tile in
 * DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
#define tanh_tile(dst_tile_index)                                                                  hlk_sfpu_tanh(nullptr, dst_tile_index)

/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for
 * dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be
 * either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in
 * acquired state via *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                     | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | reduce_func    | Enum value, specifying the type of reduce function to perform.  | uint32_t | One of ReduceFunc::Sum, ReduceFunc::Max        | True     |
 * | dim            | Dimension id, identifying the dimension to reduce in size to 1. | uint32_t | One of Reduce::R, Reduce::C, Reduce::RC        | True     |
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A         | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B               | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | coeff          | Scaling factor applied to each element of the resulting tile.   | float    | any float number                               | True     |
 */
#define reduce_tile(reduce_func, dim, lstream, lindex, dstindex, coeff)                            hlk_reduce_tile(nullptr, reduce_func, dim, lstream, lindex, dstindex, coeff)

/**
 * Performs a 32x32 transpose operation *B[w,h] = A[h,w]* on a tile in the CB
 * at a given index and writes the result to the DST register at index
 * dst_tile_index. The DST register buffer must be in acquired state via
 * *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B       | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
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
void hlk_tilize_and_copy_to_dst_init(tt_core* core_ptr);
void hlk_tilize_and_copy_to_dst_init_short(tt_core* core_ptr);
void hlk_tilize_and_copy_to_dst_init_once(tt_core* core_ptr);
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
void hlk_sfpu_tanh_init(tt_core* core_ptr);
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
