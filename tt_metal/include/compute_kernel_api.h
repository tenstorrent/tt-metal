#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"
#include "src/firmware/riscv/common/risc_attribs.h"

#define SYNC SyncHalf

#if __DOXYGEN__
    #define ALWI
#else
    #define ALWI inline __attribute__((always_inline))
#endif

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_reduce.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_reduce.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_untilize.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif

namespace ckernel {

/**
 * Helper function to reconfigure unpacker srca and srcb input data formats.
 */
ALWI void unpack_reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format(srca_new_operand, srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

/**
 * Helper function to reconfigure packer output data format.
 */
ALWI void pack_reconfig_data_format(const uint32_t new_operand) {
    #ifdef ARCH_GRAYSKULL
        PACK(( llk_pack_reconfig_data_format(new_operand) ));
    #endif
    // NOTE: For wormhole_b0, packer data format reconfig functions don;t yet exist. So skip.
}


ALWI void mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16) {
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_matmul_init() ));
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    MATH(( llk_math_matmul_init<MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SYNC>()  ));

    PACK(( llk_pack_init()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    PACK(( llk_setup_outputs()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>()  ));
    // TODO(AP): ZM-only kernel
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>()  ));
}

ALWI void mm_init_once() {

}

ALWI void unary_op_init_common(uint32_t icb)
{
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE>() ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE>(icb) ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));

    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    MATH(( llk_math_pack_sync_init<SYNC>() ));
}

ALWI void init_sfpu(uint32_t icb) {
    unary_op_init_common(icb);
}

ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1)
{
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(icb0, icb1) ));

    MATH(( llk_math_pack_sync_init<SYNC>() ));


    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));
}

ALWI void mm_init_short_with_dt(uint32_t cbid) {
    UNPACK(( llk_unpack_AB_matmul_init() ));
    UNPACK(( llk_unpack_reconfig_data_format(cbid, 1, 0, 0) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>() ));
}

ALWI void mm_init_short() {
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0)  ));

    UNPACK(( llk_unpack_AB_matmul_init(0)  ));
}

/**
 * Acquires an exclusive lock on the internal DST register for the current
 * Tensix core.
 *
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * This is only available on the compute engine.
 *
 * DOX-TODO(Describe meanings of dst_mode values).
 *
 * Return value: None
 *
 * | Argument | Description                                                | Type     | Valid Range         | Required |
 * |----------|------------------------------------------------------------|----------|---------------------|----------|
 * | dst_mode | Specifies how the destination register is going to be used | DstMode  | Full, Half, Tile    | True     |
 */
ALWI void acquire_dst(tt::DstMode mode) {
    MATH(( llk_math_wait_for_dest_available<SYNC>()  ));

    PACK(( llk_packer_wait_for_math_done()  ));
}

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
ALWI void release_dst(tt::DstMode mode) {
    MATH(( llk_math_dest_section_done<SYNC>()  ));

    PACK(( llk_pack_dest_section_done<SYNC>()  ));
}

// documented in dataflow_api.h
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    UNPACK(( llk_wait_tiles(cbid, ntiles)  ));
}

// documented in dataflow_api.h
ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles) {
    UNPACK(( llk_pop_tiles(cbid, ntiles)  ));
}

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
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void matmul_tiles(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose) {
    UNPACK(( llk_unpack_AB_matmul(c_in0,c_in1,itile0,itile1) ));
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst)  ));
}

// documented in dataflow_api.h
ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles)
{
    PACK(( llk_wait_for_free_tiles<false,false,false>(cbid,ntiles)  ));
}

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
 * Each subsequent pack call will increment the write pointer in the cb by single
 * tile size. The pointer is then again set to a valid position with space for n
 * reserved tiles by another cb_reserve_back call.
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
 * | ifrom_dst      | The index of the tile in the DST register         | uint32_t | Must be less than the size of the DST register (16) | True     |
 * | icb            | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                             | True     |
 * | icb_tile       | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                | True     |
 */
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb)
{
    PACK((  llk_pack<false, SYNC, false >(ifrom_dst, icb)  ));
}

// documented in dataflow_api.h
ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles)
{
    PACK(( llk_push_tiles<false,false>(cbid, ntiles)  ));
}

ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t cbid) {
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, false>() ));
    UNPACK(( llk_unpack_reconfig_data_format(1, cbid, 0, 0) ));
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
}

ALWI void copy_tile_to_dst_init_short()
{
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, false>()  ));

    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>()  ));
}

ALWI void copy_tile_init()
{
    copy_tile_to_dst_init_short();
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
}

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
ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(idst)  ));

    UNPACK(( llk_unpack_A(icb, itile)  ));
}


ALWI void mul_tiles_init_f() { MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() )); }
ALWI void mul_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
}

ALWI void add_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() )); }
ALWI void add_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
}

ALWI void sub_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>() )); }
ALWI void sub_tiles_init() {
    MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>() ));
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
}


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
ALWI void mul_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    //static bool first = true; // TODO(AP): static initializer causes a hang, possibly investigate
    //if (first)
    // one possible solution is to add a local context in the kernel, pass it around and store init flags in it
    // this way the compiler should be able to perform loop hoisting optimization
    // - might need to add __attribute__((pure)) to init calls for this to work
    // Also pass -fmove-loop-invariants to g++
    //mul_tiles_initf();
    //first = false;

    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    MATH(( llk_math_eltwise_binary<ELWMUL, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
}

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
ALWI void add_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    MATH(( llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
}

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
ALWI void sub_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    MATH(( llk_math_eltwise_binary<ELWSUB, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
}

ALWI void binary_op_specific_init(int op_code) // TODO(AP): better naming
{
    #ifdef ELTWISE_OP
    if constexpr (ELTWISE_OP_CODE == 0) // TODO(AP): pass an enum probably
        add_tiles_init_nof();
    if constexpr (ELTWISE_OP_CODE == 1)
        sub_tiles_init_nof();
    if constexpr (ELTWISE_OP_CODE == 2)
        mul_tiles_init_f();
    #endif
}

ALWI void gelu_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gelu_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void gelu_tile(uint32_t idst,bool fast_and_approx=true) {
  if (fast_and_approx) {
    MATH(( llk_math_eltwise_unary_sfpu_gelu<true, SyncHalf>(idst) ));
  } else {
    MATH(( llk_math_eltwise_unary_sfpu_gelu<false, SyncHalf>(idst) ));
  }
}

ALWI void rsqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void rsqrt_tile(uint32_t idst,bool fast_and_approx=true) {
  if (fast_and_approx) {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<true, SyncHalf>(idst) ));
  } else {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<false, SyncHalf>(idst) ));
  }
}

//for these files include eltwise_uanry/erf_erfc.h
//ALWI void erf_tile_init() {
//ALWI void erf_tile(uint32_t idst,bool fast_and_approx=true) {
//ALWI void erfc_tile_init() {
//ALWI void erfc_tile(uint32_t idst,bool fast_and_approx=true) {

ALWI void recip_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void recip_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal<APPROX, SyncHalf>(idst) ));
}

ALWI void exp_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exponential_init<APPROX>() ));
}

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
ALWI void exp_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exponential<APPROX, SyncHalf>(idst) ));
}

ALWI void sqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sqrt_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt<APPROX, SyncHalf>(idst) ));
}

ALWI void sigmoid_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sigmoid_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid<APPROX, SyncHalf>(idst) ));
}

ALWI void log_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_tile.
 */
ALWI void log_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_log<APPROX, SyncHalf>(idst) ));
}

ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));  // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_with_base_tile.
 */
ALWI void log_with_base_tile(uint32_t idst,uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX, SyncHalf>(idst, base_scale)));
}

ALWI void tanh_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void signbit_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_signbit_init<APPROX>() ));
}

ALWI void signbit_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_signbit<APPROX, SyncHalf>(idst) ));
}

ALWI void sin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sin_init<APPROX>() ));
}

ALWI void cos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cos_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void tanh_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_tanh<APPROX, SyncHalf>(idst) ));
}

ALWI void sin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sin<APPROX, SyncHalf>(idst) ));
}

ALWI void cos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_cos<APPROX, SyncHalf>(idst) ));
}

// relu is implemented via unpack with llk_pack_relu_config(0) enabled
ALWI void pack_relu_tile_to_stream(uint32_t idst, uint32_t cbid) {
    PACK(( llk_pack<false, SYNC, false >(idst, cbid) ));
}

ALWI void pack_relu_config(uint32_t enable) {
    PACK(( llk_pack_relu_config(enable) ));
}


//abs
ALWI void abs_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_abs<APPROX, SyncHalf>(idst) ));
}

ALWI void abs_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_abs_init<APPROX>() ));
}

//sign
ALWI void sign_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sign<APPROX, SyncHalf>(idst) ));
}

ALWI void sign_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sign_init<APPROX>() ));
}

//square
ALWI void square_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_square<APPROX, SyncHalf>(idst) ));
}

ALWI void square_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_square_init<APPROX>() ));
}

//compare to zero operators

ALWI void ltz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_ltz<APPROX, SyncHalf>(idst) ));
}

ALWI void ltz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_ltz_init<APPROX>() ));
}


ALWI void eqz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_eqz<APPROX,SyncHalf>(idst) ));
}

ALWI void eqz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_eqz_init<APPROX>() ));
}

ALWI void lez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_lez<APPROX, SyncHalf>(idst) ));
}

ALWI void lez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_lez_init<APPROX>() ));
}

ALWI void gtz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gtz<APPROX, SyncHalf>(idst) ));
}

ALWI void gtz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gtz_init<APPROX>() ));
}

ALWI void nez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_nez<APPROX, SyncHalf>(idst) ));
}

ALWI void nez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_nez_init<APPROX>() ));
}

ALWI void gez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gez<APPROX, SyncHalf>(idst) ));
}

ALWI void gez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gez_init<APPROX>() ));
}


//RELU MAX-MIN ops
ALWI void relu_max_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_relu_max<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_max_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>() ));
}

ALWI void relu_min_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_relu_min<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_min_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>() ));
}

//POWER : y = x^(const param0)
ALWI void power_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_power<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void power_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_power_init<APPROX>() ));
}


/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void init_bcast(uint32_t icb0, uint32_t icb1)
{
    if constexpr (tBcastOp == ELWMUL)
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>() ));
    else
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_init<tBcastDim>() )); // TODO(AP): move out init
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<tBcastDim>(icb0, icb1) ));
    // TODO(AP): running this specific init after common AB init causes a hang

    // clone of general init for AB TODO(AP): commonize
    //UNPACK(( llk_setup_operands() ));
    //UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    //UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(icb0, icb1) ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));
}

template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
ALWI void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<tBcastOp, tBcastDim, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<tBcastDim>(icb0, icb1, itile0, itile1) ));
}

/**
 * This documentation applies to either one of the 3 broadcast operation variants -
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
 * For *dim==BroadcastType::COL*, the input in *B* is expected to be a single tile with a
 * filled 0-column and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[w]*
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
 * | Argument       | Description                                              | Type          | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|---------------|------------------------------------------------|----------|
 * | tBcastDim      | Broadcast dimension                                      | BroadcastType | One of Dim::R, Dim::C, Dim::RC.                | True     |
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t      | 0 to 31                                        | True     |
 * | in1_cb_id      | The indentifier of the circular buffer (CB) containing B | uint32_t      | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t      | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t      | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
template<BroadcastType tBcastDim>
ALWI void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_rows to be executed correctly.
 */
ALWI void add_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_cols to be executed correctly.
 */
ALWI void add_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::COL>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_tiles_bcast_scalar_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>() ));
}

/**
 * Performs a broadcast-multiply of a tile from icb0[itile0] with a scalar encoded as a tile from icb1[itile1].
 */
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<ELWMUL, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1) ));
}


/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

/**
 * Performs a switch-from-another-op tile hw reconfiguration step needed for mul_bcast_rows to be executed correctly.
 */
ALWI void mul_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::ROW, MATH_FIDELITY>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_bcast_cols to be executed correctly.
 */
ALWI void sub_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWSUB, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)
ALWI void reduce_init(PoolType reduce_op, ReduceDim dim, uint32_t icb, float scaler)
{
    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SyncFull>() )); // TODO(AP): check full

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_hw_configure_disaggregated<false,REDUCE_OP, REDUCE_DIM>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncFull, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_reduce_init<REDUCE_OP, REDUCE_DIM>() ));
    UNPACK(( llk_unpack_reduce_hw_configure_disaggregated<REDUCE_OP, REDUCE_DIM>(icb, scaler) ));
}

// TODO(AP): v2 is based on fusion-friendly implementation of reduce, keeping the original version around for now
template<bool at_start>
ALWI void reduce_init_v2(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16)
{
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_init() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(icb, icb_scaler) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SYNC>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));
}

ALWI void reduce_init_short_v2(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16) {
    UNPACK(( llk_unpack_AB_init() ));
    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, false>(ocb) ));
}

// Delta from binary_op_init_common
template<bool at_start>
ALWI void reduce_init_delta_v2(PoolType reduce_op, ReduceDim dim, uint32_t ocb = 16)
{
    UNPACK(( llk_unpack_AB_init() ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));

    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start>(16) ));
}

ALWI void reduce_revert_delta_v2(uint32_t ocb = 16)
{
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, false, true>(ocb) ));
}

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
ALWI void reduce_tile(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t itile, uint32_t idst, float scaler)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>(idst) ));
    UNPACK(( llk_unpack_reduce<REDUCE_OP, REDUCE_DIM>(icb, itile) ));
}

// TODO(AP): v2 is based on fusion-friendly implementation of reduce, keeping the original version around for now
ALWI void reduce_tile_v2(PoolType reduce_op, ReduceDim dim, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>(idst) ));
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1) ));
}
#endif

ALWI void transpose_wh_init(uint32_t icb)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, true>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, false>() ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE, true, true, false>(0) ));
}

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
ALWI void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    UNPACK(( llk_unpack_A<BroadcastType::NONE, true>(icb, itile) ));

    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(idst) ));
}

ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb = 16)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated(icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tilize_init_short(uint32_t icb, uint32_t block)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));

    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{

    UNPACK(( llk_unpack_tilize_block(icb, block) ));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH(( llk_math_wait_for_dest_available<SYNC>() ));
        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0) ));
        PACK(( llk_pack<false, SYNC, false >(0, ocb)  ));

        // Release dest
        MATH(( llk_math_dest_section_done<SYNC>() ));
        PACK(( llk_pack_dest_section_done<SYNC>() ));
    }

}

ALWI void tilize_uninit()
{
    UNPACK(( llk_unpack_tilize_uninit() ));
}

ALWI void untilize_init(uint32_t icb, uint32_t ocb = 16)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_untilize_hw_configure_disaggregated(icb) ));
    UNPACK(( llk_unpack_untilize_init(icb) )); // init must be after configure
}

ALWI void untilize_init_short(uint32_t icb)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));

    UNPACK(( llk_unpack_untilize_init(icb) ));
}

ALWI void untilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{
    UNPACK(( llk_unpack_untilize(icb, block) ));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH(( llk_math_wait_for_dest_available<SYNC>() ));
        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0) ));
        PACK(( llk_pack<false, SYNC, false >(0, ocb)  ));

        // Release dest
        MATH(( llk_math_dest_section_done<SYNC>() ));
        PACK(( llk_pack_dest_section_done<SYNC>() ));
    }
}

ALWI void untilize_uninit(uint32_t icb) {
    UNPACK(( llk_unpack_untilize_uninit(icb) ));
}

ALWI void get_next_op_info(tt::op_info_t& op_info)
{
    MATH(( llk_get_next_op_info(op_info) ));
    PACK(( llk_get_next_op_info(op_info) ));
    UNPACK(( llk_get_next_op_info(op_info) ));
}

ALWI void graph_interpreter_init() // TODO(AP): probably duplicated, remove
{
    MATH(( llk_math_eltwise_unary_sfpu_exponential_init<APPROX>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));
    PACK(( llk_pack_init() ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(0,0) ));
}

//Leaky Relu : y = relu(x) + slope*-relu(-x)
ALWI void leaky_relu_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_leaky_relu<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void leaky_relu_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_leaky_relu_init<APPROX>() ));
}

//elu : y = relu(x) + slope*(exp(x) - 1)*(x <= 0 );
ALWI void elu_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_elu<true, SyncHalf>(idst,param0) ));
}

ALWI void elu_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_elu_init<true>() ));
}

//exp2 : y = 2 ^ x  ==> [y = exp(x * log(2))]
ALWI void exp2_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exp2<true, SyncHalf>(idst) ));
}

ALWI void exp2_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exp2_init<true>() ));
}

//heaviside : y = 0 if x < 0 , 1 if x > 0 , else value
ALWI void heaviside_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void heaviside_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>() ));
}

//expm1 : (exp(x) - 1)
ALWI void expm1_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_expm1<true, SyncHalf>(idst) ));
}

ALWI void expm1_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_expm1_init<true>() ));
}

//arcsine
ALWI void asin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_asin<true, SyncHalf>(idst) ));
}

ALWI void asin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_asin_init<true>() ));
}

//arctan
ALWI void atan_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_atan<true, SyncHalf>(idst) ));
}

ALWI void atan_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_atan_init<true>() ));
}

//arccosine
ALWI void acos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_acos<true, SyncHalf>(idst) ));
}

ALWI void acos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_acos_init<true>() ));
}
} // namespace ckernel
