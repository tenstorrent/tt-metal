#pragma once

//#include <string_view> // for constexpr magic

#include "ckernel_include.h" // some of these following 4 deps are necessary to compile because llk_pack_common.h doesn't include all the necessary header deps
#include "ckernel_globals.h"
#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "debug_print.h"
#include "chlkc_list.h"
//#include "llk_defs.h"

#include "kernels/hostdevcommon/kernel_structs.h"

#define SYNC SyncHalf
#if __DOXYGEN__
    #define ALWI
#else
    #define ALWI inline __attribute__((always_inline))
#endif

#define ALWI inline __attribute__((always_inline))

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

ALWI void mm_init() { // TODO(AP): pass cb operands
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_matmul_init() ));
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(0,1,0) ));

    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0) ));
    MATH(( llk_math_pack_sync_init<SYNC>()  ));

    PACK(( llk_pack_init()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16)  ));
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

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));

    MATH(( llk_math_pack_sync_init<SYNC>() ));
}

ALWI void mm_init_short() {
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0)  ));

    UNPACK(( llk_unpack_AB_matmul_init(0)  ));
}

ALWI void acquire_dst(tt::DstMode mode) {
    MATH(( llk_math_wait_for_dest_available<SYNC>()  ));

    PACK(( llk_packer_wait_for_math_done()  ));
}

ALWI void release_dst(tt::DstMode mode) {
    MATH(( llk_math_dest_section_done<SYNC>()  ));

    PACK(( llk_pack_dest_section_done<SYNC>()  ));
}

ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    UNPACK(( llk_wait_tiles(cbid, ntiles)  ));
}

ALWI void cb_pop_front(uint32_t cbid, uint32_t ntiles) {
    UNPACK(( llk_pop_tiles(cbid, ntiles)  ));
}

ALWI void matmul_tiles(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose) {
    UNPACK((  llk_unpack_AB_matmul(c_in0,c_in1,itile0,itile1) ));
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst)  ));
}

ALWI void cb_reserve_back(uint32_t cbid, uint32_t ntiles)
{
    PACK(( llk_wait_for_free_tiles<false,false,false>(cbid,ntiles)  ));
}

ALWI void pack_tile(uint32_t idst, uint32_t cbid)
{
    PACK(( llk_pack<false, SYNC, false >(idst, cbid)  ));
}

ALWI void cb_push_back(uint32_t cbid, uint32_t ntiles)
{
    PACK(( llk_push_tiles<false,false>(cbid, ntiles)  ));
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


// MUL
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

// ADD
ALWI void add_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));

    MATH(( llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, false>(idst) ));
}

// SUB
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

ALWI void gelu_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gelu<APPROX, SyncHalf>(idst) ));
}

ALWI void recip_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>() ));
}

ALWI void recip_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal<APPROX, SyncHalf>(idst) ));
}

ALWI void exp_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exponential_init<APPROX>() ));
}

ALWI void exp_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exponential<APPROX, SyncHalf>(idst) ));
}


ALWI void sqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt_init<APPROX>() ));
}

ALWI void sqrt_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt<APPROX, SyncHalf>(idst) ));
}

ALWI void sigmoid_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void sigmoid_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid<APPROX, SyncHalf>(idst) ));
}

ALWI void log_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void log_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_log<APPROX, SyncHalf>(idst) ));
}

ALWI void tanh_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void tanh_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_tanh<APPROX, SyncHalf>(idst) ));
}

// relu is implemented via unpack with llk_pack_relu_config(0) enabled
ALWI void pack_relu_tile_to_stream(uint32_t idst, uint32_t cbid) {
    PACK(( llk_pack<false, SYNC, false >(idst, cbid) ));
}

ALWI void pack_relu_config(uint32_t enable) {
    PACK(( llk_pack_relu_config(enable) ));
}

#if defined(BCAST_DIM) and defined(BCAST_LLKOP)
void init_bcast(uint32_t icb0, uint32_t icb1)
{
    // BCAST_LLKOP define is either ELWADD, ELWSUB or ELWMUL
    if constexpr (BCAST_LLKOP == ELWMUL) // TODO(AP): check asm
        MATH(( llk_math_eltwise_binary_init<BCAST_LLKOP, BCAST_DIM, MATH_FIDELITY>() )); // TODO(AP)
    else
        MATH(( llk_math_eltwise_binary_init<BCAST_LLKOP, BCAST_DIM>() )); // TODO(AP)

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_init<BCAST_DIM>() )); // TODO(AP): move out init
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BCAST_DIM>(icb0, icb1) ));
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

ALWI void any_tiles_bcast(tt::Dim bt, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<BCAST_LLKOP, BCAST_DIM, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BCAST_DIM>(icb0, icb1, itile0, itile1) ));
}

ALWI void add_tiles_bcast(tt::Dim bt, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast(bt, icb0, icb1, itile0, itile1, idst); }

ALWI void sub_tiles_bcast(tt::Dim bt, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast(bt, icb0, icb1, itile0, itile1, idst); }

ALWI void mul_tiles_bcast(tt::Dim bt, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast(bt, icb0, icb1, itile0, itile1, idst); }

#endif // BCAST_LLKOP

ALWI void add_bcast_rows_init_short() // TODO(AP): generalize or automate
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
}

#if defined(REDUCE_OP) and defined(REDUCE_DIM)
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

ALWI void reduce_tile(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t itile, uint32_t idst, float scaler)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>(idst) ));
    UNPACK(( llk_unpack_reduce<REDUCE_OP, REDUCE_DIM>(icb, itile) ));
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

ALWI void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    UNPACK(( llk_unpack_A<BroadcastType::NONE, true>(icb, itile) ));

    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(idst) ));
}

ALWI void tilize_init(uint32_t icb, uint32_t block)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
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

ALWI void untilize_init(uint32_t icb)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
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
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(0,1) ));
}

} // namespace ckernel

// TODO(AP): use of namespace in a header
using namespace tt; // for CB::c_in visibility
