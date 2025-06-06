// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// TODO: Do it in everyfile! (company name has been changed)

#include <compute_kernel_api/cb_api.h>
// #include <compute_kernel_api/common_globals.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>

#include <cstdint>

// TODO REMOVE UNNECESSARY INCLUDES
#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
// TODO: remove this using namespace. The functions should be accessible with out it.
using namespace ckernel;

// inline void print_loop(uint32_t count) {
//     UNPACK(DPRINT << "U-LOOP:" << (uint32_t)count << ENDL());
//     // MATH(DPRINT << "M-LOOP:" << (uint32_t)count << ENDL());
//     // PACK(DPRINT << "P-LOOP:" << (uint32_t)count << ENDL());
// }

// inline void print_full_tile_column0(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     UNPACK(DPRINT << "U=====!" << ENDL());
//     // MATH(DPRINT << "M=====!" << ENDL());
//     // PACK(DPRINT << "P=====!" << ENDL());
//     for (uint8_t r = 0; r < 32; ++r) {
//         SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
//         UNPACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
//         // MATH(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
//         // PACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " ");
//     }
//     UNPACK(DPRINT << ENDL() << "U+++++!" << ENDL());
//     // MATH(DPRINT << ENDL() << "M+++++!" << ENDL());
//     // PACK(DPRINT << ENDL() << "P+++++!" << ENDL());
// }

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "U=====!" << ENDL());
    // MATH(DPRINT << "M=====!" << ENDL());
    // PACK(DPRINT << "P=====!" << ENDL());
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        UNPACK(
            DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
                   << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        // MATH(
        //     DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
        //            << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        // PACK(
        //     DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
        //            << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL());
        break;
    }
    UNPACK(DPRINT << "U+++++!" << ENDL());
    // MATH(DPRINT << "M+++++!" << ENDL());
    // PACK(DPRINT << "P+++++!" << ENDL());
}

// 32 x 32
// 0 2 3 4 ... 15 | 16 17 18 19 ... 31

// ~16: 256 - 271
namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size =
    get_compile_time_arg_val(1);                          // Number of tiles in the inner dimention of the input tensor.
constexpr uint32_t mask_w = get_compile_time_arg_val(2);  // Unused atm.
constexpr uint32_t Wt = get_compile_time_arg_val(3);

// Think about move this to compile args to avoid mess while adjusting indicies
//  CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // Unused atm
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;  // Number of activations, i.e. c in the paper
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
// CBs with output data
// Create more intermedaite-output CBs that will be used exclusively by the writer. Do not compute anything on them
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_6;
constexpr uint32_t cb_dL_dgamma_idx = tt::CBIndex::c_7;
// CBs with intermediate computations
constexpr uint32_t cb_scaled_gain = tt::CBIndex::c_8;
constexpr uint32_t cb_gained_dL_dout = tt::CBIndex::c_9;
constexpr uint32_t cb_scale = tt::CBIndex::c_10;
constexpr uint32_t cb_ms_a = tt::CBIndex::c_11;
constexpr uint32_t cb_c_by_ms_a = tt::CBIndex::c_12;
constexpr uint32_t cb_rhs = tt::CBIndex::c_13;
constexpr uint32_t cb_a_over_rms_a = tt::CBIndex::c_14;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_15;

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W  // Unsued atm
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

// TODO: Maybe this should be moved to some utils?
inline void pack_and_push(uint32_t reg, uint32_t cb) {
    // NOTE:
    // The order of commit and wait does not matter when they are next to each other, as they handle different
    // threads. Commit releases the lock for the math thread, allowing the pack thread to start working on the
    // data, while wait is for the pack thread to finish math. In principle, you can commit first and then wait,
    // or wait first and then commit. Logically, it makes sense to say the math procedure is finished (commit)
    // and then packing can start (wait), so commit first and then wait is preferred.
    cb_reserve_back(cb, onetile);
    tile_regs_commit();  // this logically should be in compute functions, bc it belongs to MATH thread
    tile_regs_wait();
    // Q: is this pack_reconfig_data_format necessary? It seems like it is not, but it is better to be sure.
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, onetile);
}

inline void compute_and_pack_mul(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t out_cb) {
    constexpr uint32_t reg = 0;
    tile_regs_acquire();
    mul_tiles_init(cb_a, cb_b);
    mul_tiles(cb_a, cb_b, tile_a, tile_b, reg);
    pack_and_push(reg, out_cb);
}

inline void compute_and_pack_sub(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t out_cb) {
    constexpr uint32_t reg = 0;
    tile_regs_acquire();
    sub_tiles_init(cb_a, cb_b);
    sub_tiles(cb_a, cb_b, tile_a, tile_b, reg);
    pack_and_push(reg, out_cb);
}

// make it FORCE_INLINE
inline void compute_and_pack_div(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t out_cb) {
    const uint32_t reg_a = 0;
    const uint32_t reg_b = 1;
    tile_regs_acquire();
    copy_tile_init(cb_a);
    copy_tile(cb_a, tile_a, reg_a);
    copy_tile_init(cb_b);
    copy_tile(cb_b, tile_b, reg_b);
    div_binary_tile_init();
    div_binary_tile(reg_a, reg_b);  // reg_a = reg_a / reg_b
    pack_and_push(reg_a, out_cb);
}

inline void prepare_sfpu_and_binary_ops() {  // tmp
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_gamma_idx, onetile);

    if constexpr (do_mask_w) {
        // cb_wait_front(cb_mask_w_idx, onetile);
    }

    // We need to init_sfpu with cb_input_idx and cb_dL_da_idx, so that it knows how to handle the data formats. Since
    // all computations are done in bfloat16, we do not need to reconfigure the SFPU for each operation.
    init_sfpu(cb_input_idx, cb_dL_da_idx);

    // Should be here not sure exactly why, but it is needed to initialize the SFPU for the binary operations. TBC
    // later.
    binary_op_init_common(cb_input_idx, cb_gamma_idx, cb_dL_da_idx);

    // What is the purpose of reconfig_data_format here? It might work without it, but it is better to be sure that the
    // data format is correct.
    // reconfig_data_format(cb_input_idx, cb_gamma_idx);
}

inline void compute_scaled_gain_and_gained_dL_dout(uint32_t col) {
    // 2. Compute:
    // auto scaled_gain = ttnn::divide(
    //     g,
    //     rms_a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [1,1,1,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
    // auto gained_dL_dout = ttnn::multiply(
    //     scaled_gain,
    //     dL_dout,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
    // UNPACK(DPRINT << "gamma" << ENDL());
    // print_full_tile(cb_gamma_idx, col, true);
    // UNPACK(DPRINT << "rms_a" << ENDL());
    // print_full_tile(cb_rms_a_idx, col, true);

    uint32_t rms_register = 0;
    tile_regs_acquire();
    // wait_front for rms_a, gamma and dL_out has been called before this function, so we can safely use them.
    unary_bcast_init<BroadcastType::COL>(cb_rms_a_idx, cb_rms_a_idx);
    // UNPACK(DPRINT << "col: " << col << ENDL());
    unary_bcast<BroadcastType::COL>(cb_rms_a_idx, /* tile idx */ col, /* reg tile idx */ rms_register);
    cb_pop_front(cb_rms_a_idx, onetile);
    pack_and_push(rms_register, cb_rms_a_idx);
    // UNPACK(DPRINT << "rms_a" << ENDL());
    // print_full_tile(cb_rms_a_idx, col, true);

    // Let's compute scaled_gain, pack it to cb_scaled_gain, and multiply it with dL_out to get gained_dL_dout in FPU.
    // cb_wait_front(cb_rms_a_idx, onetile);
    // Why this wait_front is hanging?
    compute_and_pack_div(cb_gamma_idx, cb_rms_a_idx, /* tile_a */ col, /* tile_b */ col, cb_scaled_gain);
    // UNPACK(DPRINT << "scaled_gain" << ENDL());
    // print_full_tile(cb_scaled_gain, 0, true);

    // UNPACK(DPRINT << "dL_out" << ENDL());
    // print_full_tile(cb_dL_out_idx, col, true);

    // We can use tile idx 0 for all cols as we popfront the cb_scaled_gain in each iteration.
    compute_and_pack_mul(cb_scaled_gain, cb_dL_out_idx, /* tile_a */ 0, /* tile_b */ col, cb_gained_dL_dout);
    cb_pop_front(cb_scaled_gain, onetile);

    // UNPACK(DPRINT << "cb_gained_dL_dout" << ENDL());
    // cb_wait_front(cb_gained_dL_dout, 1);
    // print_full_tile(cb_gained_dL_dout, col, true);
}

inline void compute_scale(uint32_t col) {
    // 3. Compute:
    // auto scale = ttml::ttnn_fixed::sum_over_dim(
    //     ttnn::multiply(a, gained_dL_dout, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
    //     3);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C] -> [B,1,S,1]
    //
    // We will calculate scale iteratively reducting it to a single, scalar value after each step.
    const uint32_t scale_register = 0;            // destination register for the reduction
    const uint32_t scale_reduction_register = 1;  // register for the reduction
    tile_regs_acquire();
    // think if that should be Wt or 1
    cb_wait_front(cb_gained_dL_dout, Wt);
    cb_wait_front(cb_input_idx, Wt);
    // UNPACK(DPRINT << "cb_gained_dL_dout" << ENDL());
    // print_full_tile(cb_gained_dL_dout, col, true);
    UNPACK(DPRINT << "cb_input_idx" << ENDL());
    print_full_tile(cb_input_idx, col, true);

    // Perform elementwise multiplication and sum reduction in one step
    reduce_init<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_gained_dL_dout, cb_input_idx, cb_scale);

    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(  // (sum over inner dimension)
        cb_gained_dL_dout,                              // main input buffer
        cb_input_idx,                                   // scaler buffer (elementwise mul)
        /* tile_idx */ col,                             // tile index in main buffer
        /* tile_idx */ col,                             // tile index in scaler buffer
        scale_register);                                // destination register
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_gained_dL_dout);

    if (col == 0) {
        copy_dest_values_init();
        copy_dest_values(scale_reduction_register, scale_register);
    } else {
        // Think how many ntiles should be here. I guess 1 is enough, but not col then but 0.
        cb_wait_front(cb_scale, Wt);
        copy_tile_init(cb_scale);
        copy_tile(cb_scale, /* tile_idx */ col, /* register_idx */ scale_reduction_register);
        // NOTE: Keep in mind that this is not the best idea to put everything in CB at one. L1 means only that
        // we read once, but we shouldn't use ~30 CBs filling them all with whole inner dimension. The reduction
        // that we do afterwards could be done here, reducing memory usage. Therefore we clean the cb_scale
        // after the first tile, so that we do not use so much memory for the cb_scale.
        cb_pop_front(cb_scale, onetile);
        add_binary_tile_init();
        add_binary_tile(scale_reduction_register, scale_register);
    }

    pack_and_push(scale_reduction_register, cb_scale);
    cb_wait_front(cb_scale, 1);
    UNPACK(DPRINT << "cb_scale" << ENDL());
    print_full_tile(cb_scale, col, true);
}

inline void compute_ms_a_and_c_by_ms_a() {
    // TODO: This comment should be improved.
    // 4. Compute c_by_ms_a. This can be done outside above loop, because rms_a and c are constant across all tiles
    // in the row.
    //
    // auto ms_a = ttnn::square(rms_a);  // [B,1,S,1] -> [B,1,S,1]
    // auto c_by_ms_a = ttnn::multiply(
    //     ms_a, c, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // [B,1,S,1] x [1] ->
    // [B,1,S,1] (bcast)

    // NOTE: be careful rms_a is bcasted across all cols here. See compute_scaled_gain_and_gained_dL_dout
    // Not sure if this is expected behaviour here
    // NOTE2: RMS is bcasted and scaler is filled with the same value for all cols so we would have this outputvalue
    // already bcasted. Not sure if we want to have like this, but it is how it is now.

    // cb_wait_front(cb_rms_a_idx, onetile);
    // cb_wait_front(cb_scaler_idx, onetile);
    // We do not need to wait for cb_rms_a_idx and cb_scaler_idx, because they are already filled with the
    // bcasted values. We can use them directly.

    // UNPACK(DPRINT << "cb_rms_a_idx" << ENDL());
    // print_full_tile(cb_rms_a_idx, 0, true);
    compute_and_pack_mul(cb_rms_a_idx, cb_rms_a_idx, /* tile_a */ 0, /* tile_b */ 0, cb_ms_a);
    // UNPACK(DPRINT << "cb_ms_a" << ENDL());
    // print_full_tile(cb_ms_a, 0, true);
    // UNPACK(DPRINT << "cb_scaler_idx" << ENDL());
    // print_full_tile(cb_scaler_idx, 0, true);
    // NOTE: div because scale is 1/c - maybe it should be changed to c
    compute_and_pack_div(cb_ms_a, cb_scaler_idx, /* tile_a */ 0, /* tile_b */ 0, cb_c_by_ms_a);
    // cb_wait_front(cb_c_by_ms_a, onetile);
    // UNPACK(DPRINT << "cb_c_by_ms_a" << ENDL());
    // print_full_tile(cb_c_by_ms_a, 0, true);
    // We can pop_front cb_ms_a, since we do not need it anymore.
    cb_pop_front(cb_ms_a, onetile);
}

inline void compute_rhs(uint32_t col) {
    // 5. Compute:
    // auto scaled_outer = ttnn::multiply(
    //     scale,
    //     a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,1] x [B,1,S,C] -> [B,1,S,C] (bcast)
    // auto rhs = ttnn::divide(
    //     scaled_outer,
    //     c_by_ms_a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast)

    // NOTE: I don't like the fact that we are coping c_by_ms_a to register every time, but idk if
    // there is a way to avoid it. We need to have it in register to perform the division.

    // NOTE: Do not use compute_and_pack_mul and compute_and_pack_div here, because we would unnecessarily pack and
    // unpack the intermediate data to/from CB. We can use registers to perform the operations, and then pack the result
    // to cb_rhs.
    cb_wait_front(cb_scale, onetile);
    UNPACK(DPRINT << "cb_scale" << ENDL());
    print_full_tile(cb_scale, col, true);

    uint32_t scale_register = 0;
    tile_regs_acquire();
    // wait_front for rms_a, gamma and dL_out has been called before this function, so we can safely use them.
    unary_bcast_init<BroadcastType::COL>(cb_scale, cb_scale);
    unary_bcast<BroadcastType::COL>(cb_scale, /* tile idx */ col, /* reg tile idx */ scale_register);
    cb_pop_front(cb_scale, onetile);
    pack_and_push(scale_register, cb_scale);
    cb_wait_front(cb_scale, onetile);
    UNPACK(DPRINT << "cb_scale" << ENDL());
    print_full_tile(cb_scale, col, true);

    cb_wait_front(cb_input_idx, onetile);  // cannot stack them
    UNPACK(DPRINT << "cb_input_idx" << ENDL());
    print_full_tile(cb_input_idx, col, true);
    uint32_t rhs_register = 0;
    uint32_t c_by_ms_a_register = 1;
    tile_regs_acquire();
    // cb1 * cb2 = reg1
    // reg1 /= reg2

    mul_tiles_init(cb_input_idx, cb_scale);
    // We can use tile idx 0 for all cols as we have a reducted, single value in cb_scale.
    mul_tiles(cb_input_idx, cb_scale, /* tile_idx */ col, /* tile_idx */ 0, rhs_register);
    // pack_and_push(rhs_register, cb_rhs);
    // cb_wait_front(cb_rhs, onetile);
    // UNPACK(DPRINT << "cb_rhs (scaled_outer in practise)" << ENDL());
    // print_full_tile(cb_rhs, col, true);

    cb_wait_front(cb_c_by_ms_a, onetile);
    copy_tile_init(cb_c_by_ms_a);
    copy_tile(cb_c_by_ms_a, /* tile_idx */ 0, /* register_idx */ c_by_ms_a_register);
    // this bcast here is prob unnecessary, because we bcasted rms_a, so c_by_ms_a is already bcasted
    // // unary_bcast_init<BroadcastType::COL>(cb_c_by_ms_a, cb_c_by_ms_a);
    // // unary_bcast<BroadcastType::COL>(cb_c_by_ms_a, col, c_by_ms_a_register);
    // UNPACK(DPRINT << "cb_c_by_ms_a" << ENDL());
    // print_full_tile(cb_c_by_ms_a, 0, true);

    div_binary_tile_init();  // Q: can this clear out the rhs_register?
    div_binary_tile(rhs_register, c_by_ms_a_register);

    // // Now we have rhs in rhs_register, we can pack it to cb_rhs.
    pack_and_push(rhs_register, cb_rhs);
    cb_wait_front(cb_rhs, onetile);
    UNPACK(DPRINT << "cb_rhs" << ENDL());
    // // UNPACK(DPRINT << "cb_rhs col: " << col << ENDL());
    print_full_tile(cb_rhs, col, true);
}

inline void compute_dL_da(uint32_t col) {
    // 6. Compute:
    // auto dL_da = ttnn::subtract(
    //     gained_dL_dout,
    //     rhs,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
    // cb_wait_front(cb_rhs, Wt);
    // cb_wait_front(cb_gained_dL_dout, Wt);
    // UNPACK(DPRINT << "inside compute dL_da" << ENDL());
    // UNPACK(DPRINT << "cb_rhs" << ENDL());
    // print_full_tile(cb_rhs, col, true);
    // UNPACK(DPRINT << "cb_gained_dL_dout" << ENDL());
    // print_full_tile(cb_gained_dL_dout, col, true);

    compute_and_pack_sub(cb_gained_dL_dout, cb_rhs, /* tile_a */ col, /* tile_b */ col, cb_dL_da_idx);
    cb_wait_front(cb_dL_da_idx, onetile);
    UNPACK(DPRINT << "cb_dL_da_idx" << ENDL());
    print_full_tile(cb_dL_da_idx, col, true);
    // // We can pop_front rsh, since we do not need it anymore.
    // cb_pop_front(cb_rhs, onetile);

    // dL_dgamma comps:
    // compute_and_pack_div(cb_input_idx, cb_rms_a_idx, /* tile_a */ col, /* tile_b */ 0, cb_a_over_rms_a);
    // compute_and_pack_mul(cb_dL_out_idx, cb_a_over_rms_a, /* tile_a */ 0, /* tile_b */ 0, cb_dL_dgamma_components);

    // cb_wait_front(cb_dL_dgamma_components, onetile);
    // UNPACK(DPRINT << "cb_dL_dgamma_components" << ENDL());
    // print_full_tile(cb_dL_dgamma_components, col, true);
}

// inline void compute_dL_dgamma_components(uint32_t col) {
//     // 7. Compute:
//     // auto dL_dg_components = ttnn::multiply(
//     //     dL_dout,
//     //     ttnn::divide(a, rms_a, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
//     //     std::nullopt,
//     //     std::nullopt,
//     //     std::nullopt,
//     //     none,
//     //     none,
//     //     none,
//     //     false);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast); checked by add_grad
//     // auto dL_dg = ttnn::sum(
//     //     dL_dg_components,
//     //     /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
//     //     /* keep_dim */ true,
//     //     /* output_mem_config */ std::nullopt,
//     //     /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] -> [1,1,1,C]
//     // NOTE: To compute dL_dg, we need to process all batches. Therefore, we will compute here only dL_dg_components
//     // for each tile, and then store them in CB. The reduction will be done in a separate program.

//     // Let's compute and pack it, so that we can perform the multiplication on FPU.
//     compute_and_pack_div(cb_input_idx, cb_rms_a_idx, /* tile_a */ col, /* tile_b */ 0, cb_a_over_rms_a);

//     // Now we can perform the multiplication with dL_out.
//     // We can use tile idx 0 for all cols as we do not need to store all of the a over rms_a values, but only the
//     // current tile value.
//     compute_and_pack_mul(cb_dL_out_idx, cb_a_over_rms_a, /* tile_a */ 0, /* tile_b */ 0, cb_dL_dgamma_components);
//     // We can pop_front cb_a_over_rms_a, since we do not need it anymore.
//     cb_pop_front(cb_a_over_rms_a, onetile);

//     // cb_wait_front(cb_dL_dgamma_components, onetile);
//     // UNPACK(DPRINT << "cb_dL_dgamma_components" << ENDL());
//     // print_full_tile(cb_dL_dgamma_components, col, true);
// }

// Figure out why MAIN without ( )
void MAIN {
    prepare_sfpu_and_binary_ops();

    // UNPACK(DPRINT << "whatever" << ENDL());
    // UNPACK(DPRINT << Wt << " tiles in row " << ENDL());
    // UNPACK(DPRINT << "num_rows_per_core: " << num_rows_per_core << ENDL());

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // 1. Wait for the input tensor, rms_a and dL_out to be ready.
        cb_wait_front(cb_input_idx, Wt);
        // RMS(a) is a scalar, so we wait for one tile only.
        cb_wait_front(cb_rms_a_idx, onetile);
        cb_wait_front(cb_dL_out_idx, Wt);

        for (uint32_t col = 0; col < Wt; ++col) {
            compute_scaled_gain_and_gained_dL_dout(col);
            UNPACK(DPRINT << "scaled_gain and gained_dL_dout done " << ENDL());
            compute_scale(col);
            UNPACK(DPRINT << "scale done " << ENDL());
        }
        compute_ms_a_and_c_by_ms_a();
        // break;

        // We need to store in registers scale and c_by_ms_a, and iterate over all tiles in cb_input_idx to calculate
        // rhs for each tile.
        for (uint32_t col = 0; col < Wt; ++col) {
            compute_rhs(col);
            // nop to check sync
            for (uint32_t i = 0; i < 100000; ++i) {
                asm volatile("nop");
            }
            compute_dL_da(col);
            // compute_dL_dgamma_components(col);
        }

        // pop from the input CBs

        // TODO Make sure that we wait and resever for all necessary data in buffers! (probably not)
        // TODO2 Make sure if we do not need to reconfigure data format for any calculation. I neglected it for now.
    }
    UNPACK(DPRINT << "compute kernel done" << ENDL());
}

}  // namespace NAMESPACE
