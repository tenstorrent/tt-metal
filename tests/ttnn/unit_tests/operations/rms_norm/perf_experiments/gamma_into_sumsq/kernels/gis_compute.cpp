// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — "where does rms_norm's apply_gamma pass belong?", compute half.
//
// The shipped op runs THREE full passes over each block's B*S input tiles:
//     cp_sumsq       Sum(x*x) per tile-row, folded in DEST     (reads x, no pack)
//     cp_scale       x *= 1/rms, packed IN PLACE               (post-combine)
//     cp_apply_gamma x *= gamma, packed to the output shard    (post-combine)
// Passes 2 and 3 are both AFTER the cross-core combine only because pass 2 needs
// 1/rms.  Multiplication commutes and gamma does NOT depend on the combine, so the
// gamma multiply can move to either side of it:
//
//   V_BASELINE     sumsq | combine | x*=rms (in place) | x*=gamma (-> out)
//   V_GAMMA_FIRST  sumsq | x*=gamma (in place) | combine | x*=rms (-> out)
//   V_FUSED        [sumsq AND x*gamma in ONE pass] | combine | x*=rms (-> out)
//
// V_GAMMA_FIRST moves the SAME pass (same helper, same chain, same tile and pack
// counts) from the post-combine tail into the window where this thread is otherwise
// blocked in `cp_rms_wait`.  V_FUSED additionally removes one whole pass, but pays
// for it by producing the row partials LATER (the gather cannot start until they
// exist), which is the thing the measurement is for.
//
// ---------------------------------------------------------------------------
// RAW-LLK, and WHY (V_FUSED only)
// ---------------------------------------------------------------------------
// Helper bypassed: `ckl::sum_of_squares` / `ckl::eltwise_chain`.
//
// `sum_of_squares<Input, row_output(cb)>` is `square` with
// DestAccumulation::PerRow, i.e. `BinaryFpu<Mul, x, x, D0, PerRow>` +
// `PackTile<...>`.  A second output is INEXPRESSIBLE in that chain: with any DEST
// accumulation the chain static_asserts
//     "DEST accumulation requires exactly one PackTile"   (chain.inl:2916)
//     "DEST accumulation cannot mix ordinary and accumulating outputs" (:2923)
// and its walk only packs at the END of a row (`per_row_dest_accumulation`,
// chain.inl:3117), so there is no per-tile pack hook at all.
//
// Nor can the pack simply be hoisted out of the accumulating window: on the
// half-sync DEST (`dst_full_sync_en=False`, the USER's config) `tile_regs_release`
// zeroes the packed half and FLIPS the dest-offset id
// (llk_pack_common.h:_llk_pack_dest_section_done_), so the running Sum(x*x) does
// NOT survive a release/acquire pair.  Everything a tile-row needs must live in ONE
// dst-sync window.
//
// Hence the raw form below: one window per tile-row holding
//     D0        = running Sum(x*x) for the row
//     D1..D_S   = x_c * gamma_c, packed in place at the row end
// which needs S+1 DEST slots and is therefore only expressible for
// S + 1 <= DEST_AUTO_LIMIT (8 at fp32_dest_acc_en=False).  No init switching is
// needed inside the loop: ELWMUL on this arch always accumulates into DEST (the
// MOP's acc_to_dest bit is hardcoded, llk_math_eltwise_binary.h:163) and DEST is
// zeroed on release, so `mul_tiles` into a fresh slot IS an assignment and
// `mul_tiles` repeatedly into D0 IS the accumulation -- exactly the mechanism
// `sum_of_squares` itself relies on.  gamma is pre-expanded to FULL tiles once, so
// both multiplies are BroadcastType::NONE ELWMUL over two bf16 CBs and one
// `mul_init` covers the whole pass.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_output_tiles = 9;
constexpr uint32_t cb_thread_sync = 12;

constexpr uint32_t V_BASELINE = 0;
constexpr uint32_t V_GAMMA_FIRST = 1;
constexpr uint32_t V_FUSED = 2;

constexpr uint32_t dest_block_divisor(uint32_t width, uint32_t cap) {
    for (uint32_t d = (cap < width ? cap : width); d > 1; --d) {
        if (width % d == 0) {
            return d;
        }
    }
    return 1;
}

// PACK -> UNPACK ordering edge for an in-place handoff (verbatim from the op).
ALWI void sync_pack_to_unpack() {
    cb_reserve_back(cb_thread_sync, 1);
    cb_push_back(cb_thread_sync, 1);
    cb_wait_front(cb_thread_sync, 1);
    cb_pop_front(cb_thread_sync, 1);
}

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(3);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t IN_CAPACITY_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t DEST_BLOCK_TILES = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(7);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(8);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(9);

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t cb_combine_out = (NUM_OWNERS > 1) ? cb_slice_stat : cb_rms_bcast;

    static_assert(
        VARIANT != V_FUSED || !HAS_GAMMA || (SLICE_HIDDEN_TILES + 1 <= ckl::DEST_AUTO_LIMIT),
        "V_FUSED needs S+1 DEST slots in one window (the row accumulator plus one "
        "transient per hidden tile); it is INEXPRESSIBLE for larger S.");

    constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
    constexpr auto COMBINE_ALGORITHM = NUM_HIDDEN_SLICES >= COMBINE_ACCUMULATE_MIN_TILES
                                           ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                           : ckl::ReduceAlgorithm::Auto;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_owner = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    constexpr auto x_held =
        ckl::input(cb_input_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto rms_col = ckl::input(
        cb_rms_recip,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Col,
        ckl::TileOffset::Unset);
    constexpr auto gamma_row = ckl::input(
        cb_gamma_tiles,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::TileOffset::Unset);
    constexpr auto gamma_expand_in = ckl::input(
        cb_gamma_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, ckl::TileOffset::Unset);
    constexpr auto gamma_expand_out =
        ckl::output(cb_gamma_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    constexpr auto in_place =
        ckl::output(cb_input_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    constexpr auto to_output_batched =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto block_shape = ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES);
    constexpr uint32_t DEST_BLOCK = dest_block_divisor(SLICE_HIDDEN_TILES, DEST_BLOCK_TILES);
    constexpr auto block_shape_batched =
        ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES).block_size(DEST_BLOCK);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t pack_base = (block * BLOCK_TILES) % IN_CAPACITY_TILES;

        {
            MaybeDeviceZoneScope("cp_wait_in");
            cb_wait_front(cb_input_tiles, IN_WAIT_TILES);
        }

        // ================= pre-combine local compute =================
        if constexpr (HAS_GAMMA && VARIANT == V_FUSED) {
            {
                MaybeDeviceZoneScope("cp_gamma_wait");
                cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);
            }
            if (block == 0) {
                MaybeDeviceZoneScope("cp_gamma_expand");
                // Row-0 vector -> FULL tiles, once.  Lets both multiplies below be
                // plain BroadcastType::NONE ELWMULs, so ONE init covers the pass.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(SLICE_HIDDEN_TILES),
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_expand_in>{},
                    ckl::PackTile<gamma_expand_out>{0});
                sync_pack_to_unpack();
            }
            {
                MaybeDeviceZoneScope("cp_sumsq_gamma");
                // ---- RAW: one dst-sync window per tile-row (see the header note) ----
                // The chain that ran last left srcB on a float32 stat CB, so the
                // unpack formats are ours to restore (a chain would fold this).
                reconfig_data_format(cb_input_tiles, cb_gamma_tiles);
                mul_init(cb_input_tiles, cb_gamma_tiles, /*acc_to_dest=*/true);
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                    cb_reserve_back(cb_sq_partials, 1);
                    tile_regs_acquire();
                    for (uint32_t c = 0; c < SLICE_HIDDEN_TILES; ++c) {
                        const uint32_t i = r * SLICE_HIDDEN_TILES + c;
                        mul_tiles(cb_input_tiles, cb_gamma_tiles, i, c, 1 + c);  // D_{1+c} = x*gamma
                        mul_tiles(cb_input_tiles, cb_input_tiles, i, i, 0);      // D0 += x*x
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_input_tiles);
                    for (uint32_t c = 0; c < SLICE_HIDDEN_TILES; ++c) {
                        pack_tile<true>(1 + c, cb_input_tiles, pack_base + r * SLICE_HIDDEN_TILES + c);
                    }
                    pack_reconfig_data_format(cb_sq_partials);
                    pack_tile<true>(0, cb_sq_partials, 0);
                    tile_regs_release();
                    cb_push_back(cb_sq_partials, 1);
                }
                sync_pack_to_unpack();  // x*gamma packed in place; the scale pass unpacks it
            }
        } else {
            {
                MaybeDeviceZoneScope("cp_sumsq");
                ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);
            }
            if constexpr (HAS_GAMMA && VARIANT == V_GAMMA_FIRST) {
                {
                    MaybeDeviceZoneScope("cp_gamma_wait");
                    cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);
                }
                MaybeDeviceZoneScope("cp_apply_gamma");
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, gamma_row>{},
                    ckl::PackTile<in_place>{pack_base});
                sync_pack_to_unpack();  // x*gamma packed in place; the scale pass unpacks it
            }
        }

        // ================= cross-core combine =================
        if constexpr (NUM_HIDDEN_SLICES > 1) {
            if (is_owner) {
                {
                    MaybeDeviceZoneScope("cp_combine_wait");
                    cb_wait_front(cb_gathered_partials, NUM_HIDDEN_SLICES * OWN_ROWS);
                }
                MaybeDeviceZoneScope("cp_combine");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered_partials,
                    cb_scaler,
                    cb_combine_out,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    COMBINE_ALGORITHM,
                    ckl::NoAccumulation,
                    decltype(finalize)>(
                    ckl::ReduceInputBlockShape::of(OWN_ROWS, NUM_HIDDEN_SLICES),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            }
        } else {
            MaybeDeviceZoneScope("cp_collapse");
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_rms_recip,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::Auto,
                ckl::NoAccumulation,
                decltype(finalize)>(
                ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                finalize);
        }

        if constexpr (NUM_HIDDEN_SLICES > 1) {
            MaybeDeviceZoneScope("cp_rms_wait");
            cb_wait_front(cb_rms_recip, BLOCK_ROWS);
        }

        // ================= post-combine local compute =================
        if constexpr (HAS_GAMMA && VARIANT == V_BASELINE) {
            {
                MaybeDeviceZoneScope("cp_scale");
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                    ckl::PackTile<in_place>{pack_base});
                sync_pack_to_unpack();
            }
            {
                MaybeDeviceZoneScope("cp_gamma_wait");
                cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);
            }
            MaybeDeviceZoneScope("cp_apply_gamma");
            ckl::mul<x_held, gamma_row, to_output_batched>(block_shape_batched);
        } else {
            // gamma (if any) is already folded into x; one pass closes the block.
            MaybeDeviceZoneScope("cp_scale");
            ckl::mul<x_held, rms_col, to_output_batched>(block_shape_batched);
        }

        cb_pop_front(cb_input_tiles, BLOCK_TILES);
    }
}
