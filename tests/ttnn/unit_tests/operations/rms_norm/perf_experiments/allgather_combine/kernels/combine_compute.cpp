// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — the compute half of the rms_norm cross-core combine.
// Companion to combine_dataflow.cpp; see that file's header for the three modes and
// the race argument. Everything here is the op's own chain code, verbatim in shape:
//   * the combine is `eltwise_chain` BinaryFpu Add with DestAccumulation::PerRow over
//     grid(rows_t, fan_in), the host-zeroed identity tile as the B operand;
//   * the finalize is CopyTile -> AddUnaryColValid{eps} -> RsqrtColValid -> PackTile,
//     with the two column-valid SFPU chain elements copied from the op (a
//     REDUCE_ROW result is column-0-valid, and `ckl::Rsqrt`/`ckl::AddUnary` hardcode
//     VectorMode::RC, so the op builds these as chain ELEMENTS with VectorMode::C).
// What differs per MODE is only WHO runs which stage:
//   MODE 0  leaders combine S1, the root combines S2 and finalizes.
//   MODE 1  leaders combine S1 into cb_branch_sum; EVERY core combines the S2
//           broadcast branch sums and finalizes its own rstd.
//   MODE 2  EVERY core combines the G broadcast partials and finalizes.
// The precision contract is the op's and is identical in every mode: bf16 stat pages,
// MathFidelity::HiFi2, fp32_dest_acc_en = false (16-bit DEST).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_stat_gather2 = 15;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_branch_sum = 18;
constexpr uint32_t cb_ag = 19;

// --- the op's column-valid SFPU chain elements (rms_norm_compute.cpp) ------------
template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtColValid : ckl::UnaryOp<RsqrtColValid<Slot>, Slot> {
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false /* FAST_APPROX */, false /* legacy */),
            ckl::to_u32(Slot) + slot_offset,
            VectorMode::C));
    }
};

template <ckl::Dst Slot = ckl::Dst::D0>
struct AddUnaryColValid : ckl::UnaryOp<AddUnaryColValid<Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddUnaryColValid(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ckernel::ADD_UNARY, 8 /* ITERATIONS */),
            ckl::to_u32(Slot) + slot_offset,
            VectorMode::C,
            param));
    }
};

// Sum `SPAN` gathered tiles per tile-row inside one DEST accumulation.
template <uint32_t IN_CB, uint32_t OUT_CB, uint32_t SPAN>
ALWI void combine_span(uint32_t rows_t) {
    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(rows_t, SPAN),
        ckl::BinaryFpu<
            ckl::input(IN_CB, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(cb_zero_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::Dst::D0,
            ckl::DestAccumulation::PerRow>{},
        ckl::PackTile<ckl::output(
            OUT_CB,
            ckl::ReservePolicy::PerOuter,
            ckl::PushPolicy::PerOuter,
            ckl::DataFormatReconfig::Enabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::PerRow)>{});
}

// rsqrt(x + eps) on the column-0-valid stat tile.
template <uint32_t IN_CB, uint32_t OUT_CB>
ALWI void finalize_span(uint32_t rows_t, uint32_t eps_bits) {
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(rows_t),
        ckl::CopyTile<ckl::input(IN_CB)>{},
        AddUnaryColValid<>{eps_bits},
        RsqrtColValid<>{},
        ckl::PackTile<ckl::output(OUT_CB)>{});
}
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t S1 = get_compile_time_arg_val(1);
    constexpr uint32_t S2 = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr bool TWO_STAGE = S2 > 1;
    constexpr uint32_t AG_SPAN = (MODE == 2) ? G : ((MODE == 4) ? 1 : S2);
    constexpr bool GATHER_TREE = (MODE == 0) || (MODE == 4);
    // A flat tree's level-1 chain IS the whole reduction, so it packs straight into
    // cb_stat_sum (the op's Phase 0 behaviour). Everywhere else it packs the branch sum.
    constexpr uint32_t L1_OUT = (GATHER_TREE && !TWO_STAGE) ? cb_stat_sum : cb_branch_sum;

    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t is_leader = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t is_root = get_arg_val<uint32_t>(3);

    {
        compute_kernel_hw_startup(cb_stat_gather, cb_zero_tile, cb_stat_sum);
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        // MODE 3 — ABLATION (see combine_dataflow.cpp): finalize this core's OWN
        // partial, no collective at all. Wrong answer by design.
        if constexpr (MODE == 3) {
            finalize_span<cb_stat_partial, cb_out>(rows_t, EPS_BITS);
            continue;
        }

        // ---- combine level 1 (row leaders) ---------------------------------
        if constexpr (MODE != 2) {
            if (is_leader) {
                {
                    // The peer RENDEZVOUS, hoisted out of the chain's own Upfront wait
                    // so the gather latency is a separate number from the tile adds.
                    cb_wait_front(cb_stat_gather, rows_t * S1);
                }
                combine_span<cb_stat_gather, L1_OUT, S1>(rows_t);
            }
        }

        if constexpr (GATHER_TREE) {
            if constexpr (TWO_STAGE) {
                if (is_root) {
                    {
                        cb_wait_front(cb_stat_gather2, rows_t * S2);
                    }
                    combine_span<cb_stat_gather2, cb_stat_sum, S2>(rows_t);
                }
            }
            // MODE 0 finalizes ON THE ROOT and broadcasts the result. MODE 4 hands the
            // raw sum to the writer and every core finalizes its own copy below.
            if constexpr (MODE == 0) {
                if (is_root) {
                    finalize_span<cb_stat_sum, cb_rstd_send>(rows_t, EPS_BITS);
                }
            }
        }
        if constexpr (MODE != 0) {
            {
                // The whole collective, seen from the consumer: this is the number the
                // idea is trying to shrink.
                cb_wait_front(cb_ag, rows_t * AG_SPAN);
            }
            if constexpr (AG_SPAN > 1) {
                {
                    combine_span<cb_ag, cb_stat_sum, AG_SPAN>(rows_t);
                }
                finalize_span<cb_stat_sum, cb_out>(rows_t, EPS_BITS);
            } else {
                // A single sender already holds the whole sum: no second combine pass.
                finalize_span<cb_ag, cb_out>(rows_t, EPS_BITS);
            }
        }
    }
}
