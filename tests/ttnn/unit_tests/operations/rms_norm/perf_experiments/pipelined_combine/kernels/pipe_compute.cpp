// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — compute half of idea I9 (pipelined combine rendezvous).
// Companion to pipe_dataflow.cpp. The chains are the op's own:
//   combine  = eltwise_chain BinaryFpu Add, DestAccumulation::PerRow, grid(rows_t, fan_in),
//              host-zeroed identity tile as the B operand;
//   finalize = CopyTile -> AddUnaryColValid{eps} -> RsqrtColValid -> PackTile.
// The ONLY thing MODE 6 changes is the gather operand's WaitPolicy:
//   Upfront    (MODE 0/5) — the chain blocks until the whole fan-in has landed, then
//                           does every add. This is the op's current behaviour.
//   Cumulative (MODE 6)   — the chain waits for k+1 tiles before step k, i.e. it adds
//                           partial k while partials k+1.. are still in flight.
// Precision contract identical in every mode: bf16 pages, HiFi2, fp32_dest_acc_en=false.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_stat_sum = 9;
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_stat_gather2 = 15;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_branch_sum = 18;

template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtColValid : ckl::UnaryOp<RsqrtColValid<Slot>, Slot> {
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, 8, DST_ACCUM_MODE, false, false),
            ckl::to_u32(Slot) + slot_offset,
            VectorMode::C));
    }
};

template <ckl::Dst Slot = ckl::Dst::D0>
struct AddUnaryColValid : ckl::UnaryOp<AddUnaryColValid<Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddUnaryColValid(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t, uint32_t slot_offset) const {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ckernel::ADD_UNARY, 8),
            ckl::to_u32(Slot) + slot_offset,
            VectorMode::C,
            param));
    }
};

template <uint32_t IN_CB, uint32_t OUT_CB, uint32_t SPAN, ckl::WaitPolicy WAIT>
ALWI void combine_span(uint32_t rows_t) {
    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(rows_t, SPAN),
        ckl::BinaryFpu<
            ckl::input(IN_CB, WAIT, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
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
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr bool TWO_STAGE = S2 > 1;
    constexpr uint32_t L1_OUT = TWO_STAGE ? cb_branch_sum : cb_stat_sum;
    constexpr ckl::WaitPolicy WAIT = (MODE == 6 || MODE == 7) ? ckl::WaitPolicy::Cumulative : ckl::WaitPolicy::Upfront;

    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t is_leader = get_arg_val<uint32_t>(2);
    const uint32_t is_root = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_stat_gather, cb_zero_tile, cb_stat_sum);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        if (is_leader) {
            combine_span<cb_stat_gather, L1_OUT, S1, WAIT>(rows_t);
        }
        if constexpr (TWO_STAGE) {
            if (is_root) {
                combine_span<cb_stat_gather2, cb_stat_sum, S2, WAIT>(rows_t);
            }
        }
        if (is_root) {
            finalize_span<cb_stat_sum, cb_rstd_send>(rows_t, EPS_BITS);
        }
    }
}
