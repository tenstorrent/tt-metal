// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH COMPUTE (perf idea I4). The rms_norm statistics head, verbatim in
// structure but stripped to one block / one chunk / no tail:
//
//   sumsq_block       eltwise_chain(Mul(in,in), DestAccumulation::PerRow) over the
//                     block's C hidden tiles -> ONE cb_stat_sq tile per tile-row.
//   reduce_stat_block ckl::reduce<SUM, REDUCE_ROW, cb_stat_sq, cb_scaler,
//                     cb_stat_partial, BulkWaitBulkPop> — THE consumer of cb_scaler,
//                     and the reason the scaler tile must be valid at this point and
//                     not one nanosecond earlier.
//
// `cp_wait_scaler` is an idempotent cb_wait_front hoisted out of the reduce helper's
// own wait, so a deferred scaler preparation that ever STARVES the reduce shows up
// as a non-zero number there instead of hiding inside cp_reduce_stat.

#include <stdint.h>

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_stat_sq = 5;
constexpr uint32_t cb_stat_partial = 7;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    // HAS_ANY_TAIL replica: the last hidden tile goes through the masked tail chain
    // (Mul by the row-0 mask, then Square) into its OWN stat column, exactly as
    // rms_norm's mask_tail_block does; the bulk covers the first C-1 tiles.
    constexpr uint32_t WITH_MASK = get_compile_time_arg_val(1);
    constexpr uint32_t C_FULL = C - WITH_MASK;
    constexpr uint32_t NC = 1 + WITH_MASK;

    {
        MaybeDeviceZoneScope("cp_hw_startup");
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_stat_sq);
    }

    {
        // STARVATION, not work: how long unpack sat waiting for the reader's block.
        MaybeDeviceZoneScope("cp_wait_in");
        cb_wait_front(cb_input_tiles, C);
    }

    {
        MaybeDeviceZoneScope("cp_sumsq");
        cb_reserve_back(cb_stat_sq, NC);
        const ckl::StridedTileRange src{0, C};
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(1, C_FULL),
            ckl::BinaryFpu<
                ckl::input(
                    cb_input_tiles,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Strided),
                ckl::input(
                    cb_input_tiles,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Strided),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::Dst::D0,
                ckl::DestAccumulation::PerRow>{src, src},
            ckl::PackTile<ckl::output(
                cb_stat_sq,
                ckl::ReservePolicy::None,
                ckl::PushPolicy::None,
                ckl::DataFormatReconfig::Enabled,
                ckl::PackRelu::Disabled,
                ckl::L1Accumulation::Disabled,
                ckl::DestAccumulation::PerRow,
                ckl::TileOffset::Strided)>{ckl::StridedTileRange{0, NC}});
        if constexpr (WITH_MASK) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(1, 1),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_input_tiles,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Strided),
                    ckl::input(cb_wmask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Scalar),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Row,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::Disabled>{ckl::StridedTileRange{C - 1, C}},
                ckl::Square<>{},
                ckl::PackTile<ckl::output(
                    cb_stat_sq,
                    ckl::ReservePolicy::None,
                    ckl::PushPolicy::None,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::Disabled,
                    ckl::TileOffset::Strided)>{ckl::StridedTileRange{1, NC}});
        }
        cb_push_back(cb_stat_sq, NC);
    }

    {
        MaybeDeviceZoneScope("cp_wait_scaler");
        cb_wait_front(cb_scaler, 1);
    }

    {
        MaybeDeviceZoneScope("cp_reduce_stat");
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_stat_sq,
            cb_scaler,
            cb_stat_partial,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::of(1, NC, 1));
    }

    cb_pop_front(cb_input_tiles, C);
}
