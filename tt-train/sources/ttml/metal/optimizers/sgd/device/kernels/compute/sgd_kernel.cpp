// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

constexpr auto dfb_param_in_idx_id = tt::CBIndex::c_0;
constexpr auto dfb_grad_idx_id = tt::CBIndex::c_1;
constexpr auto dfb_momentum_in_idx_id = tt::CBIndex::c_2;

constexpr auto dfb_param_wd_idx_id = tt::CBIndex::c_3;
constexpr auto dfb_grad_wd_idx_id = tt::CBIndex::c_4;

constexpr auto dfb_momentum_scaled_idx_id = tt::CBIndex::c_5;
constexpr auto dfb_momentum_out_idx_id = tt::CBIndex::c_6;
constexpr auto dfb_momentum_dram_idx_id = tt::CBIndex::c_7;

constexpr auto dfb_grad_dampened_idx_id = tt::CBIndex::c_8;

constexpr auto dfb_nesterov_momentum_idx_id = tt::CBIndex::c_9;
constexpr auto dfb_nesterov_update_idx_id = tt::CBIndex::c_10;

constexpr auto dfb_update_idx_id = tt::CBIndex::c_11;

constexpr auto dfb_bcast_lr_idx_id = tt::CBIndex::c_12;
constexpr auto dfb_bcast_momentum_idx_id = tt::CBIndex::c_13;
constexpr auto dfb_bcast_one_minus_dampening_idx_id = tt::CBIndex::c_14;
constexpr auto dfb_bcast_wd_idx_id = tt::CBIndex::c_15;

constexpr auto dfb_output_idx_id = tt::CBIndex::c_16;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

template <uint32_t Dfb, bool Consume, ckl::InputTileMapping Kind = ckl::InputTileMapping::Block>
constexpr ckl::InputSpec block_input() {
    if constexpr (Consume) {
        return ckl::input(Dfb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, Kind);
    } else {
        return ckl::input(Dfb, ckl::WaitPolicy::None, ckl::PopPolicy::None, Kind);
    }
}

template <
    uint32_t DfbA,
    uint32_t DfbB,
    uint32_t DfbOut,
    ckl::BinaryFpuOp Op,
    ckl::BroadcastDim Bcast = ckl::BroadcastDim::None,
    bool ConsumeA = true,
    bool ConsumeB = true,
    ckl::InputTileMapping BKind = ckl::InputTileMapping::Block>
ALWI void binary_block() {
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_size).block_size(block_size),
        ckl::BinaryFpu<Op, block_input<DfbA, ConsumeA>(), ckl::input(block_input<DfbB, ConsumeB, BKind>(), Bcast)>{},
        ckl::PackTile<ckl::output(
            DfbOut,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{});
}

template <uint32_t GradDfb>
ALWI void finish_momentum() {
    DataflowBuffer grad_dfb(GradDfb);
    grad_dfb.wait_front(block_size);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_size).block_size(block_size),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Add,
            block_input<dfb_momentum_scaled_idx_id, true>(),
            block_input<GradDfb, false>()>{},
        ckl::PackTile<ckl::output(
            dfb_momentum_out_idx_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{},
        ckl::PackTile<ckl::output(
            dfb_momentum_dram_idx_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{});

#if USE_NESTEROV
    binary_block<
        dfb_momentum_out_idx_id,
        dfb_bcast_momentum_idx_id,
        dfb_nesterov_momentum_idx_id,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();
    binary_block<
        dfb_nesterov_momentum_idx_id,
        GradDfb,
        dfb_nesterov_update_idx_id,
        ckl::BinaryFpuOp::Add,
        ckl::BroadcastDim::None,
        true,
        false>();
    grad_dfb.pop_front(block_size);
#else
    grad_dfb.pop_front(block_size);
#endif
}

template <uint32_t GradDfb>
ALWI void process_update(bool use_dampening, DataflowBuffer& dfb_param_in) {
#if USE_MOMENTUM
    binary_block<
        dfb_momentum_in_idx_id,
        dfb_bcast_momentum_idx_id,
        dfb_momentum_scaled_idx_id,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();

    if (use_dampening) {
        binary_block<
            GradDfb,
            dfb_bcast_one_minus_dampening_idx_id,
            dfb_grad_dampened_idx_id,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::Scalar,
            true,
            false,
            ckl::InputTileMapping::Scalar>();
        finish_momentum<dfb_grad_dampened_idx_id>();
    } else {
        finish_momentum<GradDfb>();
    }

#if USE_NESTEROV
    constexpr auto update_not_scaled = dfb_nesterov_update_idx_id;
#else
    constexpr auto update_not_scaled = dfb_momentum_out_idx_id;
#endif
#else
    constexpr auto update_not_scaled = GradDfb;
#endif

    binary_block<
        update_not_scaled,
        dfb_bcast_lr_idx_id,
        dfb_update_idx_id,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();
    binary_block<
        dfb_param_in_idx_id,
        dfb_update_idx_id,
        dfb_output_idx_id,
        ckl::BinaryFpuOp::Sub,
        ckl::BroadcastDim::None,
        false,
        true>();
    dfb_param_in.pop_front(block_size);
}

void kernel_main() {
    compute_kernel_hw_startup(dfb_grad_idx_id, dfb_bcast_lr_idx_id, dfb_update_idx_id);

    uint32_t runtime_args_counter = 0;
    const bool use_weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);
    const bool use_dampening = get_arg_val<uint32_t>(runtime_args_counter++);

    DataflowBuffer dfb_param_in(dfb_param_in_idx_id);
    DataflowBuffer dfb_bcast_lr(dfb_bcast_lr_idx_id);
    DataflowBuffer dfb_bcast_momentum(dfb_bcast_momentum_idx_id);
    DataflowBuffer dfb_bcast_one_minus_dampening(dfb_bcast_one_minus_dampening_idx_id);
    DataflowBuffer dfb_bcast_wd(dfb_bcast_wd_idx_id);

    dfb_bcast_lr.wait_front(1);
    dfb_bcast_momentum.wait_front(1);
    dfb_bcast_one_minus_dampening.wait_front(1);
    dfb_bcast_wd.wait_front(1);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        dfb_param_in.wait_front(block_size);
        if (use_weight_decay) {
            binary_block<
                dfb_param_in_idx_id,
                dfb_bcast_wd_idx_id,
                dfb_param_wd_idx_id,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                false,
                false,
                ckl::InputTileMapping::Scalar>();
            binary_block<dfb_param_wd_idx_id, dfb_grad_idx_id, dfb_grad_wd_idx_id, ckl::BinaryFpuOp::Add>();
            process_update<dfb_grad_wd_idx_id>(use_dampening, dfb_param_in);
        } else {
            process_update<dfb_grad_idx_id>(use_dampening, dfb_param_in);
        }
    }
    dfb_bcast_lr.pop_front(1);
    dfb_bcast_momentum.pop_front(1);
    dfb_bcast_one_minus_dampening.pop_front(1);
    dfb_bcast_wd.pop_front(1);
}
