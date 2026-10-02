// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

constexpr auto cb_param_in_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_momentum_in_idx = tt::CBIndex::c_2;

constexpr auto cb_param_wd_idx = tt::CBIndex::c_3;
constexpr auto cb_grad_wd_idx = tt::CBIndex::c_4;

constexpr auto cb_momentum_scaled_idx = tt::CBIndex::c_5;
constexpr auto cb_momentum_out_idx = tt::CBIndex::c_6;
constexpr auto cb_momentum_dram_idx = tt::CBIndex::c_7;

constexpr auto cb_grad_dampened_idx = tt::CBIndex::c_8;

constexpr auto cb_nesterov_momentum_idx = tt::CBIndex::c_9;
constexpr auto cb_nesterov_update_idx = tt::CBIndex::c_10;

constexpr auto cb_update_idx = tt::CBIndex::c_11;

constexpr auto cb_bcast_lr_idx = tt::CBIndex::c_12;
constexpr auto cb_bcast_momentum_idx = tt::CBIndex::c_13;
constexpr auto cb_bcast_one_minus_dampening_idx = tt::CBIndex::c_14;
constexpr auto cb_bcast_wd_idx = tt::CBIndex::c_15;

constexpr auto cb_output_idx = tt::CBIndex::c_16;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

template <uint32_t Cb, bool Consume, ckl::InputTileMapping Kind = ckl::InputTileMapping::Block>
constexpr ckl::InputSpec block_input() {
    if constexpr (Consume) {
        return ckl::input(
            Cb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, Kind, ckl::DataFormatReconfig::Disabled);
    } else {
        return ckl::input(Cb, ckl::WaitPolicy::None, ckl::PopPolicy::None, Kind, ckl::DataFormatReconfig::Disabled);
    }
}

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    ckl::BinaryFpuOp Op,
    ckl::BroadcastDim Bcast = ckl::BroadcastDim::None,
    bool ConsumeA = true,
    bool ConsumeB = true,
    ckl::InputTileMapping BKind = ckl::InputTileMapping::Block>
ALWI void binary_block() {
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_size).block_size(block_size),
        ckl::BinaryFpu<Op, block_input<CbA, ConsumeA>(), ckl::input(block_input<CbB, ConsumeB, BKind>(), Bcast)>{},
        ckl::PackTile<ckl::output(
            CbOut,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{});
}

template <uint32_t AliasGradDampened>
ALWI void finish_momentum() {
    cb_wait_front(AliasGradDampened, block_size);
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_size).block_size(block_size),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Add,
            block_input<cb_momentum_scaled_idx, true>(),
            block_input<AliasGradDampened, false>()>{},
        ckl::PackTile<ckl::output(
            cb_momentum_out_idx,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{},
        ckl::PackTile<ckl::output(
            cb_momentum_dram_idx,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{});
#if USE_NESTEROV
    binary_block<
        cb_momentum_out_idx,
        cb_bcast_momentum_idx,
        cb_nesterov_momentum_idx,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();
    binary_block<
        cb_nesterov_momentum_idx,
        AliasGradDampened,
        cb_nesterov_update_idx,
        ckl::BinaryFpuOp::Add,
        ckl::BroadcastDim::None,
        true,
        false>();
    cb_pop_front(AliasGradDampened, block_size);
#else
    cb_pop_front(AliasGradDampened, block_size);
#endif
}

template <uint32_t AliasGradModified>
ALWI void process_update(bool use_dampening) {
#if USE_MOMENTUM
    binary_block<
        cb_momentum_in_idx,
        cb_bcast_momentum_idx,
        cb_momentum_scaled_idx,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();

    if (use_dampening) {
        binary_block<
            AliasGradModified,
            cb_bcast_one_minus_dampening_idx,
            cb_grad_dampened_idx,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::Scalar,
            true,
            false,
            ckl::InputTileMapping::Scalar>();
        finish_momentum<cb_grad_dampened_idx>();
    } else {
        finish_momentum<AliasGradModified>();
    }

#if USE_NESTEROV
    constexpr auto alias_update_not_scaled = cb_nesterov_update_idx;
#else
    constexpr auto alias_update_not_scaled = cb_momentum_out_idx;
#endif
#else
    constexpr auto alias_update_not_scaled = AliasGradModified;
#endif
    // grad * lr
    binary_block<
        alias_update_not_scaled,
        cb_bcast_lr_idx,
        cb_update_idx,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::InputTileMapping::Scalar>();

    // param - grad * lr
    binary_block<
        cb_param_in_idx,
        cb_update_idx,
        cb_output_idx,
        ckl::BinaryFpuOp::Sub,
        ckl::BroadcastDim::None,
        false,
        true>();

    cb_pop_front(cb_param_in_idx, block_size);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const bool use_weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);
    const bool use_dampening = get_arg_val<uint32_t>(runtime_args_counter++);

    compute_kernel_hw_startup(cb_grad_idx, cb_bcast_lr_idx, cb_update_idx);

    cb_wait_front(cb_bcast_lr_idx, 1);
    cb_wait_front(cb_bcast_momentum_idx, 1);
    cb_wait_front(cb_bcast_one_minus_dampening_idx, 1);
    cb_wait_front(cb_bcast_wd_idx, 1);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        cb_wait_front(cb_param_in_idx, block_size);
        if (use_weight_decay) {
            // param * wd
            binary_block<
                cb_param_in_idx,
                cb_bcast_wd_idx,
                cb_param_wd_idx,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                false,
                false,
                ckl::InputTileMapping::Scalar>();

            // param * wd + grad
            binary_block<cb_param_wd_idx, cb_grad_idx, cb_grad_wd_idx, ckl::BinaryFpuOp::Add>();
            process_update<cb_grad_wd_idx>(use_dampening);
        } else {
            process_update<cb_grad_idx>(use_dampening);
        }
    }
    cb_pop_front(cb_bcast_lr_idx, 1);
    cb_pop_front(cb_bcast_momentum_idx, 1);
    cb_pop_front(cb_bcast_one_minus_dampening_idx, 1);
    cb_pop_front(cb_bcast_wd_idx, 1);
}
