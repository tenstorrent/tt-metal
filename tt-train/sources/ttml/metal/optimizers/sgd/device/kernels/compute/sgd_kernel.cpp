// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

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

template <uint32_t Cb, bool Consume, ckl::OperandKind Kind = ckl::OperandKind::Block>
constexpr ckl::InputSpec block_input() {
    if constexpr (Consume) {
        return ckl::input(Cb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, Kind);
    } else {
        return ckl::input(Cb, ckl::WaitPolicy::None, ckl::PopPolicy::None, Kind);
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
    ckl::OperandKind BKind = ckl::OperandKind::Block>
ALWI void binary_block() {
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(block_size, block_size),
        ckl::BinaryFpu<block_input<CbA, ConsumeA>(), block_input<CbB, ConsumeB, BKind>(), Op, Bcast>{},
        ckl::PackTile<ckl::output(
            CbOut,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Enabled)>{});
}

template <uint32_t GradCb>
ALWI void finish_momentum() {
    cb_wait_front(GradCb, block_size);
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(block_size, block_size),
        ckl::BinaryFpu<
            block_input<cb_momentum_scaled_idx, true>(),
            block_input<GradCb, false>(),
            ckl::BinaryFpuOp::Add>{},
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
        ckl::OperandKind::Scalar>();
    binary_block<
        cb_nesterov_momentum_idx,
        GradCb,
        cb_nesterov_update_idx,
        ckl::BinaryFpuOp::Add,
        ckl::BroadcastDim::None,
        true,
        false>();
    cb_pop_front(GradCb, block_size);
#else
    cb_pop_front(GradCb, block_size);
#endif
}

template <uint32_t GradCb>
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
        ckl::OperandKind::Scalar>();

    if (use_dampening) {
        binary_block<
            GradCb,
            cb_bcast_one_minus_dampening_idx,
            cb_grad_dampened_idx,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::Scalar,
            true,
            false,
            ckl::OperandKind::Scalar>();
        finish_momentum<cb_grad_dampened_idx>();
    } else {
        finish_momentum<GradCb>();
    }

#if USE_NESTEROV
    constexpr auto update_not_scaled = cb_nesterov_update_idx;
#else
    constexpr auto update_not_scaled = cb_momentum_out_idx;
#endif
#else
    constexpr auto update_not_scaled = GradCb;
#endif

    binary_block<
        update_not_scaled,
        cb_bcast_lr_idx,
        cb_update_idx,
        ckl::BinaryFpuOp::Mul,
        ckl::BroadcastDim::Scalar,
        true,
        false,
        ckl::OperandKind::Scalar>();
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
    compute_kernel_hw_startup(cb_grad_idx, cb_bcast_lr_idx, cb_update_idx);

    uint32_t runtime_args_counter = 0;
    const bool use_weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);
    const bool use_dampening = get_arg_val<uint32_t>(runtime_args_counter++);

    cb_wait_front(cb_bcast_lr_idx, 1);
    cb_wait_front(cb_bcast_momentum_idx, 1);
    cb_wait_front(cb_bcast_one_minus_dampening_idx, 1);
    cb_wait_front(cb_bcast_wd_idx, 1);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        cb_wait_front(cb_param_in_idx, block_size);
        if (use_weight_decay) {
            binary_block<
                cb_param_in_idx,
                cb_bcast_wd_idx,
                cb_param_wd_idx,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                false,
                false,
                ckl::OperandKind::Scalar>();
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
