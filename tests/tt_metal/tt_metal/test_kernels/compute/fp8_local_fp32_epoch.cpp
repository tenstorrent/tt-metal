// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
#ifdef USE_DFB_ACCESSORS
    constexpr auto fp8_cb_id = dfb::fp8_input;
    constexpr auto fp32_scale_cb_id = dfb::fp32_scale;
    constexpr auto bf16_cb_id = dfb::bf16_input;
    constexpr auto fp32_out_cb_id = dfb::fp32_output;
    constexpr auto bf16_out_cb_id = dfb::bf16_output;
#else
    constexpr auto fp8_cb_id = tt::CBIndex::c_0;
    constexpr auto fp32_scale_cb_id = tt::CBIndex::c_1;
    constexpr auto bf16_cb_id = tt::CBIndex::c_2;
    constexpr auto fp32_out_cb_id = tt::CBIndex::c_16;
    constexpr auto bf16_out_cb_id = tt::CBIndex::c_17;
#endif
    constexpr uint32_t dst_data = 0;
    constexpr uint32_t dst_scale = 1;

    CircularBuffer fp8_cb(fp8_cb_id);
    CircularBuffer fp32_scale_cb(fp32_scale_cb_id);
    CircularBuffer bf16_cb(bf16_cb_id);
    CircularBuffer fp32_out_cb(fp32_out_cb_id);
    CircularBuffer bf16_out_cb(bf16_out_cb_id);

    compute_kernel_hw_startup(fp8_cb_id, fp32_out_cb_id);

    fp8_cb.wait_front(1);
    fp32_scale_cb.wait_front(1);
    fp32_out_cb.reserve_back(1);

    set_fp32_dest_acc<true>();
    copy_init<true>(fp8_cb_id);
    sfpu_bcast_init<ckernel::BroadcastType::ROW>();

    tile_regs_acquire();
    copy_tile<true>(fp8_cb_id, 0, dst_data);

    reconfig_data_format_srca(fp8_cb_id, fp32_scale_cb_id);
    set_fp32_dest_acc<true>();
    copy_init<true>(fp32_scale_cb_id);
    copy_tile<true>(fp32_scale_cb_id, 0, dst_scale);

    sfpu_bcast<ckernel::BroadcastType::ROW, ckernel::EltwiseBinaryType::ELWMUL, true>(dst_data, dst_scale);

    tile_regs_commit<true>();
    tile_regs_wait();
    pack_tile<false, true>(dst_data, fp32_out_cb_id);
    tile_regs_release<true>();

    fp8_cb.pop_front(1);
    fp32_scale_cb.pop_front(1);
    fp32_out_cb.push_back(1);

    restore_fp32_dest_acc<true>();
    reconfig_data_format_srca(fp32_scale_cb_id, bf16_cb_id);
    pack_reconfig_data_format(fp32_out_cb_id, bf16_out_cb_id);
    copy_init(bf16_cb_id);
    relu_tile_init();

    bf16_cb.wait_front(1);
    bf16_out_cb.reserve_back(1);

    tile_regs_acquire();
    copy_tile(bf16_cb_id, 0, 0);
    relu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, bf16_out_cb_id);
    tile_regs_release();

    bf16_cb.pop_front(1);
    bf16_out_cb.push_back(1);
}
