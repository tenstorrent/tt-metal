// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode recurrence reader for one (b, vh) core. Pulls this head's conv q/k/v
// blocks (GQA: q/k come from key-head vh/RF), the z gate block and the a/b column
// tile from qkvzab, the per-head gate params, the norm weight, and the fp32
// recurrent state. Also fabricates the all-ones tile the compute kernel uses for
// row-sum matmuls and row-select broadcasts.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_qin = 0, cb_kin = 1, cb_vin = 2, cb_zin = 3;
constexpr uint32_t cb_ab = 4, cb_dtb = 5, cb_nega = 6, cb_w = 7, cb_ones = 8;
constexpr uint32_t cb_h = 9;

void kernel_main() {
    constexpr uint32_t NV = get_named_compile_time_arg_val("nv");
    constexpr uint32_t RF = get_named_compile_time_arg_val("rf");
    constexpr uint32_t DKT = get_named_compile_time_arg_val("dkt");
    constexpr uint32_t DVT = get_named_compile_time_arg_val("dvt");
    constexpr uint32_t KD_T = get_named_compile_time_arg_val("kd_t");
    constexpr uint32_t VOFF_T = get_named_compile_time_arg_val("voff_t");
    constexpr uint32_t ZOFF_T = get_named_compile_time_arg_val("zoff_t");
    constexpr uint32_t AB_T = get_named_compile_time_arg_val("ab_t");
    // Anchor-from-stash (commit-by-select as pure data): the state buffer is the
    // rank-4 stash and the anchor row is (b*SEQ_W + accepts[b]) — the row the
    // PREVIOUS verify wrote for the accepted candidate. accepts lives in a
    // device tensor so replays commit with zero host-side device ops.
    constexpr uint32_t SEQ_W = get_named_compile_time_arg_val("seq_rows");
    constexpr bool anchor_from_stash = get_named_compile_time_arg_val("anchor_from_stash") == 1;
    constexpr uint32_t ACC_PAGE = get_named_compile_time_arg_val("acc_page_bytes");
    constexpr bool acc_is_dram = get_named_compile_time_arg_val("acc_is_dram") == 1;
    constexpr bool conv_is_dram = get_named_compile_time_arg_val("conv_is_dram") == 1;
    constexpr bool qkvzab_is_dram = get_named_compile_time_arg_val("qkvzab_is_dram") == 1;
    constexpr bool state_is_dram = get_named_compile_time_arg_val("state_is_dram") == 1;
    constexpr bool params_is_dram = get_named_compile_time_arg_val("params_is_dram") == 1;

    const uint32_t conv_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t qkvzab_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t state_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t dtb_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t nega_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t acc_addr = get_common_arg_val<uint32_t>(6);

    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t vh = get_arg_val<uint32_t>(1);
    const uint32_t kh = vh / RF;

    const uint32_t tb = get_tile_size(cb_qin);   // bf16 operands
    const uint32_t tf = get_tile_size(cb_h);     // fp32 state
    const auto conv_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<conv_is_dram>(), conv_addr, tb);
    const auto qkvzab_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<qkvzab_is_dram>(), qkvzab_addr, tb);
    const auto state_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<state_is_dram>(), state_addr, tf);
    const auto dtb_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<params_is_dram>(), dtb_addr, tb);
    const auto nega_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<params_is_dram>(), nega_addr, tb);
    const auto w_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<params_is_dram>(), w_addr, tb);

    // All-ones fp32 tile for row-sum matmuls and row-select broadcast multiplies
    // (fp32 keeps every ones-pairing same-format with the fp32 intermediates, the
    // combination the chunk_gated_delta_rule kernels already validate). The tile's
    // reserve region doubles as scratch for the accepts read (which completes,
    // and is consumed into anchor_base, before the ones fill overwrites it).
    cb_reserve_back(cb_ones, 1);
    uint32_t anchor_base = (b * NV + vh) * DKT * DVT;
    if constexpr (anchor_from_stash) {
        const auto acc_acc =
            TensorAccessor(tensor_accessor::make_interleaved_dspec<acc_is_dram>(), acc_addr, ACC_PAGE);
        const uint32_t scratch = get_write_ptr(cb_ones);
        noc_async_read_page(0, acc_acc, scratch);
        noc_async_read_barrier();
        const uint32_t m = reinterpret_cast<volatile uint32_t*>(scratch)[b];
        anchor_base = ((b * SEQ_W + m) * NV + vh) * DKT * DVT;
    }
    {
        auto* p = reinterpret_cast<uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t i = 0; i < tf / 4; i++) {
            p[i] = 0x3F800000;  // fp32 1.0
        }
    }
    cb_push_back(cb_ones, 1);

    auto read_block = [&](const auto& acc, uint32_t cb, uint32_t base, uint32_t n, uint32_t page) {
        cb_reserve_back(cb, n);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; t++) {
            noc_async_read_page(base + t, acc, l1 + t * page);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };

    // Gate params first: the compute kernel starts with the scalar pipeline.
    read_block(qkvzab_acc, cb_ab, AB_T, 1, tb);
    read_block(dtb_acc, cb_dtb, 0, 1, tb);
    read_block(nega_acc, cb_nega, 0, 1, tb);

    read_block(conv_acc, cb_qin, kh * DKT, DKT, tb);
    read_block(conv_acc, cb_kin, KD_T + kh * DKT, DKT, tb);
    read_block(conv_acc, cb_vin, VOFF_T + vh * DVT, DVT, tb);
    read_block(qkvzab_acc, cb_zin, ZOFF_T + vh * DVT, DVT, tb);
    read_block(w_acc, cb_w, 0, DVT, tb);
    read_block(state_acc, cb_h, anchor_base, DKT * DVT, tf);
}
