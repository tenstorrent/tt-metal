// SPDX-License-Identifier: Apache-2.0
// Hy4 indexer: gather candidates by top-k position (no sort). pos u32 [rows,K] (positions into the W candidates of the
// row), cid u32 [rows,W] (global ids), cval bf16 [rows,W] (scores; only when WANT_VALS) -> oid u32 [rows,K] (=
// cid[pos], position >= W gives 0xFFFFFFFF), oval bf16 [rows,K] (WANT_VALS). Order = top-k order (descending score).
// work item = (row, part), K split in SPLIT parts. The whole candidate row is read into L1 (W*4 (+ W*2) bytes).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_pos = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cid = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cval = get_compile_time_arg_val(2);
    constexpr uint32_t cb_oid = get_compile_time_arg_val(3);
    constexpr uint32_t cb_oval = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(7);
    constexpr uint32_t WANT_VALS = get_compile_time_arg_val(8);
    constexpr uint32_t KP = K / SPLIT;
    constexpr auto pos_args = TensorAccessorArgs<9>();
    constexpr auto cid_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto cval_args = TensorAccessorArgs<cid_args.next_compile_time_args_offset()>();
    constexpr auto oid_args = TensorAccessorArgs<cval_args.next_compile_time_args_offset()>();
    constexpr auto oval_args = TensorAccessorArgs<oid_args.next_compile_time_args_offset()>();

    const uint32_t t_first = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t cid_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t cval_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t oid_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t oval_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto pos_acc = TensorAccessor(pos_args, pos_addr, K * 4);
    const auto cid_acc = TensorAccessor(cid_args, cid_addr, W * 4);
    const auto cval_acc = TensorAccessor(cval_args, cval_addr, W * 2);
    const auto oid_acc = TensorAccessor(oid_args, oid_addr, K * 4);
    const auto oval_acc = TensorAccessor(oval_args, oval_addr, K * 2);
    experimental::CB cbp(cb_pos), cbc(cb_cid), cbv(cb_cval), cbo(cb_oid), cbw(cb_oval);
    cbp.reserve_back(1);
    cbc.reserve_back(1);
    cbv.reserve_back(1);
    cbo.reserve_back(1);
    cbw.reserve_back(1);
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp.get_write_ptr());
    volatile tt_l1_ptr uint32_t* cid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());
    volatile tt_l1_ptr uint16_t* cval = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbv.get_write_ptr());
    volatile tt_l1_ptr uint32_t* oid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
    volatile tt_l1_ptr uint16_t* oval = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbw.get_write_ptr());

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        const uint32_t row = t / SPLIT;
        const uint32_t part = t % SPLIT;
        noc.async_read(pos_acc, cbp, KP * 4, {.page_id = row, .offset_bytes = part * KP * 4}, {.offset_bytes = 0});
        noc.async_read(cid_acc, cbc, W * 4, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = 0});
        if constexpr (WANT_VALS) {
            noc.async_read(cval_acc, cbv, W * 2, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < KP; ++i) {
            const uint32_t p = pos[i];
            oid[i] = (p < W) ? cid[p] : 0xFFFFFFFFu;
            if constexpr (WANT_VALS) {
                oval[i] = (p < W) ? cval[p] : (uint16_t)0xFF80;
            }
        }
        noc.async_write(cbo, oid_acc, KP * 4, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = part * KP * 4});
        if constexpr (WANT_VALS) {
            noc.async_write(
                cbw, oval_acc, KP * 2, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = part * KP * 2});
        }
        noc.async_write_barrier();
    }
}
