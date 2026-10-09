// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Exact fp32 top-K router for prefill (Laguna), row-major outputs for token dispatch. Inputs: sel = sigmoid(logits)
// + bias and scores = sigmoid(logits), both [T, E] fp32 TILE. Work unit = 8 token rows of one 32-row tile row (unit
// u: tile row u / 4, rows (u % 4) * 8 ..); a core takes units core, core + cores, ... and, per unit, reads the 8
// rows of every [32 x 32] tile as one 512-byte run per face (rows 0-15 live in faces 0/1, 16-31 in faces 2/3). Per
// row it picks the K experts with the largest sel (fp32 compared as order-preserving integers; a strictly larger
// key displaces, so equal keys keep the lower expert id) and writes
//   idx[t, :]  = the K expert ids, best first (uint16)
//   wgt[t, :]  = score * routed_scaling / (sum of the K picked scores)   (bf16, round to nearest even)
// as row t of two [T, K] row-major tensors (one page per token).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    constexpr uint32_t norm = get_compile_time_arg_val(4);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t Tt = get_compile_time_arg_val(7);  // 32-row tile rows
    constexpr uint32_t idx_page = get_compile_time_arg_val(8);
    constexpr uint32_t wgt_page = get_compile_time_arg_val(9);
    // both data-movement RISCs run this kernel: RISC r of core c takes units 2 * c + r, 2 * c + r + 2 * cores, ...
    constexpr uint32_t risc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_buf = risc;
    constexpr auto sel_args = TensorAccessorArgs<11>();
    constexpr auto sc_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto wgt_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t Et = E / 32;
    constexpr uint32_t R = 8;                    // rows per unit
    constexpr uint32_t stage = Et * 2 * R * 64;  // bytes of one tensor's unit: [tile][face half][row][16 fp32]

    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sc_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t wgt_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto sel = TensorAccessor(sel_args, sel_addr, tile_bytes);
    const auto sc = TensorAccessor(sc_args, sc_addr, tile_bytes);
    // row-major [T, K] outputs: K * 2-byte pages, addressed with the aligned page size from the accessor args
    const auto idx_out = TensorAccessor(idx_args, idx_addr);
    const auto wgt_out = TensorAccessor(wgt_args, wgt_addr);

    const uint32_t buf = get_write_ptr(cb_buf);
    const uint32_t sel_l1 = buf, sc_l1 = buf + stage, out_l1 = buf + 2 * stage;  // out: R rows x (idx 64 B, wgt 64 B)
    volatile tt_l1_ptr uint32_t* selv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_l1);
    volatile tt_l1_ptr float* scv = reinterpret_cast<volatile tt_l1_ptr float*>(sc_l1);
    union {
        uint32_t u;
        float f;
    } scale;
    scale.u = scale_bits;

    for (uint32_t u = 2 * core + risc; u < Tt * 4; u += 2 * num_cores) {
        const uint32_t g = u / 4, r0 = (u % 4) * R;
        const uint32_t face0 = (r0 / 16) * 2, row_off = (r0 % 16) * 64;
        for (uint32_t j = 0; j < Et; ++j) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t off = (face0 + h) * 1024 + row_off;
                const uint32_t dst = (j * 2 + h) * R * 64;
                noc_async_read(sel.get_noc_addr(g * Et + j) + off, sel_l1 + dst, R * 64);
                noc_async_read(sc.get_noc_addr(g * Et + j) + off, sc_l1 + dst, R * 64);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < R; ++i) {
            // element (row i, expert e) of the staged unit
            auto at = [&](uint32_t e) { return ((e / 32) * 2 + (e % 32) / 16) * R * 16 + i * 16 + (e % 16); };
            uint32_t picked[K];
            uint32_t top[K];
            uint32_t count = 0;
            for (uint32_t e = 0; e < E; ++e) {
                const uint32_t key = order_key(selv[at(e)]);
                if (count == K && key <= top[K - 1]) {
                    continue;
                }
                uint32_t pos = count < K ? count : K - 1;
                while (pos > 0 && top[pos - 1] < key) {
                    top[pos] = top[pos - 1];
                    picked[pos] = picked[pos - 1];
                    --pos;
                }
                top[pos] = key;
                picked[pos] = e;
                if (count < K) {
                    ++count;
                }
            }
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += scv[at(picked[k])];
            }
            const float mult = norm ? scale.f / sum : scale.f;
            volatile tt_l1_ptr uint16_t* io = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1 + i * 128);
            volatile tt_l1_ptr uint16_t* wo = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1 + i * 128 + 64);
            for (uint32_t k = 0; k < K; ++k) {
                union {
                    float f;
                    uint32_t u;
                } w;
                w.f = scv[at(picked[k])] * mult;
                io[k] = static_cast<uint16_t>(picked[k]);
                wo[k] = static_cast<uint16_t>((w.u + 0x7FFFu + ((w.u >> 16) & 1u)) >> 16);
            }
            const uint32_t t = g * 32 + r0 + i;
            noc_async_write(out_l1 + i * 128, idx_out.get_noc_addr(t), K * 2);
            noc_async_write(out_l1 + i * 128 + 64, wgt_out.get_noc_addr(t), K * 2);
        }
        noc_async_write_barrier();
    }
}
