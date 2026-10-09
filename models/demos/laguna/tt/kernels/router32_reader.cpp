// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Exact fp32 top-K router for one 32-token decode tile (Laguna). Core t owns token row t: it reads row t of the
// selection scores sel = sigmoid(logits) + bias and of the unbiased scores = sigmoid(logits) (both [32, E] fp32
// TILE), picks the K experts with the largest sel (fp32 compared as order-preserving integers), and writes row t
// of dense = score * routed_scaling / (sum of the K picked scores) at the picked experts and 0 elsewhere.
// A row of an fp32 tile is two 64-byte runs: face (r / 16) * 2 for columns 0-15 and the next face for 16-31.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    constexpr uint32_t norm = get_compile_time_arg_val(3);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    // local_e > 0: one-token mode writing only this chip's local_e experts (starting at the uint32 in ep_off) as one
    // bf16 row-major row -- the routing ("sparsity") row the batch-1 MoE kernels read -- instead of the dense tile
    constexpr uint32_t local_e = get_compile_time_arg_val(6);
    constexpr uint32_t cb_buf = 0;
    constexpr auto sel_args = TensorAccessorArgs<7>();
    constexpr auto sc_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto off_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t Et = E / 32;

    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sc_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t t = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto sel = TensorAccessor(sel_args, sel_addr, tile_bytes);
    const auto sc = TensorAccessor(sc_args, sc_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    const uint32_t face0 = (t / 16) * 2;
    const uint32_t row_off = (t % 16) * 64;
    // local buffers: sel row [E], score row [E], output row [E] (fp32 each), sort keys [E] (uint32)
    const uint32_t buf = get_write_ptr(cb_buf);
    const uint32_t sel_l1 = buf, sc_l1 = buf + E * 4, out_l1 = buf + 2 * E * 4;
    for (uint32_t j = 0; j < Et; ++j) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t off = (face0 + h) * 1024 + row_off;
            noc_async_read(sel.get_noc_addr(j) + off, sel_l1 + (j * 32 + h * 16) * 4, 64);
            noc_async_read(sc.get_noc_addr(j) + off, sc_l1 + (j * 32 + h * 16) * 4, 64);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* selv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_l1);
    volatile tt_l1_ptr float* scv = reinterpret_cast<volatile tt_l1_ptr float*>(sc_l1);
    volatile tt_l1_ptr float* outv = reinterpret_cast<volatile tt_l1_ptr float*>(out_l1);

    // one pass, keeping the K best (key, expert) sorted descending; a strictly larger key displaces, so equal keys
    // keep the lower expert id (torch.topk order)
    uint32_t picked[K];
    uint32_t top[K];
    uint32_t count = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t key = order_key(selv[e]);
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
    for (uint32_t e = 0; e < E; ++e) {
        outv[e] = 0.0f;
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        sum += scv[picked[k]];
    }
    union {
        uint32_t u;
        float f;
    } scale;
    scale.u = scale_bits;
    const float mult = norm ? scale.f / sum : scale.f;
    for (uint32_t k = 0; k < K; ++k) {
        outv[picked[k]] = scv[picked[k]] * mult;
    }

    if constexpr (local_e > 0) {
        // this chip's expert offset (one uint32 page of the mesh-sharded ep_off tensor), then the bf16 row
        const auto off_acc = TensorAccessor(off_args, get_common_arg_val<uint32_t>(3));
        const uint32_t off_l1 = out_l1 + E * 4;
        noc_async_read(off_acc.get_noc_addr(0), off_l1, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t e0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_l1)[0];
        const uint32_t row_l1 = (off_l1 + 64 + 63) & ~63u;
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row_l1);
        volatile tt_l1_ptr uint32_t* outu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        for (uint32_t e = 0; e < local_e; ++e) {
            const uint32_t u = outu[e0 + e];
            row[e] = static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        const auto row_acc = TensorAccessor(out_args, out_addr);
        noc_async_write(row_l1, row_acc.get_noc_addr(0), local_e * 2);
        noc_async_write_barrier();
        return;
    }
    for (uint32_t j = 0; j < Et; ++j) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t off = (face0 + h) * 1024 + row_off;
            noc_async_write(out_l1 + (j * 32 + h * 16) * 4, out.get_noc_addr(j) + off, 64);
        }
    }
    noc_async_write_barrier();
}
