// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn routing pre-pass (proxy scores -> threshold -> chunk mask / bias), reader (NCRISC).
//
// Work unit = (head h, Q-chunk tile row it): the routing of local Q chunks [32*it, 32*it+32) of head h against the
// pooled K blocks of every shard. Per core, once: the ones-row tile, the Q-chunk selector (qsel), the block->chunk
// selector (ksel) and the 32 row-broadcast selectors (erow). Per unit: the head's whole pooled K [NBG, D] in D-major
// tile order (position n*NBGT + c holds tile (c, n), so compute can use it as a transposed matmul operand) and the
// head's pooled Q [NBP, D]; then per column shard sp: the constant OR-mask tiles of this Q-chunk tile row against
// shard sp's blocks, and the column constants (full / valid / length-bias rows) of shard sp.
//
// Compile-time args: 0 H, 1 S, 2 NBPT, 3 DT, 4 NQCT, 5 NKCLT, 6 NQC, 7 NKC, 8 NKCP, 9 MODE, 10 EMIT_BIAS, 11 TAU,
// 12 EPS2, 13 NEG, 14 INV_NV, 15 TQ, then TensorAccessorArgs of q_bar, kc, const_or, colconst, qsel, ksel, erow, ones.
// Common runtime args: 0..7 the buffer addresses in the same order. Per-core: none.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_kc = 0, cb_q = 1, cb_ones = 2, cb_qsel = 3, cb_ksel = 4, cb_erow = 5, cb_constor = 6, cb_colc = 7;
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NQCT = get_compile_time_arg_val(4);
    constexpr uint32_t NKCLT = get_compile_time_arg_val(5);
    constexpr uint32_t NBGT = S * NBPT;
    constexpr auto qb_args = TensorAccessorArgs<20>();
    constexpr auto kc_args = TensorAccessorArgs<qb_args.next_compile_time_args_offset()>();
    constexpr auto co_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto cc_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr auto qs_args = TensorAccessorArgs<cc_args.next_compile_time_args_offset()>();
    constexpr auto ks_args = TensorAccessorArgs<qs_args.next_compile_time_args_offset()>();
    constexpr auto er_args = TensorAccessorArgs<ks_args.next_compile_time_args_offset()>();
    constexpr auto on_args = TensorAccessorArgs<er_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(17);
    constexpr uint32_t NY = get_compile_time_arg_val(18);
    constexpr uint32_t UNITS = get_compile_time_arg_val(19);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto qb = TensorAccessor(qb_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(kc_args, get_common_arg_val<uint32_t>(1));
    const auto co = TensorAccessor(co_args, get_common_arg_val<uint32_t>(2));
    const auto cc = TensorAccessor(cc_args, get_common_arg_val<uint32_t>(3));
    const auto qs = TensorAccessor(qs_args, get_common_arg_val<uint32_t>(4));
    const auto ks = TensorAccessor(ks_args, get_common_arg_val<uint32_t>(5));
    const auto er = TensorAccessor(er_args, get_common_arg_val<uint32_t>(6));
    const auto on = TensorAccessor(on_args, get_common_arg_val<uint32_t>(7));

    Noc noc;
    auto read_pages = [&](auto& acc, uint32_t cb_id, uint32_t first, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(acc, cb, TILE_BYTES, {.page_id = first + i}, {.offset_bytes = i * TILE_BYTES});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    read_pages(on, cb_ones, 0, 1);
    read_pages(qs, cb_qsel, 0, NQCT * NBPT);
    read_pages(ks, cb_ksel, 0, NBPT * NKCLT);
    read_pages(er, cb_erow, 0, 32);

    CircularBuffer cbkc(cb_kc), cbco(cb_constor), cbcc(cb_colc);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NQCT;
        const uint32_t it = u % NQCT;

        // Pooled K of the head, D-major tile order.
        cbkc.reserve_back(NBGT * DT);
        {
            uint32_t issued = 0;
            for (uint32_t n = 0; n < DT; ++n) {
                for (uint32_t cg = 0; cg < NBGT; ++cg) {
                    noc.async_read(
                        kc,
                        cbkc,
                        TILE_BYTES,
                        {.page_id = (h * NBGT + cg) * DT + n},
                        {.offset_bytes = (n * NBGT + cg) * TILE_BYTES});
                    if (++issued == 32) {
                        noc.async_read_barrier();
                        issued = 0;
                    }
                }
            }
            noc.async_read_barrier();
        }
        cbkc.push_back(NBGT * DT);

        read_pages(qb, cb_q, h * NBPT * DT, NBPT * DT);

        for (uint32_t sp = 0; sp < S; ++sp) {
            cbco.reserve_back(NBPT);
            for (uint32_t c = 0; c < NBPT; ++c) {
                noc.async_read(
                    co, cbco, TILE_BYTES, {.page_id = it * NBGT + sp * NBPT + c}, {.offset_bytes = c * TILE_BYTES});
            }
            cbcc.reserve_back(3 * NBPT);
            for (uint32_t kind = 0; kind < 3; ++kind) {
                for (uint32_t j = 0; j < NBPT; ++j) {
                    noc.async_read(
                        cc,
                        cbcc,
                        TILE_BYTES,
                        {.page_id = (kind * S + sp) * NBPT + j},
                        {.offset_bytes = (kind * NBPT + j) * TILE_BYTES});
                }
            }
            noc.async_read_barrier();
            cbco.push_back(NBPT);
            cbcc.push_back(3 * NBPT);
        }
    }
}
