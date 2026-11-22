// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Block residual reader. One unit = (head h, local Q tile row r): its q_bar tile row, then head h's gathered block
// means kc with their log-length tiles and the unit's 0/1 mask tiles in chunks of KCH key tiles, then vc in the same
// chunks. The constant tiles once per core.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_q = 0, cb_kc = 1, cb_ka = 2, cb_mk = 3, cb_vc = 4, cb_tiles = 5;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t N_TILES = 5;
}  // namespace

void kernel_main() {
    constexpr uint32_t NBPT = get_compile_time_arg_val(1);
    constexpr uint32_t NBGT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NX = get_compile_time_arg_val(4);
    constexpr uint32_t NY = get_compile_time_arg_val(5);
    constexpr uint32_t UNITS = get_compile_time_arg_val(6);
    constexpr uint32_t KCH = get_compile_time_arg_val(7);
    static_assert(NBGT % KCH == 0, "key tiles come in whole chunks");
    constexpr auto q_args = TensorAccessorArgs<12>();
    constexpr auto kc_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto mk_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto ka_args = TensorAccessorArgs<mk_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<ka_args.next_compile_time_args_offset()>();

    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(kc_args, get_common_arg_val<uint32_t>(1));
    const auto vc = TensorAccessor(vc_args, get_common_arg_val<uint32_t>(2));
    const auto mk = TensorAccessor(mk_args, get_common_arg_val<uint32_t>(3));
    const auto ka = TensorAccessor(ka_args, get_common_arg_val<uint32_t>(4));
    const auto tl = TensorAccessor(t_args, get_common_arg_val<uint32_t>(5));

    Noc noc;
    CircularBuffer cbq(cb_q), cbkc(cb_kc), cbka(cb_ka), cbmk(cb_mk), cbvc(cb_vc), cbt(cb_tiles);
    cbt.reserve_back(N_TILES);
    for (uint32_t i = 0; i < N_TILES; ++i) {
        noc.async_read(tl, cbt, TILE_BYTES, {.page_id = i}, {.offset_bytes = i * TILE_BYTES});
    }
    noc.async_read_barrier();
    cbt.push_back(N_TILES);

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NBPT, r = u % NBPT;
        cbq.reserve_back(DT);
        for (uint32_t n = 0; n < DT; ++n) {
            noc.async_read(q, cbq, TILE_BYTES, {.page_id = (h * NBPT + r) * DT + n}, {.offset_bytes = n * TILE_BYTES});
        }
        noc.async_read_barrier();
        cbq.push_back(DT);

        for (uint32_t c0 = 0; c0 < NBGT; c0 += KCH) {
            cbkc.reserve_back(KCH * DT);
            cbka.reserve_back(KCH);
            cbmk.reserve_back(KCH);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                const uint32_t c = c0 + cc;
                for (uint32_t n = 0; n < DT; ++n) {
                    noc.async_read(
                        kc,
                        cbkc,
                        TILE_BYTES,
                        {.page_id = (h * NBGT + c) * DT + n},
                        {.offset_bytes = (cc * DT + n) * TILE_BYTES});
                }
                noc.async_read(ka, cbka, TILE_BYTES, {.page_id = c}, {.offset_bytes = cc * TILE_BYTES});
                noc.async_read(
                    mk, cbmk, TILE_BYTES, {.page_id = (h * NBPT + r) * NBGT + c}, {.offset_bytes = cc * TILE_BYTES});
            }
            noc.async_read_barrier();
            cbkc.push_back(KCH * DT);
            cbka.push_back(KCH);
            cbmk.push_back(KCH);
        }
        for (uint32_t c0 = 0; c0 < NBGT; c0 += KCH) {
            cbvc.reserve_back(KCH * DT);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                for (uint32_t n = 0; n < DT; ++n) {
                    noc.async_read(
                        vc,
                        cbvc,
                        TILE_BYTES,
                        {.page_id = (h * NBGT + c0 + cc) * DT + n},
                        {.offset_bytes = (cc * DT + n) * TILE_BYTES});
                }
            }
            noc.async_read_barrier();
            cbvc.push_back(KCH * DT);
        }
    }
}
