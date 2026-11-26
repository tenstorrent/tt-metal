// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2 on the 2x4 (tt/spa_union.py, two stages), COLUMN lane NCRISC = stage A: waits for the
// column partner's row OR (its column lane's BRISC writes it into this lane's OR slot 1 and bumps ARR_BM + its
// position; generation GEN + 1 of this lane's sync words), then walks it over the words THIS chip owns (w % N == k) in
// pool order: pool number i of that walk goes to the partner's relay rows 4 i .. 4 i + 3 (the partner's row lane reads
// them back in its stage B with the same index: the rank among its row-OR pools in this chip's words). Per pool its 4
// local latent rows -> CB_STAGE, (slot i, the partner's position on the column) -> CB_META for this lane's BRISC
// (spa_us.cpp); then an end marker. Compile-time: CB_STAGE, CB_META, WORDS, then the local store's TensorAccessorArgs.
// Runtime: sync (this lane's), OR slots, store, p (position on the column), N, k, cap (relay rows / 4).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_stage.hpp"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t WORDS = get_compile_time_arg_val(2);
constexpr uint32_t CB_ORS =
    get_compile_time_arg_val(3);  // the OR slots (this lane's CB: slot 0 this row, 1 the partner's)
constexpr uint32_t RB = 1024;
constexpr uint32_t UNIT = 128;
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t GEN = 0, ARR_BM = 1;

void kernel_main() {
    constexpr auto lat_args = TensorAccessorArgs<4>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    a++;  // (the OR slots: CB_ORS)
    cb_reserve_back(CB_ORS, 1);
    const uint32_t ors = get_write_ptr(CB_ORS);
    const uint32_t lat_addr = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t nall = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t cap = get_arg_val<uint32_t>(a++);
    const auto lat = TensorAccessor(lat_args, lat_addr);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    invalidate_l1_cache();
    const uint32_t g = sw[GEN] + 1;
    const uint32_t q = 1 - p;  // the column partner (a column of 2)
    while (sw[ARR_BM + q] < g) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* other = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ors + WORDS * 4);  // slot 1
    uint32_t i = 0;
    spa::Stager<CB_STAGE, CB_META> st;
    for (uint32_t w = k; w < WORDS; w += nall) {
        const uint32_t x = other[w];
        if (x == 0) {
            continue;
        }
        for (uint32_t b = 0; b < 32; ++b) {
            if (!((x >> b) & 1u)) {
                continue;
            }
            if (i < cap) {
                st.add(lat, (w / nall) * UNIT + 4 * b, i, q);
            }
            ++i;
        }
    }
    sw[1 + 2 * 2] = i > cap ? i : 0;  // STATUS (a column line of 2): the relay set exceeded the relay rows
    st.end();
}
