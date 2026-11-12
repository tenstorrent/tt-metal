// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// L1 cell probe: a March C- pass over a private L1 region (up-write P, up-read P/write ~P, down-read ~P/
// write P, down-read P). The pattern carries the word address so an addressing fault cannot hide behind
// equal data, and the seed changes every pass so every bit toggles both ways. No NoC traffic.
//
// Runtime args: base_addr, bytes, n_passes, result_addr, core_id

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "nd_stage_result.h"

namespace {

FORCE_INLINE uint32_t pattern(uint32_t i, uint32_t seed) { return (i * 0x01000193u) ^ seed; }

}  // namespace

void kernel_main() {
    const uint32_t base_addr = get_arg_val<uint32_t>(0);
    const uint32_t bytes = get_arg_val<uint32_t>(1);
    const uint32_t n_passes = get_arg_val<uint32_t>(2);
    const uint32_t result_addr = get_arg_val<uint32_t>(3);
    const uint32_t core_id = get_arg_val<uint32_t>(4);

    nd_stage::Result res(result_addr, nd_stage::MAGIC_L1, core_id);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr);
    const uint32_t n = bytes / sizeof(uint32_t);

    for (uint32_t pass = 0; pass < n_passes; ++pass) {
        const uint32_t seed = (pass + 1) * 0x9E3779B9u;
        bool bad = false;
        auto check = [&](uint32_t i, uint32_t want) {
            const uint32_t got = m[i];
            if (got != want) {
                bad = true;
                res.record_mismatch<0>(pass, i, got, want, 0);
            }
        };

        for (uint32_t i = 0; i < n; ++i) {
            m[i] = pattern(i, seed);
        }
        // Each phase reads what the previous phase wrote; drop cached lines so the reads come from L1.
        invalidate_l1_cache();
        for (uint32_t i = 0; i < n; ++i) {
            check(i, pattern(i, seed));
            m[i] = ~pattern(i, seed);
        }
        invalidate_l1_cache();
        for (uint32_t i = n; i-- > 0;) {
            check(i, ~pattern(i, seed));
            m[i] = pattern(i, seed);
        }
        invalidate_l1_cache();
        for (uint32_t i = n; i-- > 0;) {
            check(i, pattern(i, seed));
        }
        res.end_iteration(true, bad, pass);
    }
    res.finish();
}
