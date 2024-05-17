// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_floor() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat orig = dst_reg[0];

        vFloat res = 0;
        val = sfpi::abs(val);

        v_if(val < 1200001 && val > 120000) {
            v_if(val > 500000) { val = val - 500000; }
            v_endif;
            v_if(val > 250000) { val = val - 250000; }
            v_endif;
            v_if(val > 250000) { val = val - 250000; }
            v_endif;
            v_if(val > 100000) { val = val - 100000; }
            v_endif;
            v_if(val > 100000) { val = val - 100000; }
            v_endif;
        }
        v_endif;

        v_if(val < 120001 && val > 12000) {
            v_if(val > 50000) { val = val - 50000; }
            v_endif;
            v_if(val > 25000) { val = val - 25000; }
            v_endif;
            v_if(val > 25000) { val = val - 20000; }
            v_endif;
            v_if(val > 10000) { val = val - 10000; }
            v_endif;
            v_if(val > 10000) { val = val - 10000; }
            v_endif;
        }
        v_endif;

        v_if(val < 12001 && val > 1200) {
            v_if(val > 5000) { val = val - 5000; }
            v_endif;
            v_if(val > 2500) { val = val - 2500; }
            v_endif;
            v_if(val > 2500) { val = val - 2500; }
            v_endif;
            v_if(val > 1000) { val = val - 1000; }
            v_endif;
            v_if(val > 1000) { val = val - 1000; }
            v_endif;
        }
        v_endif;

        v_if(val < 1201 && val > 120) {
            v_if(val > 500) { val = val - 500; }
            v_endif;
            v_if(val > 250) { val = val - 250; }
            v_endif;
            v_if(val > 250) { val = val - 250; }
            v_endif;
            v_if(val > 100) { val = val - 100; }
            v_endif;
            v_if(val > 100) { val = val - 100; }
            v_endif;
        }
        v_endif;

        v_if(val < 121 && val > 10) {
            v_if(val > 50) { val = val - 50; }
            v_endif;
            v_if(val > 25) { val = val - 25; }
            v_endif;
            v_if(val > 25) { val = val - 25; }
            v_endif;
            v_if(val > 10) { val = val - 10; }
            v_endif;
            v_if(val > 10) { val = val - 10; }
            v_endif;
        }
        v_endif;

        v_if(val < 11) {
            v_if(val > 5) { val = val - 5; }
            v_endif;
            v_if(val > 2) { val = val - 2; }
            v_endif;
            v_if(val > 2) { val = val - 2; }
            v_endif;
            v_if(val > 1) { val = val - 1; }
            v_endif;
        }
        v_endif;

        val = setsgn(val, orig);

        v_if(val > 0) {
            res = orig - val;
            v_if(orig == 1 + res) { res += 1; }
            v_endif;
        }
        v_elseif(val < 0) { res = orig - val - 1; }
        v_endif;
        dst_reg[0] = res;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
