// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_log.h"
#include "sfpi.h"

namespace ckernel::sfpu {

//
// log1p(x) = log(1 + x), numerically stable for |x| << 1.
//
// See wormhole_b0 version for full commentary.
//

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    sfpi::vFloat u = sfpi::vConst1 + x;
    sfpi::vFloat log_u = _calculate_log_body_no_init_(u);

    sfpi::vFloat u_m1 = u - sfpi::vConst1;
    sfpi::vFloat diff = x - u_m1;

    sfpi::vFloat inv_u;
    if constexpr (APPROXIMATION_MODE) {
        inv_u = sfpi::approx_recip(u);
    } else {
        sfpi::vFloat r = sfpi::approx_recip(u);
        r = r * (sfpi::vFloat(2.0f) - u * r);
        r = r * (sfpi::vFloat(2.0f) - u * r);
        inv_u = r;
    }

    sfpi::vFloat correction = diff * inv_u;
    sfpi::vFloat result = log_u + correction;

    v_if(x == sfpi::vConst0) {
        result = sfpi::vConst0;
    }
    v_endif;

    return result;
}

}  // namespace ckernel::sfpu
