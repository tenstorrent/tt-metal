// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_log_fused.h"

namespace ckernel::sfpu::bf16 {
// log_B(x) = e*scale + P(m - 1) for x = 2^e * m, m in [1, 2), with P(u) = u*H(u) in base B.
// The last Horner step adds e*scale instead of P's zero constant term: y = H(u)*u + e*scale.
// Every constant lives in L0 and L4..L7 for the callback, so the kernel leaves the
// programmable constants to whatever stock init owns them; the row keeps to L1..L3.
template <typename Config>
inline void init_log_fused() {
    // The shared whole-tile callback advances one vector per store.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
