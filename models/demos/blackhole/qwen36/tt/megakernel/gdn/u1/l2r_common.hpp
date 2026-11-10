// GDN_STEP_U1 prefix L2-norm root (one core per (head, q|k)): CB ids. Formats mirror gdn_spec_step's dfbs:
// src / tmp / stats / scratch / inv / dst fp32, scaler fp32 (1.0), eps_l2 bf16 col-scalar, mask_T bf16.
#pragma once
#include <cstdint>
constexpr uint32_t L_SRC = 0;      // N conv tiles (fp32)
constexpr uint32_t L_TMP = 1;      // N squared tiles (fp32)
constexpr uint32_t L_SCALER = 2;   // reduce scaler (1.0)
constexpr uint32_t L_STATS = 3;    // row sums of squares (column 0)
constexpr uint32_t L_EPS = 4;      // eps_l2 col-scalar (bf16)
constexpr uint32_t L_MASK = 5;     // mask_T: 1.0 at (t, 0), t < T (bf16)
constexpr uint32_t L_SCRATCH = 6;  // rsqrt(stats + eps) * post
constexpr uint32_t L_INV = 7;      // * mask
constexpr uint32_t L_DST = 16;     // N normalized tiles (fp32)
