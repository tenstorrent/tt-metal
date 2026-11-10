// GDN_STEP_U1 op-layout prefix (tp_*): CB ids. One core per job: kind 0 = the conv + SiLU + one-hot placement of one
// q / k / v chunk (gdn_spec_step K4 phase A's op sequence on the op's own tensors), kind 1 = silu(z) of one z tile
// (c_silu's sequence). Output: fp32 tiles of the `prefixed` tensor (kn | qn | vc | zs per value head; q / k raw here,
// normalized in place by the L2 roots).
#pragma once
#include <cstdint>
constexpr uint32_t TP_ST = 0;    // [window tile | raw qkv tile] (bf16, whole tiles)
constexpr uint32_t TP_TAP = 3;   // the chunk's tap tile (bf16, tap j in row j: bcast_row_idx j)
constexpr uint32_t TP_SH = 4;    // K shifted operands (bf16): SH_j row t = W row t + j, other rows 0
constexpr uint32_t TP_P = 5;     // placement one-hot (bf16): 1.0 at (t, t), t < T
constexpr uint32_t TP_Z = 6;     // z tile (bf16)
constexpr uint32_t TP_W = 7;     // rebuilt window tile (bf16): rows 0..Lw_w-1 -> win[1 - par]
constexpr uint32_t TP_CV = 24;   // conv + SiLU tile (fp32)
constexpr uint32_t TP_OUT = 16;  // output tile (fp32)

inline uint32_t tp_elem(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}
