// GDN_STEP_U1 lever 1 test producer (tg_*): one core per value head computes the T gate pairs with gdn_spec_step's
// SFPU gate sequence (lz_sfpu_gates) and builds the step writer's 8 WY coefficient tiles (C filled with the replay
// decay) plus a gate page (dec_t at (r0 + t, 0), beta_t at (r0 + t, 1)) -> `prefixed` tiles h*S + 2Kt + 2Vt + [0, 9).
#pragma once
#include <cstdint>
constexpr uint32_t TG_AB = 0;     // raw a|b tile (bf16)
constexpr uint32_t TG_GT = 16;    // [dec over a's tile | beta over b's tile] (fp32)
constexpr uint32_t TG_COEF = 24;  // 9 fp32 tiles: NGB G E BETA NBE EYE W C | GD
constexpr uint32_t TG_GP = 25;    // the previous stash's decay page (fp32)
inline uint32_t tg_elem(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}
