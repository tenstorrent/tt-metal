// Megakernel attention block, MKA1: the verify-SDPA phase (ATTN role) on all 110 cores, with STREAM-core weight
// prefetch running concurrently. CB / semaphore ids follow DESIGN_megakernel.md §6 interface spec v0:
// ATTN CBs 9-12 (Q, K, V, O), block scratch 24-31, ATTN semaphores 16-19.
#pragma once
#include <stdint.h>
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12;
constexpr uint32_t CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28, CB_PF = 29;
constexpr uint32_t DH = 8;        // head_dim 256 = 8 tiles
constexpr uint32_t NO = DH + 1;   // O_aug = [P·V | P·1]: 8 value tiles + the denominator tile
constexpr uint32_t QT = 32;       // Q of all 4 own users: 4 × 8 tiles (rows (h, j) = 4h + j, 24 used)
// semaphores
constexpr uint32_t SEM_QFLAG = 12, SEM_LVL0 = 13, SEM_LVL1 = 14, SEM_DONE = 15;  // spec v0 said 16-19, but ids are 0-15 only (standalone MKA1 uses 12-15)
constexpr uint32_t FANOUT = 6;    // tree fanout: groups of <= 36 cores reduce in 2 levels
