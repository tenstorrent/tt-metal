// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the reader, compute and writer kernels of the fused TensorOcean kernel (v2).
// The host prepends a generated header (namespace P) with the plan constants.
#pragma once
#include <cstdint>
constexpr uint32_t CHB = P::CH * 4;  // chunk bytes (128 fp32)
// level-pair tiles per step in CB_LVL (reader: pairs 0, 4, 8, ...) and CB_LVL2 (writer: pairs 2, 6, ...), sized for a
// full pass of LP levels. Every core reserves and pushes this many per step even when it has fewer levels (compute
// pops the unused rest), so a reservation never straddles the end of the CB ring (it would overrun the next CB).
constexpr uint32_t LVL_STEP_TILES = 3 * ((P::LP / 2 + 1) / 2), LVL2_STEP_TILES = 3 * ((P::LP / 2) / 2);
constexpr uint32_t TILEB = 4096;
// CB ids
constexpr uint32_t CB_STAT = 0, CB_LVL = 1, CB_OI = 2, CB_TOK = 3, CB_PLANES = 4, CB_FBUF = 5, CB_STAGE = 6,
                   CB_TOK2 = 7, CB_MCW = 8, CB_PROG = 9, CB_FMK = 10, CB_SSTAGE = 11, CB_RARR = 12, CB_INV = 13,
                   CB_OI2 = 14, CB_INV2 = 15;
constexpr uint32_t CB_F = 16, CB_OUT = 17;
constexpr uint32_t CB_ADDR = 18, CB_TOK3 = 19, CB_TOK4 = 20;
constexpr uint32_t CB_LVL2 = 21, CB_FMK2 = 22, CB_TOK5 = 23;
constexpr uint32_t CB_TOK6 = 24;  // v26+: per-level plane-loaded tokens (reader: even levels on CB_TOK4, writer: odd on
                                  // CB_TOK6)   // v24+: odd level pairs assembled by the writer   // v22+: address-only
                                  // CB for unpack/pack at any L1 address; shift tokens
// plane slot (level-in-core li, plane p, shift index k): k = 0 unshifted, k >= 1 the k-th shifted copy of plane p
constexpr uint32_t plane_slot_bytes = P::CELL_LEN * 4;
inline uint32_t plane_addr(uint32_t base, uint32_t li, uint32_t p, uint32_t k) {
    return base + (li * P::NSLOT + P::PBASE[p] + k) * plane_slot_bytes;
}
// F buffers: slot (g, li); shifted copies after the 6 groups: slot (6 + fs_index, li)
constexpr uint32_t f_slot_bytes = P::F_ALLOC * 4;
inline uint32_t f_addr(uint32_t base, uint32_t slot, uint32_t li) { return base + (slot * P::LP + li) * f_slot_bytes; }

// staging slot for (stat 24 chunks + f/mask of LC levels) prefetched from DRAM, double buffered
constexpr uint32_t STAGE_STAT_B = 24 * CHB;
constexpr uint32_t STAGE_SLOT_B = STAGE_STAT_B + P::LP * 2 * CHB;
// shifted plane copies: entries with k % 2 == 0 are made by the reader, k % 2 == 1 by the writer
inline void shift_copy(uint32_t src_addr, uint32_t dst_addr, uint32_t rho, uint32_t words) {
    const uint32_t* s = (const uint32_t*)src_addr + rho;
    uint32_t* d = (uint32_t*)dst_addr;
    uint32_t n = words - rho;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
        d[0] = a0;
        d[1] = a1;
        d[2] = a2;
        d[3] = a3;
        d[4] = a4;
        d[5] = a5;
        d[6] = a6;
        d[7] = a7;
        s += 8;
        d += 8;
    }
    for (; i < n; ++i) {
        *d++ = *s++;
    }
    for (; i < words; ++i) {
        *d++ = 0;
    }
}

// partial shifted copy: dst[i] = src[i + rho] for i in [a, e), zero beyond the source end
inline void shift_range(uint32_t src_addr, uint32_t dst_addr, uint32_t rho, uint32_t a, uint32_t e, uint32_t words) {
    const uint32_t* s = (const uint32_t*)src_addr + a + rho;
    uint32_t* d = (uint32_t*)dst_addr + a;
    uint32_t lim = e < words - rho ? e : words - rho;
    uint32_t i = a;
    for (; i + 8 <= lim; i += 8) {
        uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
        d[0] = a0;
        d[1] = a1;
        d[2] = a2;
        d[3] = a3;
        d[4] = a4;
        d[5] = a5;
        d[6] = a6;
        d[7] = a7;
        s += 8;
        d += 8;
    }
    for (; i < lim; ++i) {
        *d++ = *s++;
    }
    for (; i < e; ++i) {
        *d++ = 0;
    }
}
constexpr uint32_t PASS_TAG = 1u << 20;  // shift-progress words are tagged with the pass index

// block-major F buffers (v8+): array s, block b, level li -> 128 floats; LPT levels per block (tile multiple)
#define fb_addr(base, s, b, li) ((base) + (((s) * P::FBLK + (b)) * P::LPT + (li)) * CHB)

// F ring (v10+): array s, level li, block b -> slot b % FR of 128 floats
#define fr_addr(base, s, li, b) ((base) + ((((s) * P::LP + (li)) * P::FR) + ((b) % P::FR)) * CHB)

// v16+: the output stage lags one block: at the end of block b, finish the output blocks that were ready
// after block b-1 (everything at the last block)
#define ob_target(b) ((uint32_t)((b) + 1 == P::NBLK ? P::NOBLK : ((b) ? P::OB_READY[(b) - 1] : 0)))

// v23+: F ring with a mirror slot: slot FR repeats slot 0, so a 128-item window starting anywhere in the ring is
// contiguous (one fixed-size read). Block b lives in slot b % FR (and slot FR too when b % FR == 0).
#define frm_addr(base, s, li, slot) ((base) + ((((s) * P::LP + (li)) * (P::FR + 1)) + (slot)) * CHB)
