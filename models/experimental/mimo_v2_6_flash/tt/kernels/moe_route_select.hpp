// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode routing on the data-movement RISCs (tt/moe_stream.py): per user, the fp32 top-k over the biased sigmoid
// scores (ties to the lower expert id, sorted descending like ttnn.topk) and the normalized, scaled, bf16-rounded
// and token-masked routing weights from the unbiased scores, read from the user rows (0..rows-1, rows 4, 8 or 16) of
// the [32, E] fp32 tiles.
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace mimo_moe {

constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kFp32FaceBytes = 1024;
constexpr uint32_t kBf16MaskTileBytes = 2048;
constexpr uint32_t kStageRowBytes = 16 * 4;  // one fp32 face row

// Staging area: the user rows of every biased tile (faces 0 and 1: tile columns 0-31), then of every score tile,
// then the user rows of face 0 of the mask tile, then the uint16 expert-id stick ([count, unique expert ids...] with
// the deduplicated stream).
template <uint32_t expert_tiles, uint32_t rows = 4>
struct RouteStage {
    static constexpr uint32_t face_bytes = rows * kStageRowBytes;
    static constexpr uint32_t tile_bytes = 2 * face_bytes;
    static constexpr uint32_t scores = expert_tiles * tile_bytes;
    static constexpr uint32_t mask = 2 * expert_tiles * tile_bytes;
    static constexpr uint32_t stick = mask + 512;
};

union F32Bits {
    float f;
    uint32_t u;
};

inline float bits_to_float(uint32_t u) {
    F32Bits v;
    v.u = u;
    return v.f;
}

inline uint32_t float_to_bits(float f) {
    F32Bits v;
    v.f = f;
    return v.u;
}

// fp32 bits -> bf16 bits, round to nearest even.
inline uint32_t bf16_rne(uint32_t u) { return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16; }

template <uint32_t expert_tiles, bool has_mask, uint32_t rows = 4>
inline void stage_route_rows(uint32_t biased_addr, uint32_t scores_addr, uint32_t mask_addr, uint32_t stage) {
    using Stage = RouteStage<expert_tiles, rows>;
    const InterleavedAddrGen<true> biased = {.bank_base_address = biased_addr, .page_size = kFp32TileBytes};
    const InterleavedAddrGen<true> scores = {.bank_base_address = scores_addr, .page_size = kFp32TileBytes};
    for (uint32_t t = 0; t < expert_tiles; t++) {
        const uint32_t dst = stage + t * Stage::tile_bytes;
        noc_async_read(biased.get_noc_addr(t, 0), dst, Stage::face_bytes);
        noc_async_read(biased.get_noc_addr(t, kFp32FaceBytes), dst + Stage::face_bytes, Stage::face_bytes);
        const uint32_t sdst = dst + Stage::scores;
        noc_async_read(scores.get_noc_addr(t, 0), sdst, Stage::face_bytes);
        noc_async_read(scores.get_noc_addr(t, kFp32FaceBytes), sdst + Stage::face_bytes, Stage::face_bytes);
    }
    if constexpr (has_mask) {
        const InterleavedAddrGen<true> mask = {.bank_base_address = mask_addr, .page_size = kBf16MaskTileBytes};
        noc_async_read(mask.get_noc_addr(0, 0), stage + Stage::mask, rows * 16 * 2);
    }
}

template <
    uint32_t topk,
    uint32_t expert_tiles,
    bool norm,
    bool has_scale,
    uint32_t scale_bits,
    bool has_mask,
    uint32_t rows = 4>
inline void select_user(uint32_t stage, uint32_t u, uint32_t* ids, uint32_t* weight_bits) {
    using Stage = RouteStage<expert_tiles, rows>;
    // Sign-adjusted keys order exactly like the fp32 values; every finite value has a key > 0.
    uint32_t keys[topk];
    for (uint32_t i = 0; i < topk; i++) {
        keys[i] = 0;
        ids[i] = 0;
    }
    // The k-th largest key so far, in a register; a face row's 16 values are loaded before any is compared, so the
    // loads overlap instead of each stalling its compare.
    uint32_t thresh = 0;
    for (uint32_t t = 0; t < expert_tiles; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                stage + t * Stage::tile_bytes + f * Stage::face_bytes + u * kStageRowBytes);
            const uint32_t e0 = t * 32 + f * 16;
            uint32_t vals[16];
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                vals[c] = row[c];
            }
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                const uint32_t b = vals[c];
                const uint32_t key = b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
                if (key <= thresh) {
                    continue;
                }
                uint32_t j = topk - 1;
                while (j > 0 && keys[j - 1] < key) {  // strict: equal keys keep the lower expert id first
                    keys[j] = keys[j - 1];
                    ids[j] = ids[j - 1];
                    j--;
                }
                keys[j] = key;
                ids[j] = e0 + c;
                thresh = keys[topk - 1];
            }
        }
    }

    float s[topk];
    const uint32_t scores = stage + Stage::scores + u * kStageRowBytes;
    for (uint32_t i = 0; i < topk; i++) {
        const uint32_t e = ids[i];
        const uint32_t addr =
            scores + (e >> 5) * Stage::tile_bytes + ((e >> 4) & 1u) * Stage::face_bytes + (e & 15u) * 4;
        s[i] = bits_to_float(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr));
    }
    if constexpr (norm) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < topk; i++) {
            sum += s[i];
        }
        const float denom = sum + 1e-20f;
        for (uint32_t i = 0; i < topk; i++) {
            s[i] = s[i] / denom;
        }
    }
    uint32_t mask_bits = 0;
    if constexpr (has_mask) {
        volatile tt_l1_ptr uint16_t* mask = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + Stage::mask);
        mask_bits = static_cast<uint32_t>(mask[u * 16]) << 16;  // element (u, 0) of face 0
    }
    for (uint32_t i = 0; i < topk; i++) {
        float w = s[i];
        if constexpr (has_scale) {
            w = w * bits_to_float(scale_bits);
        }
        uint32_t wb = bf16_rne(float_to_bits(w)) << 16;
        if constexpr (has_mask) {
            wb = bf16_rne(float_to_bits(bits_to_float(wb) * bits_to_float(mask_bits))) << 16;
        }
        weight_bits[i] = wb;
    }
}

// Whether user u decodes this step: its token-mask element (u, 0) is nonzero (every user without a mask).
template <uint32_t expert_tiles, bool has_mask, uint32_t rows = 4>
inline bool user_active(uint32_t stage, uint32_t u) {
    if constexpr (!has_mask) {
        return true;
    }
    volatile tt_l1_ptr uint16_t* mask =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + RouteStage<expert_tiles, rows>::mask);
    return (mask[u * 16] & 0x7fffu) != 0;
}

// Appends the experts of ``ids`` not yet in ``uniq[0, n)`` (in top-k order); returns the new count. Every RISC that
// builds the list appends the active users in user order, so they all get the same order.
template <uint32_t topk>
inline uint32_t append_unique(uint32_t* uniq, uint32_t n, const uint32_t* ids) {
    for (uint32_t i = 0; i < topk; i++) {
        const uint32_t e = ids[i];
        bool seen = false;
        for (uint32_t j = 0; j < n; j++) {
            seen = seen || uniq[j] == e;
        }
        if (!seen) {
            uniq[n++] = e;
        }
    }
    return n;
}

// The distinct experts of a routing table's active users in user order, each at its first selection (with no active
// user, user 0's), as append_unique lists them; returns the count. Entry u at tbl + u * entry_bytes is [active,
// ids[topk], weight bits[topk]] (FusedStream). A bitmap over the experts replaces append_unique's list scan (n^2
// compares: ~9 us for 37 experts on these RISCs).
template <uint32_t topk, uint32_t users, uint32_t experts>
inline uint32_t table_experts(uint32_t tbl, uint32_t entry_bytes, uint32_t* uniq) {
    static_assert(experts % 32 == 0);
    uint32_t seen[experts / 32];
    for (uint32_t w = 0; w < experts / 32; w++) {
        seen[w] = 0;
    }
    uint32_t n = 0;
    for (uint32_t u = 0; u < users; u++) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl + u * entry_bytes);
        if (e[0] == 0) {
            continue;
        }
        for (uint32_t i = 0; i < topk; i++) {
            const uint32_t id = e[1 + i];
            const uint32_t bit = 1u << (id & 31u);
            if ((seen[id >> 5] & bit) == 0) {
                seen[id >> 5] |= bit;
                uniq[n++] = id;
            }
        }
    }
    if (n == 0) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl);
        for (uint32_t i = 0; i < topk; i++) {
            const uint32_t id = e[1 + i];
            const uint32_t bit = 1u << (id & 31u);
            if ((seen[id >> 5] & bit) == 0) {
                seen[id >> 5] |= bit;
                uniq[n++] = id;
            }
        }
    }
    return n;
}

// table_experts per row half (FusedStream with two halves: users h * per_half .. (h + 1) * per_half - 1): half 0's
// distinct experts, then half 1's (an expert both halves selected is listed in each); with no active user at all,
// user 0's, in half 0. ``counts[h]``: half h's count. Returns the total.
template <uint32_t topk, uint32_t per_half, uint32_t halves, uint32_t experts>
inline uint32_t table_experts_halves(uint32_t tbl, uint32_t entry_bytes, uint32_t* uniq, uint32_t* counts) {
    static_assert(experts % 32 == 0);
    uint32_t n = 0;
    for (uint32_t h = 0; h < halves; h++) {
        uint32_t seen[experts / 32];
        for (uint32_t w = 0; w < experts / 32; w++) {
            seen[w] = 0;
        }
        const uint32_t start = n;
        for (uint32_t u = h * per_half; u < (h + 1) * per_half; u++) {
            volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl + u * entry_bytes);
            if (e[0] == 0) {
                continue;
            }
            for (uint32_t i = 0; i < topk; i++) {
                const uint32_t id = e[1 + i];
                const uint32_t bit = 1u << (id & 31u);
                if ((seen[id >> 5] & bit) == 0) {
                    seen[id >> 5] |= bit;
                    uniq[n++] = id;
                }
            }
        }
        counts[h] = n - start;
    }
    if (n == 0) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl);
        uint32_t seen[experts / 32];
        for (uint32_t w = 0; w < experts / 32; w++) {
            seen[w] = 0;
        }
        for (uint32_t i = 0; i < topk; i++) {
            const uint32_t id = e[1 + i];
            const uint32_t bit = 1u << (id & 31u);
            if ((seen[id >> 5] & bit) == 0) {
                seen[id >> 5] |= bit;
                uniq[n++] = id;
            }
        }
        counts[0] = n;
    }
    return n;
}

}  // namespace mimo_moe
