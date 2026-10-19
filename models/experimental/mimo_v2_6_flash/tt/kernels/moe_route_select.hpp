// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode routing on the data-movement RISCs (tt/moe_stream.py): per user, the fp32 top-k over the biased sigmoid
// scores (ties to the lower expert id, sorted descending like ttnn.topk) and the normalized, scaled, bf16-rounded
// and token-masked routing weights from the unbiased scores, read from rows 0-3 of the [32, E] fp32 tiles.
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace mimo_moe {

constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kFp32FaceBytes = 1024;
constexpr uint32_t kBf16MaskTileBytes = 2048;
constexpr uint32_t kStagedRows = 4;
constexpr uint32_t kStageRowBytes = 16 * 4;                         // one fp32 face row
constexpr uint32_t kStageFaceBytes = kStagedRows * kStageRowBytes;  // rows 0-3 of one fp32 face: 256 B
constexpr uint32_t kStageTileBytes = 2 * kStageFaceBytes;           // faces 0 and 1 (tile columns 0-31)

// Staging area: rows 0-3 of every biased tile, then of every score tile, then rows 0-3 of face 0 of the mask
// tile, then the uint16 expert-id stick.
template <uint32_t expert_tiles>
struct RouteStage {
    static constexpr uint32_t scores = expert_tiles * kStageTileBytes;
    static constexpr uint32_t mask = 2 * expert_tiles * kStageTileBytes;
    static constexpr uint32_t stick = mask + 256;
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

template <uint32_t expert_tiles, bool has_mask>
inline void stage_route_rows(uint32_t biased_addr, uint32_t scores_addr, uint32_t mask_addr, uint32_t stage) {
    const InterleavedAddrGen<true> biased = {.bank_base_address = biased_addr, .page_size = kFp32TileBytes};
    const InterleavedAddrGen<true> scores = {.bank_base_address = scores_addr, .page_size = kFp32TileBytes};
    for (uint32_t t = 0; t < expert_tiles; t++) {
        const uint32_t dst = stage + t * kStageTileBytes;
        noc_async_read(biased.get_noc_addr(t, 0), dst, kStageFaceBytes);
        noc_async_read(biased.get_noc_addr(t, kFp32FaceBytes), dst + kStageFaceBytes, kStageFaceBytes);
        const uint32_t sdst = dst + RouteStage<expert_tiles>::scores;
        noc_async_read(scores.get_noc_addr(t, 0), sdst, kStageFaceBytes);
        noc_async_read(scores.get_noc_addr(t, kFp32FaceBytes), sdst + kStageFaceBytes, kStageFaceBytes);
    }
    if constexpr (has_mask) {
        const InterleavedAddrGen<true> mask = {.bank_base_address = mask_addr, .page_size = kBf16MaskTileBytes};
        noc_async_read(mask.get_noc_addr(0, 0), stage + RouteStage<expert_tiles>::mask, kStagedRows * 16 * 2);
    }
}

template <uint32_t topk, uint32_t expert_tiles, bool norm, bool has_scale, uint32_t scale_bits, bool has_mask>
inline void select_user(uint32_t stage, uint32_t u, uint32_t* ids, uint32_t* weight_bits) {
    // Sign-adjusted keys order exactly like the fp32 values; every finite value has a key > 0.
    uint32_t keys[topk];
    for (uint32_t i = 0; i < topk; i++) {
        keys[i] = 0;
        ids[i] = 0;
    }
    for (uint32_t t = 0; t < expert_tiles; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                stage + t * kStageTileBytes + f * kStageFaceBytes + u * kStageRowBytes);
            const uint32_t e0 = t * 32 + f * 16;
            for (uint32_t c = 0; c < 16; c++) {
                const uint32_t b = row[c];
                const uint32_t key = b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
                if (key <= keys[topk - 1]) {
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
            }
        }
    }

    float s[topk];
    const uint32_t scores = stage + RouteStage<expert_tiles>::scores + u * kStageRowBytes;
    for (uint32_t i = 0; i < topk; i++) {
        const uint32_t e = ids[i];
        const uint32_t addr = scores + (e >> 5) * kStageTileBytes + ((e >> 4) & 1u) * kStageFaceBytes + (e & 15u) * 4;
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
        volatile tt_l1_ptr uint16_t* mask =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + RouteStage<expert_tiles>::mask);
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

}  // namespace mimo_moe
