// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Payload pattern pieces shared by the data-movement kernels and the compute (TRISC) payload generator, so both
// produce the same per-packet seeds. Freestanding: no dataflow or compute API.

#include <cstdint>

namespace tt::tt_fabric::fabric_tests {

// Per-packet seed step. Payload word i of a packet is seed + i.
inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Payloads on the compute engine are split into Int32 tiles (word w of a payload is word w % 1024 of tile w / 1024), at
// most SFPU_PAYLOAD_MAX_TILES of them, as many as half of dest holds in 32-bit mode. That covers every fabric packet
// (15232 B at most on Blackhole).
inline constexpr uint32_t SFPU_TILE_BYTES = 4096;
inline constexpr uint32_t SFPU_TILE_WORDS = SFPU_TILE_BYTES / 4;
inline constexpr uint32_t SFPU_PAYLOAD_MAX_TILES = 4;

// CB layout of the compute payload kernels. Indices are fixed and the config count is a runtime arg, so one build of
// each kernel serves every core. The CBs on a core are part of every kernel build there, so every sender core declares
// the payload CB and every receiver core the request, flag and all SFPU_VERIFY_MAX_CONFIGS data CBs, as placeholders
// where nothing uses them. Configs 0 .. SFPU_VERIFY_MAX_CONFIGS - 1 of a receiver can be verified on its compute
// engine; data CB c lies over config c's receive ring.
inline constexpr uint32_t SFPU_PAYLOAD_CB_INDEX = 0;
inline constexpr uint32_t SFPU_PAYLOAD_MAX_CONFIGS = 64;
inline constexpr uint32_t SFPU_VERIFY_REQ_CB_INDEX = 0;
inline constexpr uint32_t SFPU_VERIFY_FLAG_CB_INDEX = 1;
inline constexpr uint32_t SFPU_VERIFY_DATA_CB_BASE = 2;
inline constexpr uint32_t SFPU_VERIFY_MAX_CONFIGS = 8;

// Receiver -> compute verify request, one CB page of uint32 words (tt_fabric_test_payload_verify.cpp).
inline constexpr uint32_t SFPU_VERIFY_REQ_OP = 0;
inline constexpr uint32_t SFPU_VERIFY_REQ_CONFIG = 1;
inline constexpr uint32_t SFPU_VERIFY_REQ_SEED = 2;
inline constexpr uint32_t SFPU_VERIFY_REQ_WORDS = 3;
// Where the expected words shift within the packet's slot: segment s covers slot words [end[s - 1], end[s]) and
// expects word w to be seed + w + delta[s]. A contiguous payload is one segment {payload words, 0}; a scatter write
// is one segment per chunk, as its chunks land in the slot out of payload order.
inline constexpr uint32_t SFPU_VERIFY_REQ_SEGMENTS = 4;
inline constexpr uint32_t SFPU_VERIFY_REQ_SEGMENT_END = 5;    // + 2 * s
inline constexpr uint32_t SFPU_VERIFY_REQ_SEGMENT_DELTA = 6;  // + 2 * s
inline constexpr uint32_t SFPU_VERIFY_MAX_SEGMENTS = 4;
inline constexpr uint32_t SFPU_VERIFY_REQ_PAGE_BYTES = 64;
static_assert(SFPU_VERIFY_REQ_SEGMENT_DELTA + 2 * (SFPU_VERIFY_MAX_SEGMENTS - 1) < SFPU_VERIFY_REQ_PAGE_BYTES / 4);
inline constexpr uint32_t SFPU_VERIFY_OP_CHECK = 1;
inline constexpr uint32_t SFPU_VERIFY_OP_EXIT = 2;
// The verdict page's flag lives in its even words 0, 2, ..., 62 (one per SFPU lane).
inline constexpr uint32_t SFPU_VERIFY_FLAG_LANES = 32;

}  // namespace tt::tt_fabric::fabric_tests
