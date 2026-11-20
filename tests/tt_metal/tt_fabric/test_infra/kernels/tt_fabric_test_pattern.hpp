// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Limits and payload pattern pieces shared by the host, the data-movement kernels and the compute (TRISC) payload
// kernels, so all of them use the same per-core limits and per-packet seeds. Freestanding: no dataflow or compute API.

#include <cstdint>

namespace tt::tt_fabric::fabric_tests {

// Most traffic configs a sender core runs from its hot path, which keeps per-config state on the BRISC stack. Stack
// budget = BRISC local memory - firmware data - kernel globals. The sender's kernel_main frame is a fixed part (mostly
// the local_connections copy) + 52 B per config (SenderHotState and a config pointer), and the functions kernel_main
// calls need ~256 B more.
// - Wormhole (4 KB local memory): ~2048 B for kernel_main and its callees and a 624 B fixed part, so ~22 configs can
//   fit (2048 + 624 + (22 * 52) = 3816), 20 keeps a margin.
// - Blackhole (8 KB local memory, ~3.2 KB of it firmware data): ~4.7 KB for kernel_main and its callees and a 736 B
//   fixed part (five connections), so ~72 configs can fit (736 + (72 * 52) + 256 = 4736), 39 is the most validated
//   (2768 B frame on a Blackhole Galaxy).
// An overflow can build fine and overwrite firmware data at runtime, so re-measure when SenderHotState, the connection
// array or the hot loop grows: build the sender at a few config counts and read kernel_main's frame from the
// `addi sp,sp,-N` lines of `objdump -d brisc.elf` (a frame over 2 KB takes two). A core with more configs sends from
// its L1 state instead (the slower path). The host splits senders to stay within this (split_senders_to_fit_core_limits
// in tt_fabric_test_allocator.hpp).
inline constexpr uint32_t HOT_STATE_MAX_CONFIGS_WORMHOLE = 20;
inline constexpr uint32_t HOT_STATE_MAX_CONFIGS_BLACKHOLE = 39;

// Buffers in every channel of a test mux: the host builds each mux with this many per channel, and each client sizes
// its mux connection (FabricConnectionArray in tt_fabric_test_kernels_utils.hpp) to match.
inline constexpr uint8_t MUX_BUFFERS_PER_CHANNEL = 8;

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
// engine; data CB c lies over config c's receive ring. The verifier finds a config's data CB from the request at
// runtime, so the cap costs one CB index per config and nothing else on the device.
inline constexpr uint32_t SFPU_PAYLOAD_CB_INDEX = 0;
inline constexpr uint32_t SFPU_PAYLOAD_MAX_CONFIGS = 64;
inline constexpr uint32_t SFPU_VERIFY_REQ_CB_INDEX = 0;
inline constexpr uint32_t SFPU_VERIFY_FLAG_CB_INDEX = 1;
inline constexpr uint32_t SFPU_VERIFY_DATA_CB_BASE = 2;
inline constexpr uint32_t SFPU_VERIFY_MAX_CONFIGS = 16;
// Wormhole has 32 CB indices (Blackhole 64).
static_assert(SFPU_VERIFY_DATA_CB_BASE + SFPU_VERIFY_MAX_CONFIGS <= 32);

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
// Packets a receiver may have handed to its compute verifier without a verdict yet. The request and flag CBs hold this
// many pages each, so the verifier can always post a verdict for every request.
inline constexpr uint32_t SFPU_VERIFY_DEPTH = 4;

}  // namespace tt::tt_fabric::fabric_tests
