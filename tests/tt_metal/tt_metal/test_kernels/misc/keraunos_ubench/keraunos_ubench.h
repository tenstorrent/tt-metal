// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Layout, result block and device helpers shared by the Keraunos CCE microbenchmark kernels and their host test
// (tests/tt_metal/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench.cpp).
//
// Every offset is relative to a per-core base the host passes as runtime arg 0: a 64-byte-aligned flat offset in CCE
// SRAM at or above MEM_IERISC_ROUTER_UNRESERVED_BASE, so the Ethernet-link slot and router regions below it stay
// intact. Two cores of one benchmark use the same base, because an Ethernet send lands at the same offset in the
// link peer's SRAM.
//
// Measurement rules (as in the X280 suite's design.md):
//   - A kernel reads only its own hart's cycle counter (rdcycle). Timestamps from different harts are never
//     subtracted; every reported interval starts and ends on one hart.
//   - Kernels record raw cycle counts. The host does every statistic and every unit conversion.
//   - A round trip is reported as a round trip. It is never halved into a one-way latency.

#pragma once

#include <cstdint>

#if defined(KERNEL_BUILD)
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#endif

namespace keraunos_ubench {

// ---------------------------------------------------------------------------------------------------------------
// Region layout, relative to the base.
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t kMaxHarts = 8;

// One result block per hart.
constexpr uint32_t kResultOffset = 0x0000;
constexpr uint32_t kResultStride = 0x100;
// Start gate for multi-hart runs: arrive[kMaxHarts] words, then the go word.
constexpr uint32_t kGateOffset = 0x0800;
constexpr uint32_t kGateGoOffset = kGateOffset + kMaxHarts * sizeof(uint32_t);
// Mailboxes: one 16-byte slot per hart. 8-byte aligned, so an inline Ethernet write (8 bytes) fits one slot.
constexpr uint32_t kMailboxOffset = 0x0900;
constexpr uint32_t kMailboxStride = 0x10;
// iDMA source staging words: one 16-byte slot per hart.
constexpr uint32_t kStagingOffset = 0x0A00;
constexpr uint32_t kStagingStride = 0x10;
// Ethernet bandwidth: the sender's ack word (receiver -> sender CPU store) and the receiver's flag word
// (sender -> receiver inline Ethernet write).
constexpr uint32_t kEthAckOffset = 0x0B00;
constexpr uint32_t kEthFlagOffset = 0x0B10;
// Per-hart sample arrays (uint32 cycle deltas).
constexpr uint32_t kSamplesOffset = 0x1000;
constexpr uint32_t kMaxSamples = 1024;
constexpr uint32_t kSamplesStride = kMaxSamples * sizeof(uint32_t);
// Bulk data. Each hart owns kDataStride bytes of source, and on a local copy kDataStride bytes of destination
// kDstOffset above its source.
constexpr uint32_t kDataOffset = 0x10000;
constexpr uint32_t kDataStride = 0x10000;
constexpr uint32_t kMaxTransferBytes = kDataStride;
constexpr uint32_t kDstOffset = kMaxHarts * kDataStride;
// Bytes above the base the layout uses.
constexpr uint32_t kRegionBytes = kDataOffset + 2 * kMaxHarts * kDataStride;

constexpr uint32_t result_offset(uint32_t hart) { return kResultOffset + hart * kResultStride; }
constexpr uint32_t mailbox_offset(uint32_t hart) { return kMailboxOffset + hart * kMailboxStride; }
constexpr uint32_t staging_offset(uint32_t hart) { return kStagingOffset + hart * kStagingStride; }
constexpr uint32_t samples_offset(uint32_t hart) { return kSamplesOffset + hart * kSamplesStride; }
constexpr uint32_t src_offset(uint32_t hart) { return kDataOffset + hart * kDataStride; }
constexpr uint32_t local_dst_offset(uint32_t hart) { return src_offset(hart) + kDstOffset; }

// ---------------------------------------------------------------------------------------------------------------
// Keraunos addresses of an HSIO tile's CCE0 SRAM.
// ---------------------------------------------------------------------------------------------------------------

// Chiplet-local view, which the iDMA takes for its own tile's SRAM (eth_cce_send.h's sram_local_addr()).
constexpr uint64_t kHsioLocalBase = 0x20000000ULL;
constexpr uint64_t kHsioCceSramOffset = 0x2800000ULL;
// Global (package) view, which a CPU load or store or an iDMA copy to another tile uses (remap entries 6-9, as in
// KeraunosEmu.EthernetKernelWritesOtherTileSram and quasar_cce_noc_routes()).
constexpr uint64_t kHsioGlobalSramBase = 0x1206800000ULL;

constexpr uint64_t hsio_local_addr(uint32_t hsio, uint64_t tile_stride, uint32_t offset) {
    return kHsioLocalBase + hsio * tile_stride + kHsioCceSramOffset + offset;
}
constexpr uint64_t hsio_global_addr(uint32_t hsio, uint64_t tile_stride, uint32_t offset) {
    return kHsioGlobalSramBase + hsio * tile_stride + offset;
}

// ---------------------------------------------------------------------------------------------------------------
// Modes and status words.
// ---------------------------------------------------------------------------------------------------------------

// How a latency participant puts its token into the peer's mailbox.
enum class LatencyTransport : uint32_t {
    // 32-bit CPU store: to the peer's uncached alias on the same core, or its global SRAM address on another tile.
    CpuStore = 0,
    // 8-byte iDMA copy from this hart's staging word to the peer's mailbox (cce_noc::copy, command buffer 0).
    Idma = 1,
    // Inline 8-byte Ethernet write to the link peer (cce_eth::send_dword), after an optional payload send.
    Eth = 2,
};

enum class LatencyRole : uint32_t {
    Initiator = 0,
    Responder = 1,
};

// How a transport-bandwidth hart moves each transfer.
enum class TransportMech : uint32_t {
    // cce_noc::copy(): the whole command buffer programmed per transfer, as metal's noc_async_write does on a CCE.
    IdmaMetal = 0,
    // The command buffer programmed once; per transfer only source, destination and length, then issue.
    IdmaLean = 1,
    // 64-bit CPU stores through the destination's CPU address.
    CpuStore64 = 2,
};

constexpr uint32_t kStatusRunning = 0x5254'0001u;
constexpr uint32_t kStatusDone = 0x444F'4E45u;
constexpr uint32_t kStatusTimeout = 0x544D'4F55u;

// ---------------------------------------------------------------------------------------------------------------
// The result block. What cycles[] and aux[] hold is per kernel (see each kernel's header comment).
// ---------------------------------------------------------------------------------------------------------------

struct Result {
    uint32_t status;
    uint32_t detail;  // on kStatusTimeout, which wait timed out
    uint32_t iterations_done;
    uint64_t cycles[8];
    uint64_t aux[8];
};
static_assert(sizeof(Result) <= kResultStride);

// Indices into Result::cycles.
namespace timer_check {
constexpr uint32_t kEmptyMin = 0;
constexpr uint32_t kEmptyMax = 1;
constexpr uint32_t kAddLoop = 2;                // dependent adds
constexpr uint32_t kLocalLoadChase = 3;         // dependent loads, own SRAM uncached
constexpr uint32_t kRemoteLoadChase = 4;        // dependent loads, peer tile global address (when the host passes one)
constexpr uint32_t kMonotonicBackwards = 0;     // aux index: reads that went backwards
constexpr uint32_t kMonotonicZeroDelta = 1;     // aux index: reads equal to the previous one
constexpr uint32_t kMonotonicReadCount = 1024;  // back-to-back reads in the monotonic check
}  // namespace timer_check

namespace transport {
constexpr uint32_t kInjection = 0;  // first issue to last issue returned (CPU: last store retired)
constexpr uint32_t kDrained = 1;    // first issue to every transfer acked (iDMA only)
}  // namespace transport

namespace eth_bw {
constexpr uint32_t kInjection = 0;  // first send to last send issued
constexpr uint32_t kAcked = 1;      // first send to every send acked on this core's transaction id
constexpr uint32_t kDelivered = 2;  // first send to the receiver's ack of the closing flag (includes one return path)
constexpr uint32_t kHandshakeRtt = 3;  // the opening flag -> ack round trip, untimed traffic idle
constexpr uint32_t kTxqBusySpins = 0;  // aux index: polls that found the chosen TX queue busy
}  // namespace eth_bw

// Word j of hart h's data pattern. Distinct per hart and per word.
constexpr uint32_t pattern_word(uint32_t hart, uint32_t j) { return ((hart + 1) * 0x9E3779B1u) ^ (j * 0x01000193u); }

#if defined(KERNEL_BUILD)

// This CCE's SRAM at flat offset @p flat, through the uncached alias. Every word another hart, the iDMA, the host or
// an Ethernet RX writes is read this way: none of those writes snoop this hart's cache.
template <typename T = uint32_t>
inline volatile T* uncached(uint32_t flat) {
    return reinterpret_cast<volatile T*>(static_cast<uintptr_t>(flat + MEM_IERISC_UNCACHED_BASE));
}

inline volatile Result* result_block(uint32_t base, uint32_t hart) {
    return uncached<Result>(base + result_offset(hart));
}

// This hart's cycle counter, fenced on both sides so no memory access moves across the read.
inline __attribute__((always_inline)) uint64_t now() {
    uint64_t c;
    asm volatile("fence iorw, iorw\n\trdcycle %0\n\tfence iorw, iorw" : "=r"(c) : : "memory");
    return c;
}

inline uint64_t arg_u64(uint32_t lo_index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(lo_index + 1)) << 32) | get_arg_val<uint32_t>(lo_index);
}

// Polls @p word until it reads @p want, at most @p polls times. False on a timeout.
template <typename T>
inline bool wait_eq(const volatile T* word, T want, uint32_t polls) {
    for (uint32_t p = 0; p < polls; p++) {
        if (*word == want) {
            return true;
        }
    }
    return *word == want;
}

inline void finish(volatile Result* r, uint32_t status) {
    asm volatile("fence iorw, iorw" ::: "memory");
    r->status = status;
    asm volatile("fence iorw, iorw" ::: "memory");
}

// Star barrier over harts [0, num_harts) of this core: each hart sets its arrive word to 1; hart 0 waits for all of
// them, then releases everyone through the go word. The host zeroes the gate before every launch. Returns false on a
// timeout.
inline bool gate(uint32_t base, uint32_t hart, uint32_t num_harts, uint32_t polls) {
    volatile uint32_t* arrive = uncached(base + kGateOffset);
    volatile uint32_t* go = uncached(base + kGateGoOffset);
    arrive[hart] = 1;
    asm volatile("fence iorw, iorw" ::: "memory");
    if (hart == 0) {
        for (uint32_t h = 1; h < num_harts; h++) {
            if (!wait_eq<uint32_t>(&arrive[h], 1, polls)) {
                return false;
            }
        }
        *go = 1;
        asm volatile("fence iorw, iorw" ::: "memory");
        return true;
    }
    return wait_eq<uint32_t>(go, 1, polls);
}

#endif  // KERNEL_BUILD

}  // namespace keraunos_ubench
