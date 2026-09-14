// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

enum class DevicePrintRiscCoreState : uint8_t {
    KernelNotPrinted = 0,
    KernelPrinted = 1,
    PrintingDisabled = 2,
};

template <uint32_t BufferSize, uint32_t ProcessorCount, uint32_t ProcessorOffset = 0>
struct DevicePrintBuffer {
    static constexpr uint32_t buffer_size = BufferSize;
    static constexpr uint32_t processor_count = ProcessorCount;
    static constexpr uint32_t processor_offset = ProcessorOffset;

    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
        DevicePrintRiscCoreState risc_state[ProcessorCount];  // Has kernel printed since starting
#if defined(ARCH_WORMHOLE)
        uint32_t lock;  // Lock for synchronizing access to the buffer. 0 means free, other values indicate locked by
                        // that processor.
#elif defined(ARCH_QUASAR)
        // The lock is taken with an amoswap through the DM's cached L1 alias, while wpos/rpos/risc_state
        // are written through the uncached alias by the device and over the NoC by the host. A dirty cache
        // line holding the lock is eventually written back with a STALE snapshot of everything that shares
        // its 64 bytes (seen on the qsr.s1 model, even across device resets), so the lock gets a 64-byte
        // line of its own: for any word-aligned buffer base its line lies within Aux bytes 1..127, which
        // hold nothing but padding and the lock. The host mirrors this 128-byte header (dprint_server.cpp).
        uint8_t pad_before_lock[64 - 2 * sizeof(uint32_t) - ((ProcessorCount + 3) / 4) * 4];
        std::atomic<uint32_t> lock;  // Lock for synchronizing access to the buffer. 0 means free, 1 means locked.
        uint8_t pad_after_lock[64 - sizeof(uint32_t)];
#else
        std::atomic<uint32_t> lock;  // Lock for synchronizing access to the buffer. 0 means free, 1 means locked.
#endif
    } aux;
#if defined(ARCH_QUASAR)
    static_assert(sizeof(Aux) == 128, "Aux struct size must be correct");
    static_assert(offsetof(Aux, lock) == 64, "The print lock must start its own 64-byte line");
#else
    static_assert(
        sizeof(Aux) == sizeof(uint32_t) + sizeof(uint32_t) +
                           (ProcessorCount * sizeof(DevicePrintRiscCoreState) + sizeof(uint32_t) - 1) /
                               sizeof(uint32_t) * sizeof(uint32_t) +
                           sizeof(uint32_t),
        "Aux struct size must be correct");
#endif
    static_assert(sizeof(Aux) % 4 == 0, "Aux struct must be a multiple of 4 bytes for proper alignment of data");
    uint8_t data[BufferSize - sizeof(Aux)];
    static_assert(sizeof(data) % 4 == 0, "Data array size must be a multiple of 4 bytes for proper alignment");
};
