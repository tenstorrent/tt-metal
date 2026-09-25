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
#elif defined(ARCH_QUASAR) && !defined(ENV_LLK_INFRA)
        // The lock is taken with an amoswap through the cached L1 alias while the rest of the header is
        // written uncached or over the NoC; a dirty line write-back would clobber its neighbours, so
        // the lock gets a 64-byte line of its own.
        uint8_t pad_before_lock[64 - 2 * sizeof(uint32_t) - ProcessorCount];
        std::atomic<uint32_t> lock;
        uint8_t pad_after_lock[64 - sizeof(uint32_t)];
#else
        std::atomic<uint32_t> lock;  // Lock for synchronizing access to the buffer. 0 means free, 1 means locked.
#endif
    } aux;
#if defined(ARCH_QUASAR) && !defined(ENV_LLK_INFRA)
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
