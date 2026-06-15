// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
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
#else
        std::atomic<uint32_t> lock;  // Lock for synchronizing access to the buffer. 0 means free, 1 means locked.
#endif
    } aux;
    static_assert(
        sizeof(Aux) == sizeof(uint32_t) + sizeof(uint32_t) +
                           (ProcessorCount * sizeof(DevicePrintRiscCoreState) + sizeof(uint32_t) - 1) /
                               sizeof(uint32_t) * sizeof(uint32_t) +
                           sizeof(uint32_t),
        "Aux struct size must be correct");
    static_assert(sizeof(Aux) % 4 == 0, "Aux struct must be a multiple of 4 bytes for proper alignment of data");
    uint8_t data[BufferSize - sizeof(Aux)];
    static_assert(sizeof(data) % 4 == 0, "Data array size must be a multiple of 4 bytes for proper alignment");
};
