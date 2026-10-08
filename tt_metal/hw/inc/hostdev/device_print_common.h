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

constexpr uint32_t DEVICE_PRINT_QUASAR_L2_CACHE_LINE_SIZE = 64;

// Lock for synchronizing access to a print buffer.
#if defined(ARCH_WORMHOLE)
using DevicePrintLockType = uint32_t;  // 0 means free, other values indicate locked by that processor.
#else
using DevicePrintLockType = std::atomic<uint32_t>;  // 0 means free, 1 means locked.
#endif

template <uint32_t LineBytes>
struct alignas(alignof(DevicePrintLockType)) DevicePrintIsolatedLock {
    static_assert(
        LineBytes >= sizeof(DevicePrintLockType) && (LineBytes & (LineBytes - 1)) == 0,
        "Line size must be a power of 2 that holds the lock");
    uint8_t lines[2 * LineBytes];

    DevicePrintLockType& get() { return *reinterpret_cast<DevicePrintLockType*>(line()); }
    volatile DevicePrintLockType& get() volatile { return *reinterpret_cast<volatile DevicePrintLockType*>(line()); }

private:
    uintptr_t line() const volatile {
        return (reinterpret_cast<uintptr_t>(this) + LineBytes - 1) & ~static_cast<uintptr_t>(LineBytes - 1);
    }
};

template <>
struct DevicePrintIsolatedLock<0> {
    DevicePrintLockType lock;

    DevicePrintLockType& get() { return lock; }
    volatile DevicePrintLockType& get() volatile { return lock; }
};

// Offset of the print data in a DevicePrintBuffer, i.e. sizeof(Aux). The host uses it to find the data,
// since it knows the processor count only at run time.
constexpr uint32_t device_print_buffer_data_offset(uint32_t processor_count, uint32_t lock_line_bytes) {
    const uint32_t risc_state_bytes = (processor_count * sizeof(DevicePrintRiscCoreState) + sizeof(uint32_t) - 1) /
                                      sizeof(uint32_t) * sizeof(uint32_t);
    const uint32_t lock_bytes = lock_line_bytes == 0 ? sizeof(DevicePrintLockType) : 2 * lock_line_bytes;
    return sizeof(uint32_t) + sizeof(uint32_t) + risc_state_bytes + lock_bytes;
}

template <uint32_t BufferSize, uint32_t ProcessorCount, uint32_t ProcessorOffset = 0, uint32_t LockLineBytes = 0>
struct DevicePrintBuffer {
    static constexpr uint32_t buffer_size = BufferSize;
    static constexpr uint32_t processor_count = ProcessorCount;
    static constexpr uint32_t processor_offset = ProcessorOffset;

    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
        DevicePrintRiscCoreState risc_state[ProcessorCount];  // Has kernel printed since starting
        DevicePrintIsolatedLock<LockLineBytes> lock;          // Use lock.get()
    } aux;
    static_assert(
        sizeof(Aux) == device_print_buffer_data_offset(ProcessorCount, LockLineBytes),
        "Aux struct size must be correct");
    static_assert(sizeof(Aux) % 4 == 0, "Aux struct must be a multiple of 4 bytes for proper alignment of data");
    uint8_t data[BufferSize - sizeof(Aux)];
    static_assert(sizeof(data) % 4 == 0, "Data array size must be a multiple of 4 bytes for proper alignment");
};
