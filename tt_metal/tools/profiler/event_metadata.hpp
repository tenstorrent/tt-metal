// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>  // for std::memcpy

struct alignas(uint64_t) KernelProfilerNocEventMetadata {
    enum class NocEventType : unsigned char {
        UNDEF = 0,
        READ,
        READ_SET_STATE,
        READ_SET_TRID,
        READ_WITH_STATE,
        READ_WITH_STATE_AND_TRID,
        READ_BARRIER_START,
        READ_BARRIER_END,
        READ_BARRIER_WITH_TRID,

        WRITE_,
        WRITE_INLINE,
        WRITE_MULTICAST,
        WRITE_SET_STATE,
        WRITE_WITH_STATE,
        WRITE_BARRIER_START,
        WRITE_BARRIER_END,
        WRITE_BARRIER_WITH_TRID,
        WRITE_FLUSH,

        ATOMIC_BARRIER,
        SEMAPHORE_INC,
        SEMAPHORE_WAIT,
        SEMAPHORE_SET,

        UNSUPPORTED
    };
    enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };
    using NocVirtualChannel = int8_t;

    KernelProfilerNocEventMetadata() = default;

    // used during deserialization
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    // these can be compressed to bit-fields if needed, but byte orientated has less overhead
    int8_t dst_x = -1;
    int8_t dst_y = -1;
    NocEventType noc_xfer_type = NocEventType::UNDEF;
    NocType noc_type = NocType::UNDEF;
    NocVirtualChannel noc_vc = -1;
    uint32_t num_bytes : 24;

    uint64_t asU64() {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
