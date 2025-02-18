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
        READ_DRAM_SHARDED_SET_STATE,
        READ_DRAM_SHARDED_WITH_STATE,

        WRITE_,
        WRITE_WITH_TRID,
        WRITE_INLINE,
        WRITE_MULTICAST,
        WRITE_SET_STATE,
        WRITE_WITH_STATE,
        WRITE_WITH_TRID_SET_STATE,
        WRITE_WITH_TRID_WITH_STATE,
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
    static constexpr int8_t INVALID_COORD_VAL = -1;

    KernelProfilerNocEventMetadata() = default;

    // used during deserialization
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    // these can be compressed to bit-fields if needed, but byte orientated has less overhead
    int8_t dst_x = INVALID_COORD_VAL;
    int8_t dst_y = INVALID_COORD_VAL;
    int8_t mcast_end_dst_x = INVALID_COORD_VAL;
    int8_t mcast_end_dst_y = INVALID_COORD_VAL;
    NocEventType noc_xfer_type;
    NocType noc_type : 4;
    NocVirtualChannel noc_vc : 4;
    uint16_t num_bytes;

    uint64_t asU64() const {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
