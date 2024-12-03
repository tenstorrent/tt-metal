#pragma once

#include <cstdint>
#include <cstring>  // for std::memcpy

struct alignas(uint64_t) KernelProfilerNocEventMetadata {
    enum class NocEventType : unsigned char {
        UNDEF = 0,
        READ = 1,
        WRITE = 2,
        READ_BARRIER = 3,
        WRITE_BARRIER = 4,
        WRITE_FLUSH = 5
    };
    enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };

    KernelProfilerNocEventMetadata() = default;

    // used during deserialization
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    // these can be compressed to bit-fields if needed, but byte orientated has less overhead
    uint8_t dst_x = 0;
    uint8_t dst_y = 0;
    NocEventType noc_xfer_type = NocEventType::UNDEF;
    NocType noc_type = NocType::UNDEF;
    uint32_t num_bytes = 0;

    uint64_t asU64() {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
