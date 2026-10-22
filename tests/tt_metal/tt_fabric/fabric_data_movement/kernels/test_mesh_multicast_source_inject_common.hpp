// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::fabric_router_tests::source_inject {

constexpr uint32_t ATOMIC_PACKET_COUNT = 100;
constexpr uint32_t PAYLOAD_PACKET_COUNT = 10;
constexpr uint32_t ATOMIC_COUNTER_COUNT = ATOMIC_PACKET_COUNT + 2 * PAYLOAD_PACKET_COUNT;
constexpr uint32_t PAYLOAD_ALIGNMENT = 16;
constexpr uint32_t SENTINEL = 0xA5A5A5A5;
constexpr uint32_t STATUS_READY_TO_STOP = TT_FABRIC_STATUS_MASK | 0x2;

enum class ValidationPhase : uint32_t {
    ATOMIC = 1,
    FUSED_WRITE = 2,
    FUSED_SCATTER = 3,
    PLAIN_WRITE = 4,
    NON_TARGET = 5,
};

constexpr uint32_t align_up(uint32_t value, uint32_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

constexpr uint32_t first_scatter_chunk_size(uint32_t payload_size) {
    return (payload_size / 2) & ~(PAYLOAD_ALIGNMENT - 1);
}

constexpr uint32_t second_scatter_chunk_size(uint32_t payload_size) {
    return payload_size - first_scatter_chunk_size(payload_size);
}

// One source slot is followed by all destination state. Keeping those regions disjoint makes the
// source-chip sentinel check meaningful even though the fixture normally aliases source and target offsets.
constexpr uint32_t counter_base(uint32_t data_base, uint32_t payload_size) {
    return align_up(data_base + payload_size, PAYLOAD_ALIGNMENT);
}

constexpr uint32_t counter_address(uint32_t data_base, uint32_t payload_size, uint32_t index) {
    return counter_base(data_base, payload_size) + index * sizeof(uint32_t);
}

constexpr uint32_t fused_write_base(uint32_t data_base, uint32_t payload_size) {
    return align_up(counter_base(data_base, payload_size) + ATOMIC_COUNTER_COUNT * sizeof(uint32_t), PAYLOAD_ALIGNMENT);
}

constexpr uint32_t fused_write_address(uint32_t data_base, uint32_t payload_size, uint32_t packet_index) {
    return fused_write_base(data_base, payload_size) + packet_index * payload_size;
}

constexpr uint32_t scatter_first_base(uint32_t data_base, uint32_t payload_size) {
    return fused_write_base(data_base, payload_size) + PAYLOAD_PACKET_COUNT * payload_size;
}

constexpr uint32_t scatter_first_address(uint32_t data_base, uint32_t payload_size, uint32_t packet_index) {
    return scatter_first_base(data_base, payload_size) + packet_index * first_scatter_chunk_size(payload_size);
}

// Put plain writes between the two scatter regions. Adjacent halves could let an accidental ordinary
// unicast write masquerade as a correct two-destination scatter.
constexpr uint32_t plain_write_base(uint32_t data_base, uint32_t payload_size) {
    return scatter_first_base(data_base, payload_size) + PAYLOAD_PACKET_COUNT * first_scatter_chunk_size(payload_size);
}

constexpr uint32_t plain_write_address(uint32_t data_base, uint32_t payload_size, uint32_t packet_index) {
    return plain_write_base(data_base, payload_size) + packet_index * payload_size;
}

constexpr uint32_t scatter_second_base(uint32_t data_base, uint32_t payload_size) {
    return plain_write_base(data_base, payload_size) + PAYLOAD_PACKET_COUNT * payload_size;
}

constexpr uint32_t scatter_second_address(uint32_t data_base, uint32_t payload_size, uint32_t packet_index) {
    return scatter_second_base(data_base, payload_size) + packet_index * second_scatter_chunk_size(payload_size);
}

constexpr uint32_t data_end(uint32_t data_base, uint32_t payload_size) {
    return scatter_second_base(data_base, payload_size) +
           PAYLOAD_PACKET_COUNT * second_scatter_chunk_size(payload_size);
}

constexpr uint32_t data_size(uint32_t payload_size) { return data_end(0, payload_size); }

constexpr uint32_t payload_word(ValidationPhase phase, uint32_t packet_index, uint32_t payload_word_index) {
    return 0x10000000u | (static_cast<uint32_t>(phase) << 24) | (packet_index << 16) | payload_word_index;
}

}  // namespace tt::tt_fabric::fabric_router_tests::source_inject
