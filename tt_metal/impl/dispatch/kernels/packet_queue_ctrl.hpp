// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t MAX_SWITCH_FAN_IN = 4;
constexpr uint32_t MAX_SWITCH_FAN_OUT = 4;

constexpr uint32_t MAX_SRC_ENDPOINTS = 32;
constexpr uint32_t MAX_DEST_ENDPOINTS = 32;

constexpr uint32_t INPUT_QUEUE_START_ID = 0;
constexpr uint32_t OUTPUT_QUEUE_START_ID = MAX_SWITCH_FAN_IN;

constexpr uint32_t PACKET_QUEUE_REMOTE_READY_FLAG = 0xA;
constexpr uint32_t PACKET_QUEUE_REMOTE_FINISHED_FLAG = 0xB;

constexpr uint32_t PACKET_QUEUE_STAUS_MASK = 0xabc00000;
constexpr uint32_t PACKET_QUEUE_TEST_STARTED = PACKET_QUEUE_STAUS_MASK | 0x0;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = PACKET_QUEUE_STAUS_MASK | 0x1;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = PACKET_QUEUE_STAUS_MASK | 0x2;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = PACKET_QUEUE_STAUS_MASK | 0x3;

// indexes of return values in test results buffer
constexpr uint32_t PQ_TEST_STATUS_INDEX = 0;
constexpr uint32_t PQ_TEST_WORD_CNT_INDEX = 2;
constexpr uint32_t PQ_TEST_CYCLES_INDEX = 4;
constexpr uint32_t PQ_TEST_ITER_INDEX = 6;
constexpr uint32_t PQ_TEST_MISC_INDEX = 16;


enum DispatchPacketFlag : uint32_t {
    PACKET_CMD_START = (0x1 << 1),
    PACKET_CMD_END = (0x1 << 2),
    PACKET_MULTI_CMD = (0x1 << 3),
    PACKET_TEST_LAST = (0x1 << 15),  // test only
};

enum DispatchRemoteNetworkType : uint32_t {
    NOC0 = 0,
    NOC1 = 1,
    ETH = 2,
    NONE = 3
};

inline bool is_remote_network_type_noc(DispatchRemoteNetworkType type) {
    return type == NOC0 || type == NOC1;
}

struct dispatch_packet_header_t {

    uint32_t packet_size_bytes;
    uint16_t packet_src;
    uint16_t packet_dest;
    uint16_t packet_flags;
    uint16_t num_cmds;
    uint32_t tag;

    inline bool check_packet_flags(uint32_t flags) const {
        return (packet_flags & flags) == flags;
    }

    inline void set_packet_flags(uint32_t flags) {
        packet_flags |= flags;
    }

    inline void clear_packet_flags(uint32_t flags) {
        packet_flags &= ~flags;
    }
};

#define is_power_of_2(x) (((x) > 0) && (((x) & ((x) - 1)) == 0))

inline uint32_t packet_switch_4B_pack(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
    return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
}

static_assert(MAX_DEST_ENDPOINTS <= 32,
              "MAX_DEST_ENDPOINTS must be <= 32 for the packing funcitons below to work");

static_assert(MAX_SWITCH_FAN_OUT <= 4,
              "MAX_SWITCH_FAN_OUT must be <= 4 for the packing funcitons below to work");

inline uint64_t packet_switch_dest_pack(uint32_t* dest_output_map_array, uint32_t num_dests) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < num_dests; i++) {
        result |= ((uint64_t)(dest_output_map_array[i])) << (2*i);
    }
    return result;
}
