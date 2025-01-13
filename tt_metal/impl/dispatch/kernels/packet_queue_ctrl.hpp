// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t MAX_SWITCH_FAN_IN = 4;
constexpr uint32_t MAX_SWITCH_FAN_OUT = 4;
constexpr uint32_t MAX_TUNNEL_LANES = 10;

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

enum DispatchRemoteNetworkType : uint8_t {
    DISABLE_QUEUE = 0, // Cheat to skip queue in process_queue
    NOC0 = 1,
    NOC1 = 2,
    ETH = 3,
    NONE = 4
};

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

// Packet Queue Scratch Buffer
struct packet_queue_ptr_buffer_t {
    // padding to make each entry == 1 packet (16B)
    uint32_t wptr;
    uint8_t padding0[12];

    uint32_t rptr_sent;
    uint8_t padding1[12];

    uint32_t rptr_cleared;
    uint8_t padding2[12];

    // Due to the lack of an inc command and we are not using semaphores,
    // a copy of the remote value is stored on the owner queue. It gets incremented here first,
    // and then we write this value to the remote L1.
    // For an input queue, it owns and updates the rptr. And wptr for output queue.
    uint32_t shadow_remote_wptr;
    uint8_t padding3[12];

    uint32_t shadow_remote_rptr_sent;
    uint8_t padding4[12];

    uint32_t shadow_remote_rptr_cleared;
    uint8_t padding5[12];

    // Sent and Recv value for ethernet acks
    uint64_t eth_sent;
    uint8_t padding6[8];

    uint64_t eth_recv;
    uint8_t padding7[8];
} __attribute__((aligned(16)));

constexpr uint32_t packet_queue_ptr_buffer_size = sizeof(packet_queue_ptr_buffer_t);

// Do not modify the scratch buffer without updating the usages of it
static_assert(packet_queue_ptr_buffer_size == 128 && "packet_queue_ptr_buffer_size expected to be 128B");

// Packet Queue Scratch Buffer Memory Layout
struct packet_queue_ptr_buffer_layout_t {
    static constexpr uint32_t WPTR_OFFSET = 0;
    static constexpr uint32_t RPTR_SENT_OFFSET = 16;
    static constexpr uint32_t RPTR_CLEARED_OFFSET = 32;
    static constexpr uint32_t SHADOW_REMOTE_WPTR_OFFSET = 48;
    static constexpr uint32_t SHADOW_REMOTE_RPTR_SENT_OFFSET = 64;
    static constexpr uint32_t SHADOW_REMOTE_RPTR_CLEARED_OFFSET = 80;
    static constexpr uint32_t ETH_SENT_OFFSET = 96;
    static constexpr uint32_t ETH_RECV_OFFSET = 112;

    static volatile uint32_t* get_wptr(uint32_t base_addr) {
        return reinterpret_cast<volatile uint32_t*>(base_addr + WPTR_OFFSET);
    }

    static volatile uint32_t* get_rptr_sent(uint32_t base_addr) {
        return reinterpret_cast<volatile uint32_t*>(base_addr + RPTR_SENT_OFFSET);
    }

    static volatile uint32_t* get_rptr_cleared(uint32_t base_addr) {
        return reinterpret_cast<volatile uint32_t*>(base_addr + RPTR_CLEARED_OFFSET);
    }

    static uint32_t get_wptr_addr(uint32_t base_addr) {
        return base_addr + WPTR_OFFSET;
    }

    static uint32_t get_rptr_sent_addr(uint32_t base_addr) {
        return base_addr + RPTR_SENT_OFFSET;
    }

    static uint32_t get_rptr_cleared_addr(uint32_t base_addr) {
        return base_addr + RPTR_CLEARED_OFFSET;
    }

    static uint32_t* get_shadow_remote_wptr(uint32_t base_addr) {
        return reinterpret_cast<uint32_t*>(base_addr + SHADOW_REMOTE_WPTR_OFFSET);
    }

    static uint32_t* get_shadow_remote_rptr_sent(uint32_t base_addr) {
        return reinterpret_cast<uint32_t*>(base_addr + SHADOW_REMOTE_RPTR_SENT_OFFSET);
    }

    static uint32_t* get_shadow_remote_rptr_cleared(uint32_t base_addr) {
        return reinterpret_cast<uint32_t*>(base_addr + SHADOW_REMOTE_RPTR_CLEARED_OFFSET);
    }

    static volatile uint32_t* get_eth_sent(uint32_t base_addr) {
        return reinterpret_cast<uint32_t*>(base_addr + ETH_SENT_OFFSET);
    }

    static volatile uint32_t* get_eth_recv(uint32_t base_addr) {
        return reinterpret_cast<uint32_t*>(base_addr + ETH_RECV_OFFSET);
    }

    static uint32_t get_eth_sent_addr(uint32_t base_addr) {
        return base_addr + ETH_SENT_OFFSET;
    }

    static uint32_t get_eth_recv_addr(uint32_t base_addr) {
        return base_addr + ETH_RECV_OFFSET;
    }

    // Is this layout correct?
    static_assert(offsetof(packet_queue_ptr_buffer_t, wptr) == WPTR_OFFSET, "wptr offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, rptr_sent) == RPTR_SENT_OFFSET, "rptr_sent offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, rptr_cleared) == RPTR_CLEARED_OFFSET, "rptr_cleared offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, shadow_remote_wptr) == SHADOW_REMOTE_WPTR_OFFSET, "shadow_remote_wptr offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, shadow_remote_rptr_sent) == SHADOW_REMOTE_RPTR_SENT_OFFSET, "shadow_remote_rptr_sent offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, shadow_remote_rptr_cleared) == SHADOW_REMOTE_RPTR_CLEARED_OFFSET, "shadow_remote_rptr_cleared offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, eth_sent) == ETH_SENT_OFFSET, "eth_sent offset mismatch");
    static_assert(offsetof(packet_queue_ptr_buffer_t, eth_recv) == ETH_RECV_OFFSET, "eth_recv offset mismatch");
};

// Return the maximum number of packet_queues that can be used in one kernel
inline uint32_t get_max_queues_used() {
    return std::max(
        MAX_SWITCH_FAN_OUT + 1,
        std::max(
            MAX_SWITCH_FAN_IN + 1,
            MAX_TUNNEL_LANES * 2 // bi directional
        ));
}

// Return the ptr buffer address based on the queue id
inline uint32_t get_ptr_addr(const uint32_t ptr_base_addr, const uint32_t queue_id) {
    uint32_t offset = ptr_base_addr + (queue_id * packet_queue_ptr_buffer_size);
    if (offset > get_max_queues_used() * packet_queue_ptr_buffer_size) {
        // L1[0] is an invalid address
        return 0;
    }
    return ptr_base_addr + offset;
};
