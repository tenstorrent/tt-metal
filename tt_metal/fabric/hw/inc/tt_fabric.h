// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include <hostdevcommon/common_values.hpp>
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include <fabric_host_interface.h>
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"

using namespace tt::tt_fabric;

#ifndef DISABLE_LOW_LATENCY_ROUTING
#ifndef LOW_LATENCY_ROUTING
#define LOW_LATENCY_ROUTING
#endif
#endif

#define FVC_MODE_ROUTER 1
#define FVC_MODE_ENDPOINT 2

#define STREAM_ID_ETH_WORDS_RECEIVED 0
#define STREAM_ID_ETH_RECEIVER_BUFFER_SPACE 1
#define STREAM_ID_ETH_SENDER_BUFFER_SPACE 2
#define STREAM_ID_NOC_WORDS_RECEIVED 6
#define STREAM_ID_NOC_RECEIVER_BUFFER_SPACE 10

constexpr uint32_t SYNC_BUF_SIZE = 16;  // must be 2^N
constexpr uint32_t SYNC_BUF_SIZE_MASK = (SYNC_BUF_SIZE - 1);
constexpr uint32_t SYNC_BUF_PTR_MASK = ((SYNC_BUF_SIZE << 1) - 1);

extern uint64_t xy_local_addr;
extern volatile local_pull_request_t* local_pull_request;
extern volatile tt_l1_ptr fabric_router_l1_config_t* routing_table;
extern chan_payload_ptr inbound_rdptr_ack;
extern volatile chan_payload_ptr remote_rdptr;

void tt_fabric_reserve_pull_request_slot(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request, uint32_t num_slots = 1);
template <bool blocking_mode>
bool tt_fabric_check_pull_request_slot(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request, uint32_t wrptr);
uint64_t tt_fabric_send_pull_request(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request);
uint32_t num_words_available_to_pull(volatile pull_request_t* pull_request);
uint32_t words_before_pull_buffer_wrap(uint32_t buffer_size, uint32_t rd_ptr);
uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words);
uint32_t get_rd_ptr_offset_words(pull_request_t* pull_request);

inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

struct fvc_outbound_push_state_t {
    uint32_t local_rdptr;
    uint32_t buffer_size;
    uint32_t buffer_slot_start[FABRIC_ROUTER_OUTBOUND_BUF_SLOTS];
    uint32_t remote_buffer_size;
    uint32_t remote_buffer_slot_start[FABRIC_ROUTER_INBOUND_BUF_SLOTS];
    uint32_t slots_sent_remote_update;
    uint32_t buffer_id_patch;
    volatile uint32_t* sender_slots_cleared;
    volatile uint32_t* update_sender_slots_cleared;
    volatile uint32_t* receiver_buffer_space;
    volatile uint32_t* update_receiver_buffer_space;
    volatile uint32_t* slot_credits;
    uint32_t* update_slot_credits;
    volatile uint64_t* slots_cleared_ack_addr;

    inline void init(uint32_t buffer_id, uint32_t data_buf_start, uint32_t data_buf_size_words) {
        uint32_t words = sizeof(fvc_outbound_push_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        uint32_t buffer_start = data_buf_start;
        for (uint32_t i = 0; i < FABRIC_ROUTER_OUTBOUND_BUF_SLOTS; i++) {
            buffer_slot_start[i] = buffer_start + i * FABRIC_ROUTER_BUF_SLOT_SIZE;
        }
        buffer_size = FABRIC_ROUTER_OUTBOUND_BUF_SLOTS;
        uint32_t remote_buffer_start = FABRIC_ROUTER_DATA_BUF_START + 4 * FABRIC_ROUTER_OUTBOUND_BUF_SIZE;
        for (uint32_t i = 0; i < FABRIC_ROUTER_INBOUND_BUF_SLOTS; i++) {
            remote_buffer_slot_start[i] = remote_buffer_start + i * FABRIC_ROUTER_BUF_SLOT_SIZE;
        }
        remote_buffer_size = FABRIC_ROUTER_INBOUND_BUF_SLOTS;
        // used to send word credits to ethernet receiver
        slots_sent_remote_update =
            (STREAM_REG_ADDR(STREAM_ID_ETH_WORDS_RECEIVED, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        // written by ethernet receiver to signal words cleared from sender buffer.
        uint32_t sender_eth_credits_reg = buffer_id + STREAM_ID_ETH_SENDER_BUFFER_SPACE;
        sender_slots_cleared = reinterpret_cast<uint32_t*>(
            STREAM_REG_ADDR(sender_eth_credits_reg, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_sender_slots_cleared = reinterpret_cast<uint32_t*>(
            STREAM_REG_ADDR(sender_eth_credits_reg, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        // used to track available buffer space in ethernet receiver. Initialized to receiver buffer size in words.
        // ethernet sender decrements it. ethernet receiver increments it.
        receiver_buffer_space = reinterpret_cast<uint32_t*>(
            STREAM_REG_ADDR(STREAM_ID_ETH_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_receiver_buffer_space = reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(
            STREAM_ID_ETH_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        // credit register that is incremented by fabric data producer/pusher.
        uint32_t sender_noc_credits_reg = buffer_id + STREAM_ID_NOC_WORDS_RECEIVED;
        slot_credits = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(sender_noc_credits_reg, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_slot_credits = reinterpret_cast<uint32_t*>(
            STREAM_REG_ADDR(sender_noc_credits_reg, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        volatile uint64_t* sender_addr = (volatile uint64_t*)FABRIC_ROUTER_REQ_QUEUE_START;
        slots_cleared_ack_addr = &sender_addr[buffer_id];
        buffer_id_patch = buffer_id << 30;

        *(uint32_t*)(STREAM_REG_ADDR(sender_eth_credits_reg, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;
        *(uint32_t*)(STREAM_REG_ADDR(STREAM_ID_ETH_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) =
            remote_buffer_size;
        *(uint32_t*)(STREAM_REG_ADDR(sender_noc_credits_reg, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;

        if (buffer_id == 0) {
            // set the address where credits are returned to noc senders.
            // don't need to set the upper 32-bits since those will be updated on every write.
            // using noc 1 for stateful operation.
            // noc 0 is used for inline dword writes that go in different directions, so we cannot really
            // cache the dest address and other fields.
            noc_inline_dw_write_set_state<true>(
                STREAM_REG_ADDR(
                    STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
                0,
                0xF,
                write_at_cmd_buf,
                1);
        }
    }

    FORCE_INLINE uint32_t get_num_slots_cleared() { return *sender_slots_cleared; }

    FORCE_INLINE uint32_t get_remote_num_slots_free() { return *receiver_buffer_space; }

    FORCE_INLINE uint32_t get_local_buffer_read_addr() { return buffer_slot_start[local_rdptr]; }

    FORCE_INLINE void advance_local_rdptr(uint32_t num_words) {
        if constexpr (is_power_of_2(FABRIC_ROUTER_OUTBOUND_BUF_SLOTS)) {
            local_rdptr = (local_rdptr + num_words) & (FABRIC_ROUTER_OUTBOUND_BUF_SLOTS - 1);
        } else {
            uint32_t temp = local_rdptr + num_words;
            if (temp >= buffer_size) {
                temp -= buffer_size;
            }
            local_rdptr = temp;
        }
        *update_receiver_buffer_space = (-num_words) << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    FORCE_INLINE void advance_remote_wrptr(uint32_t& remote_wrptr, uint32_t num_words) {
        if constexpr (is_power_of_2(FABRIC_ROUTER_INBOUND_BUF_SLOTS)) {
            remote_wrptr = (remote_wrptr + num_words) & (FABRIC_ROUTER_INBOUND_BUF_SLOTS - 1);
        } else {
            uint32_t temp = remote_wrptr + num_words;
            if (temp >= remote_buffer_size) {
                temp -= remote_buffer_size;
            }
            remote_wrptr = temp;
        }
    }

    FORCE_INLINE void relay_slots_cleared() {
        uint32_t slots_cleared = get_num_slots_cleared();
        if (slots_cleared == 0) {
            return;
        } else {
            // relay the credits to noc data sender to replenish buffer space in noc sender
            noc_inline_dw_write_with_state<false, true, true, true, true>(
                slots_cleared << REMOTE_DEST_BUF_WORDS_FREE_INC,
                *slots_cleared_ack_addr >> NOC_ADDR_COORD_SHIFT,
                write_at_cmd_buf,
                1);
            // clear the credits receied from ethernet receiver.
            *update_sender_slots_cleared = (-slots_cleared) << REMOTE_DEST_BUF_WORDS_FREE_INC;
        }
    }

    inline bool forward_data_from_fvc_buffer(uint32_t& remote_wrptr) {
        relay_slots_cleared();

        uint32_t slots_to_forward = *slot_credits;
        // new_slots can be 0 here when idle
        // We will still do all the operations since having a branch here hurts perf.

        uint32_t remote_fvc_buffer_space = get_remote_num_slots_free();
        bool nothing_to_do = (slots_to_forward == 0) || (remote_fvc_buffer_space == 0);

        // if either slots to forward or remote buffer space is 0
        // there is nothing to forward
        if (nothing_to_do == true) {
            return false;
        }

        volatile uint32_t* packet_start = (volatile uint32_t*)(get_local_buffer_read_addr());
        uint32_t temp = *packet_start << 2;
        // uint32_t words_to_forward = ((temp + ((PACKET_WORD_SIZE_BYTES << 2) - 1)) >> 6);
        *packet_start = (temp >> 2) | buffer_id_patch;
        uint32_t dest_addr = remote_buffer_slot_start[remote_wrptr];
        internal_::eth_send_packet_byte_addr(
            0, (uint32_t)packet_start, dest_addr, FABRIC_ROUTER_BUF_SLOT_SIZE / PACKET_WORD_SIZE_BYTES);
        // send word credits to receiver
        internal_::eth_write_remote_reg<false>(
            0, (uint32_t)slots_sent_remote_update, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
        advance_local_rdptr(1);
        advance_remote_wrptr(remote_wrptr, 1);
        // decrement slots to be forwarded by 1.
        *update_slot_credits = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
        return true;
    }
};

static_assert(sizeof(fvc_outbound_push_state_t) % 4 == 0);

struct fvc_inbound_push_state_t {
    chan_payload_ptr inbound_wrptr;
    uint32_t slots_inbound;
    uint32_t slots_cleared;
    uint32_t packet_words_remaining;
    uint32_t packet_size_bytes;
    uint32_t fvc_out_rdptr;
    uint32_t buffer_size;
    uint32_t buffer_slot_start[FABRIC_ROUTER_INBOUND_BUF_SLOTS];
    uint32_t remote_buffer_size;
    uint32_t remote_buffer_slot_start[FABRIC_ROUTER_OUTBOUND_BUF_SLOTS];
    uint32_t remote_wrptr[4];
    uint32_t remote_wrptr_direction;
    uint32_t mcast_router_noc_xy[4];
    uint32_t router_push_addr;
#ifdef LOW_LATENCY_ROUTING
    volatile low_latency_packet_header_t* packet_header;
#else
    uint32_t my_id;
    uint32_t last_dest_id;
    uint32_t dest_addr;
    uint32_t command;
    uint32_t for_local_chip;
    volatile packet_header_t* packet_header;
#endif
    volatile uint32_t* slots_received;
    uint32_t* slots_received_local_update;
    uint32_t update_sender_buffer_space[4];
    uint32_t update_receiver_buffer_space;
    uint32_t sender_buffer_index;
    volatile uint32_t* next_router_space;  // need 3 for 3 forwarding directions
    volatile uint32_t* update_router_space;
#ifndef LOW_LATENCY_ROUTING
    uint8_t port_direction_table[16];
#endif

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words) {
        uint32_t words = sizeof(fvc_inbound_push_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
#ifndef LOW_LATENCY_ROUTING
        my_id = routing_table->my_device_id << 16 | routing_table->my_mesh_id;
        last_dest_id = 0xFFFFFFFF;  // initialize to an invalid mesh/device.
#endif
        uint32_t buffer_start = data_buf_start;
        for (uint32_t i = 0; i < FABRIC_ROUTER_INBOUND_BUF_SLOTS; i++) {
            buffer_slot_start[i] = buffer_start + i * FABRIC_ROUTER_BUF_SLOT_SIZE;
        }
        buffer_size = FABRIC_ROUTER_INBOUND_BUF_SLOTS;
        remote_buffer_size = FABRIC_ROUTER_OUTBOUND_BUF_SLOTS;
        slots_received = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(STREAM_ID_ETH_WORDS_RECEIVED, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        slots_received_local_update = reinterpret_cast<uint32_t*>(
            STREAM_REG_ADDR(STREAM_ID_ETH_WORDS_RECEIVED, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        for (uint32_t i = eth_chan_directions::EAST; i < eth_chan_directions::COUNT; i++) {
            update_sender_buffer_space[i] = (STREAM_REG_ADDR(
                STREAM_ID_ETH_SENDER_BUFFER_SPACE + i, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        }

        update_receiver_buffer_space = (STREAM_REG_ADDR(
            STREAM_ID_ETH_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        // credit register that is incremented by fabric data producer/pusher.
        // will need to be unique per direction, if routers per direction have different buffer sizes.
        // or if we want to track buffer space for each router separately.
        next_router_space = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_router_space = reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(
            STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        *(uint32_t*)(STREAM_REG_ADDR(STREAM_ID_NOC_RECEIVER_BUFFER_SPACE, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) =
            remote_buffer_size;
        *(uint32_t*)(STREAM_REG_ADDR(STREAM_ID_ETH_WORDS_RECEIVED, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)) = 0;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    void register_with_routers(uint32_t device_id, uint32_t mesh_id = 0) {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            tt::tt_fabric::chan_id_t my_chan = routing_table->intra_mesh_table.dest_entry[device_id];
            uint32_t my_direction;

            if (routing_table->port_direction.directions[eth_chan_directions::EAST] == my_chan) {
                my_direction = eth_chan_directions::EAST;
                mcast_router_noc_xy[eth_chan_directions::WEST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::WEST]];
                mcast_router_noc_xy[eth_chan_directions::NORTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::NORTH]];
                mcast_router_noc_xy[eth_chan_directions::SOUTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::SOUTH]];
            } else if (routing_table->port_direction.directions[eth_chan_directions::WEST] == my_chan) {
                my_direction = eth_chan_directions::WEST;
                mcast_router_noc_xy[eth_chan_directions::EAST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::EAST]];
                mcast_router_noc_xy[eth_chan_directions::NORTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::NORTH]];
                mcast_router_noc_xy[eth_chan_directions::SOUTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::SOUTH]];
            } else if (routing_table->port_direction.directions[eth_chan_directions::NORTH] == my_chan) {
                my_direction = eth_chan_directions::NORTH;
                mcast_router_noc_xy[eth_chan_directions::EAST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::EAST]];
                mcast_router_noc_xy[eth_chan_directions::WEST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::WEST]];
                mcast_router_noc_xy[eth_chan_directions::SOUTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::SOUTH]];
            } else if (routing_table->port_direction.directions[eth_chan_directions::SOUTH] == my_chan) {
                my_direction = eth_chan_directions::SOUTH;
                mcast_router_noc_xy[eth_chan_directions::EAST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::EAST]];
                mcast_router_noc_xy[eth_chan_directions::WEST] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::WEST]];
                mcast_router_noc_xy[eth_chan_directions::NORTH] =
                    eth_chan_to_noc_xy[noc_index][routing_table->port_direction.directions[eth_chan_directions::NORTH]];
            }

            // set the stream auto increment to use on remote router according to
            // this router's direction.
            router_push_addr = (STREAM_REG_ADDR(
                STREAM_ID_NOC_WORDS_RECEIVED + my_direction, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
            uint32_t remote_buffer_start =
                FABRIC_ROUTER_DATA_BUF_START + my_direction * FABRIC_ROUTER_OUTBOUND_BUF_SIZE;
            for (uint32_t i = 0; i < FABRIC_ROUTER_OUTBOUND_BUF_SLOTS; i++) {
                remote_buffer_slot_start[i] = remote_buffer_start + i * FABRIC_ROUTER_BUF_SLOT_SIZE;
            }
            for (uint32_t i = eth_chan_directions::EAST; i < eth_chan_directions::COUNT; i++) {
                if (i == my_direction) {
                    // skip self
                    continue;
                }
                uint32_t forwarding_channel = routing_table->port_direction.directions[i];
                if (forwarding_channel == INVALID_DIRECTION) {
                    // No channel in this forwarding direction
                    continue;
                }
#ifndef LOW_LATENCY_ROUTING
                port_direction_table[forwarding_channel] = i;
#endif
                // register the stream_reg to receive word credit updates from
                // the 3 routers that this router forwards traffic to.
                // Every outbound router has 4 data pushers. 3 are directional routers and
                // fourth is a local device worker. Each pusher has to register its local address where
                // outbound router should return credits for available buffer space.
                // Each NOC return address is 64-bits (8 Bytes).
                // A pusher registers its return address by calculating its respecitve its entry in the 4 entry
                // outbound router table.
                // direction can be one of 0, 1, 2, 3. Respective 64-bit entry is direction * 8
                uint32_t my_direction_offset = my_direction * sizeof(uint64_t);
                uint64_t router_addr = get_noc_addr_helper(
                    eth_chan_to_noc_xy[noc_index][forwarding_channel],
                    FABRIC_ROUTER_REQ_QUEUE_START + my_direction_offset);
                // Split 64-bit wirte data to two 4-byte write.
                // Write lower 4 Bytes of 8 Byte entry.
                noc_inline_dw_write(router_addr, (uint32_t)update_router_space);
                // Write upper 4 Bytes of 8 Byte entry.
                noc_inline_dw_write(router_addr + (sizeof(uint32_t)), xy_local_addr >> NOC_ADDR_COORD_SHIFT);
            }
        } else {
            uint32_t router_direction = get_next_hop_router_direction(mesh_id, device_id);
            uint32_t router_addr_h = get_next_hop_router_noc_xy(mesh_id, device_id);
            uint64_t router_addr = get_noc_addr_helper(router_addr_h, FABRIC_ROUTER_REQ_QUEUE_START);
            router_push_addr = (STREAM_REG_ADDR(
                STREAM_ID_NOC_WORDS_RECEIVED + router_direction,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
            router_addr += router_direction * sizeof(uint64_t);
            // stream register to receive router buffer space available updates.
            noc_inline_dw_write(router_addr, (uint32_t)update_router_space);
            noc_inline_dw_write(router_addr + sizeof(uint32_t), xy_local_addr >> NOC_ADDR_COORD_SHIFT);
            uint32_t remote_buffer_start =
                FABRIC_ROUTER_DATA_BUF_START + router_direction * FABRIC_ROUTER_OUTBOUND_BUF_SIZE;
            for (uint32_t i = 0; i < FABRIC_ROUTER_OUTBOUND_BUF_SLOTS; i++) {
                remote_buffer_slot_start[i] = remote_buffer_start + i * FABRIC_ROUTER_BUF_SLOT_SIZE;
            }
            remote_wrptr_direction = router_direction;
        }
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr, uint32_t inc) {
        uint32_t temp = ptr + inc;
        if (temp >= buffer_size) {
            temp -= buffer_size;
        }
        return temp;
    }

    inline void advance_local_wrptr(uint32_t num_slots) {
        inbound_wrptr.ptr = inc_ptr_with_wrap(inbound_wrptr.ptr, num_slots);
        slots_inbound += num_slots;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void advance_out_rdptr(uint32_t num_slots) {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            if constexpr (is_power_of_2(FABRIC_ROUTER_INBOUND_BUF_SLOTS)) {
                fvc_out_rdptr = (fvc_out_rdptr + num_slots) & (FABRIC_ROUTER_INBOUND_BUF_SLOTS - 1);
            } else {
                uint32_t temp = fvc_out_rdptr + num_slots;
                if (temp >= buffer_size) {
                    temp -= buffer_size;
                }
                fvc_out_rdptr = temp;
            }
            *slots_received_local_update = (-num_slots) << REMOTE_DEST_BUF_WORDS_FREE_INC;
        } else {
            uint32_t temp = fvc_out_rdptr + num_slots;
            if (temp >= buffer_size) {
                temp -= buffer_size;
            }
            fvc_out_rdptr = temp;
        }
    }

#ifdef LOW_LATENCY_ROUTING
    FORCE_INLINE void advance_remote_wrptr(uint32_t num_slots, uint32_t direction) {
        if constexpr (is_power_of_2(FABRIC_ROUTER_OUTBOUND_BUF_SLOTS)) {
            remote_wrptr[direction] = (remote_wrptr[direction] + num_slots) & (FABRIC_ROUTER_OUTBOUND_BUF_SLOTS - 1);
        } else {
            uint32_t temp = remote_wrptr[direction] + num_slots;
            if (temp >= remote_buffer_size) {
                temp -= remote_buffer_size;
            }
            remote_wrptr[direction] = temp;
        }
    }
#else
    FORCE_INLINE void advance_remote_wrptr(uint32_t num_slots) {
        if constexpr (is_power_of_2(FABRIC_ROUTER_OUTBOUND_BUF_SLOTS)) {
            remote_wrptr[remote_wrptr_direction] =
                (remote_wrptr[remote_wrptr_direction] + num_slots) & (FABRIC_ROUTER_OUTBOUND_BUF_SLOTS - 1);
        } else {
            uint32_t temp = remote_wrptr[remote_wrptr_direction] + num_slots;
            if (temp >= remote_buffer_size) {
                temp -= remote_buffer_size;
            }
            remote_wrptr[remote_wrptr_direction] = temp;
        }
    }
#endif

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE uint32_t get_num_slots_available() {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            return *slots_received;
        } else {
            return slots_inbound - slots_cleared;
        }
    }

    FORCE_INLINE
    uint32_t get_num_slots_free() { return buffer_size - slots_inbound; }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE bool get_curr_packet_valid() {
        if (packet_words_remaining) {
            return true;
        }
        if (get_num_slots_available<fvc_mode>()) {
            // Wait for a full packet header to arrive before advancing to next packet.
            return advance_next_packet<fvc_mode>();
        } else if (slots_cleared) {
            flush_async_writes<fvc_mode, true>();
        }
        return false;
    }

    FORCE_INLINE uint32_t get_local_buffer_read_addr() { return buffer_slot_start[fvc_out_rdptr]; }

    FORCE_INLINE uint32_t get_local_buffer_write_addr() { return buffer_slot_start[inbound_wrptr.ptr]; }

    FORCE_INLINE void free_sender_buffer_space(uint32_t words) {
        // send received word credits to receiver
        internal_::eth_write_remote_reg<false>(
            0, (uint32_t)update_sender_buffer_space[sender_buffer_index], words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void free_receiver_buffer_space(uint32_t words) {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            // send received word credits to receiver
            internal_::eth_write_remote_reg<false>(
                0, (uint32_t)update_receiver_buffer_space, words << REMOTE_DEST_BUF_WORDS_FREE_INC);
        } else {
            slots_inbound -= words;
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE bool advance_next_packet() {
#ifdef LOW_LATENCY_ROUTING
        packet_header = (volatile low_latency_packet_header_t*)get_local_buffer_read_addr();
#else
        packet_header = (volatile packet_header_t*)get_local_buffer_read_addr();
#endif
        uint32_t packet_size = packet_header->routing.packet_size_bytes;
        sender_buffer_index = packet_size >> 30;
        packet_size &= 0x3FFFFFFF;
        packet_size_bytes = packet_size;

        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            free_sender_buffer_space(1);
        }
#ifndef LOW_LATENCY_ROUTING
        for_local_chip = ((volatile uint32_t*)packet_header)[1] == my_id;
#endif
        packet_words_remaining = (packet_size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
        return true;
    }

#ifndef LOW_LATENCY_ROUTING
    uint32_t get_next_hop_router_noc_xy() {
        uint32_t dst_mesh_id = packet_header->routing.dst_mesh_id;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            uint32_t next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            ASSERT(next_port != INVALID_DIRECTION);
            remote_wrptr_direction = port_direction_table[next_port];
            return eth_chan_to_noc_xy[noc_index][next_port];
        } else {
            uint32_t dst_device_id = packet_header->routing.dst_dev_id;
            uint32_t next_port = routing_table->intra_mesh_table.dest_entry[dst_device_id];
            ASSERT(next_port != INVALID_DIRECTION);
            remote_wrptr_direction = port_direction_table[next_port];
            return eth_chan_to_noc_xy[noc_index][next_port];
        }
    }
#endif

    uint32_t get_next_hop_router_noc_xy(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
        if (dst_mesh_id != routing_table->my_mesh_id) {
            uint32_t next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        } else {
            uint32_t next_port = routing_table->intra_mesh_table.dest_entry[dst_dev_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        }
    }

    inline uint32_t get_next_hop_router_direction(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
        uint32_t next_port = 0;
        uint32_t direction = 0;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            ASSERT(next_port != INVALID_DIRECTION);
        } else {
            next_port = routing_table->intra_mesh_table.dest_entry[dst_dev_id];
            ASSERT(next_port != INVALID_DIRECTION);
        }

        if (routing_table->port_direction.directions[eth_chan_directions::EAST] == next_port) {
            direction = eth_chan_directions::EAST;
        } else if (routing_table->port_direction.directions[eth_chan_directions::WEST] == next_port) {
            direction = eth_chan_directions::WEST;
        } else if (routing_table->port_direction.directions[eth_chan_directions::NORTH] == next_port) {
            direction = eth_chan_directions::NORTH;
        } else if (routing_table->port_direction.directions[eth_chan_directions::SOUTH] == next_port) {
            direction = eth_chan_directions::SOUTH;
        }
        return direction;
    }

#ifdef LOW_LATENCY_ROUTING
    template <uint8_t fvc_mode = FVC_MODE_ENDPOINT>
    inline uint32_t push_data_to_eth_router(uint32_t dest_id) {
        if constexpr (fvc_mode == FVC_MODE_ENDPOINT) {
            if (*next_router_space < 1) {
                return 0;
            }
            if (slots_cleared) {
                flush_async_writes<fvc_mode, true>();
            }

            uint32_t dest_addr = get_next_hop_router_noc_xy(dest_id >> 16, dest_id & 0xFFFF);
            uint64_t buffer_wr_addr =
                get_noc_addr_helper(dest_addr, remote_buffer_slot_start[remote_wrptr[remote_wrptr_direction]]);
            // Instead of sending the packet size (packet_words_remaining * PACKET_WORD_SIZE_BYTES) which
            // may be less than the full slot, send the full slot.
            noc_async_write_one_packet(
                get_local_buffer_read_addr(), buffer_wr_addr, FABRIC_ROUTER_BUF_SLOT_SIZE, noc_index);
            advance_remote_wrptr(1, remote_wrptr_direction);
            advance_out_rdptr<fvc_mode>(1);
            uint64_t push_addr = get_noc_addr_helper(dest_addr, router_push_addr);
            noc_inline_dw_write<true, true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);

            *update_router_space = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            uint32_t words_available = packet_words_remaining;
            slots_cleared += 1;
            packet_words_remaining = 0;
            return words_available;
        }
        return 0;
    }

    FORCE_INLINE void issue_local_write() {
        uint32_t addr_l = packet_header->routing.target_offset_l;
        uint32_t addr_h = packet_header->routing.target_offset_h;
        uint32_t command = packet_header->routing.command;
        if (command & ASYNC_WR) {
            noc_async_write_one_packet(
                get_local_buffer_read_addr() + PACKET_HEADER_SIZE_BYTES,
                get_noc_addr_helper(addr_h, addr_l),
                packet_size_bytes - PACKET_HEADER_SIZE_BYTES);
        }
        if (command & ATOMIC_INC) {
            uint64_t noc_addr =
                get_noc_addr_helper(packet_header->routing.atomic_offset_h, packet_header->routing.atomic_offset_l);
            noc_fast_atomic_increment(
                noc_index,
                NCRISC_AT_CMD_BUF,
                noc_addr,
                NOC_UNICAST_WRITE_VC,
                packet_header->routing.atomic_increment,
                packet_header->routing.atomic_wrap,
                false);
        }
    }

    FORCE_INLINE void issue_forward(uint32_t route_value, uint32_t direction) {
        packet_header->routing.route_vector.hop_index = route_value + 1;
        uint64_t buffer_wr_addr =
            get_noc_addr_helper(mcast_router_noc_xy[direction], remote_buffer_slot_start[remote_wrptr[direction]]);
        noc_async_write_one_packet(get_local_buffer_read_addr(), buffer_wr_addr, FABRIC_ROUTER_BUF_SLOT_SIZE);
        advance_remote_wrptr(1, direction);
        uint64_t push_addr = get_noc_addr_helper(mcast_router_noc_xy[direction], router_push_addr);
        noc_inline_dw_write<true, true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
        *update_router_space = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER, uint32_t router_direction = 0>
    FORCE_INLINE __attribute__((optimize("jump-tables"))) void process_inbound_packet() {
        tt_l1_ptr uint8_t* route_vector = (uint8_t*)packet_header->routing.route_vector.value;
        uint32_t hop_index = packet_header->routing.route_vector.hop_index;
        uint32_t hop_cmd = route_vector[hop_index];

        if (slots_cleared) {
            flush_async_writes<fvc_mode, true>();
        }

        switch (hop_cmd) {
            case 0x0: break;
            case tt_low_latency_routing_vector::FORWARD_EAST:
                if constexpr (router_direction == eth_chan_directions::EAST) {
                    issue_local_write();
                } else {
                    if (*next_router_space < 1) {
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::EAST);
                }
                break;
            case tt_low_latency_routing_vector::FORWARD_WEST:
                if constexpr (router_direction == eth_chan_directions::WEST) {
                    issue_local_write();
                } else {
                    if (*next_router_space < 1) {
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::WEST);
                }
                break;
            case tt_low_latency_routing_vector::FORWARD_EAST | tt_low_latency_routing_vector::FORWARD_WEST:
                if constexpr (router_direction == eth_chan_directions::WEST) {
                    issue_local_write();
                    if (*next_router_space < 1) {
                        route_vector[hop_index] = hop_cmd & ~tt_low_latency_routing_vector::FORWARD_WEST;
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::EAST);
                } else {
                    issue_local_write();
                    if (*next_router_space < 1) {
                        route_vector[hop_index] = hop_cmd & ~tt_low_latency_routing_vector::FORWARD_EAST;
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::WEST);
                }
                break;
            case tt_low_latency_routing_vector::FORWARD_NORTH:
                if constexpr (router_direction == eth_chan_directions::NORTH) {
                    issue_local_write();
                } else {
                    if (*next_router_space < 1) {
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::NORTH);
                }
                break;
            case tt_low_latency_routing_vector::FORWARD_SOUTH:
                if constexpr (router_direction == eth_chan_directions::SOUTH) {
                    issue_local_write();
                } else {
                    if (*next_router_space < 1) {
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::SOUTH);
                }
                break;
            case tt_low_latency_routing_vector::FORWARD_NORTH | tt_low_latency_routing_vector::FORWARD_SOUTH:
                if constexpr (router_direction == eth_chan_directions::SOUTH) {
                    issue_local_write();
                    if (*next_router_space < 1) {
                        route_vector[hop_index] = hop_cmd & ~tt_low_latency_routing_vector::FORWARD_SOUTH;
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::NORTH);
                } else {
                    issue_local_write();
                    if (*next_router_space < 1) {
                        route_vector[hop_index] = hop_cmd & ~tt_low_latency_routing_vector::FORWARD_NORTH;
                        return;
                    }
                    issue_forward(hop_index, eth_chan_directions::SOUTH);
                }
                break;
            default: __builtin_unreachable();
        }

        advance_out_rdptr<fvc_mode>(1);
        slots_cleared = 1;
        packet_words_remaining = 0;
    }
#else
    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline uint32_t push_data_to_eth_router() {
        if (*next_router_space < 1) {
            return 0;
        }
        if (slots_cleared) {
            flush_async_writes<fvc_mode, true>();
        }
        // need to investigate this.
        // nop (and how many) affects performance.
        // Also flush_async_writes<fvc_mode, false>() affects performance positively.
        asm("nop");
        // asm("nop");
        // asm("nop");
        // asm("nop");

        uint32_t dest_id = ((volatile uint32_t*)packet_header)
            [(offsetof(packet_header_t, routing) + offsetof(tt_routing, dst_mesh_id)) / sizeof(uint32_t)];
        if (last_dest_id != dest_id) {
            dest_addr = get_next_hop_router_noc_xy();
            last_dest_id = dest_id;
        }
        uint64_t buffer_wr_addr =
            get_noc_addr_helper(dest_addr, remote_buffer_slot_start[remote_wrptr[remote_wrptr_direction]]);
        // Instead of sending the packet size (packet_words_remaining * PACKET_WORD_SIZE_BYTES) which
        // may be less than the full slot, send the full slot.
        noc_async_write_one_packet(
            get_local_buffer_read_addr(), buffer_wr_addr, FABRIC_ROUTER_BUF_SLOT_SIZE, noc_index);
        advance_remote_wrptr(1);
        advance_out_rdptr<fvc_mode>(1);
        uint64_t push_addr = get_noc_addr_helper(dest_addr, router_push_addr);
        noc_inline_dw_write<true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);

        *update_router_space = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t words_available = packet_words_remaining;
        slots_cleared += 1;
        packet_words_remaining = 0;
        return words_available;
    }

    FORCE_INLINE void issue_async_write() {
        uint32_t addr_l = packet_header->session.target_offset_l;
        uint32_t addr_h = packet_header->session.target_offset_h;
        noc_async_write_one_packet(
            get_local_buffer_read_addr() + PACKET_HEADER_SIZE_BYTES,
            get_noc_addr_helper(addr_h, addr_l),
            packet_size_bytes - PACKET_HEADER_SIZE_BYTES);
        advance_out_rdptr(1);
        slots_cleared += 1;
        packet_words_remaining = 0;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER, uint32_t router_direction = 0>
    FORCE_INLINE void process_inbound_packet() {
        if (for_local_chip) {
            if (slots_cleared) {
                flush_async_writes<fvc_mode, true>();
            }
            uint32_t command = packet_header->session.command;
            if (command & ASYNC_WR) {
                issue_async_write();
                // for fused command issue the atomic inc before invalidating the current packet
                if (command & ATOMIC_INC) {
                    uint64_t noc_addr = get_noc_addr_helper(
                        packet_header->packet_parameters.async_wr_atomic_parameters.noc_xy,
                        packet_header->packet_parameters.async_wr_atomic_parameters.l1_offset);
                    noc_fast_atomic_increment(
                        noc_index,
                        NCRISC_AT_CMD_BUF,
                        noc_addr,
                        NOC_UNICAST_WRITE_VC,
                        packet_header->packet_parameters.async_wr_atomic_parameters.increment,
                        31,
                        false);
                }
            } else if (command & ATOMIC_INC) {
                uint64_t noc_addr =
                    get_noc_addr_helper(packet_header->session.target_offset_h, packet_header->session.target_offset_l);
                noc_fast_atomic_increment(
                    noc_index,
                    NCRISC_AT_CMD_BUF,
                    noc_addr,
                    NOC_UNICAST_WRITE_VC,
                    packet_header->packet_parameters.atomic_parameters.increment,
                    packet_header->packet_parameters.atomic_parameters.wrap_boundary,
                    false);

                packet_words_remaining = 0;
                advance_out_rdptr(1);
                free_receiver_buffer_space(1);
            }
        } else {
            // push to next hop.
            push_data_to_eth_router<fvc_mode>();
        }
    }
#endif
    template <uint8_t fvc_mode = FVC_MODE_ROUTER, bool noc_flush = true>
    FORCE_INLINE void flush_async_writes() {
        if constexpr (noc_flush == true) {
            noc_async_writes_flushed();
        }
        free_receiver_buffer_space<fvc_mode>(slots_cleared);
        slots_cleared = 0;
    }
};

static_assert(sizeof(fvc_inbound_push_state_t) % 4 == 0);

struct fvc_outbound_pull_state_t {
    uint8_t chan_num;
    uint8_t pad[3];
    uint32_t packet_in_progress;
    uint32_t packet_words_remaining;
    uint32_t fvc_out_rdptr;
    uint32_t fvc_pull_wrptr;
    uint32_t buffer_size;
    uint32_t buffer_size_2x;
    uint32_t buffer_start;
    uint32_t remote_buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t total_words_to_forward;
    uint32_t words_sent_remote_update;
    volatile uint32_t* sender_buffer_space;
    volatile uint32_t* update_sender_buffer_space;
    volatile uint32_t* receiver_buffer_space;
    volatile uint32_t* update_receiver_buffer_space;

    FORCE_INLINE uint32_t get_num_words_free() { return *sender_buffer_space; }

    FORCE_INLINE uint32_t get_remote_num_words_free() { return *receiver_buffer_space; }

    inline void reset_buffer_space(uint32_t buf_size_words) {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        volatile uint32_t* ptr =
            reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        *ptr = buf_size_words;
        ptr = reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(2, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        *ptr = buf_size_words;
    }

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words) {
        uint32_t words = sizeof(fvc_outbound_pull_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        buffer_size_2x = data_buf_size_words * 2;
        remote_buffer_start = data_buf_start + buffer_size * PACKET_WORD_SIZE_BYTES;
        words_sent_remote_update = (STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        sender_buffer_space =
            reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_sender_buffer_space =
            reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        receiver_buffer_space =
            reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        update_receiver_buffer_space =
            reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        reset_buffer_space(data_buf_size_words);
    }

    FORCE_INLINE uint32_t words_before_buffer_wrap(uint32_t ptr) { return buffer_size - ptr; }

    FORCE_INLINE uint32_t words_before_local_buffer_wrap() { return buffer_size - fvc_pull_wrptr; }

    FORCE_INLINE uint32_t get_local_buffer_pull_addr() {
        return buffer_start + (fvc_pull_wrptr * PACKET_WORD_SIZE_BYTES);
    }

    FORCE_INLINE uint32_t get_local_buffer_read_addr() {
        return buffer_start + (fvc_out_rdptr * PACKET_WORD_SIZE_BYTES);
    }

    FORCE_INLINE void advance_pull_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_pull_wrptr + num_words;
        if (temp >= buffer_size) {
            temp -= buffer_size;
        }
        fvc_pull_wrptr = temp;
        *update_sender_buffer_space = (-num_words) << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    FORCE_INLINE void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size) {
            temp -= buffer_size;
        }
        fvc_out_rdptr = temp;
        *update_receiver_buffer_space = (-num_words) << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    FORCE_INLINE void register_pull_data(uint32_t num_words_to_pull) {
        pull_words_in_flight += num_words_to_pull;
        advance_pull_wrptr(num_words_to_pull);
        packet_words_remaining -= num_words_to_pull;
    }

    FORCE_INLINE void register_move_data(uint32_t num_words_to_move) {
        advance_pull_wrptr(num_words_to_move);
        packet_words_remaining -= num_words_to_move;
        total_words_to_forward += num_words_to_move;
    }

    template <bool barrier>
    inline uint32_t forward_data_from_fvc_buffer() {
        uint32_t words_to_forward = total_words_to_forward;
        uint32_t remote_fvc_buffer_space = get_remote_num_words_free();
        if (remote_fvc_buffer_space == 0) {
            // nothing to send or no space in receiver.
            if constexpr (barrier == true) {
                noc_async_read_barrier();
            }
            return 0;
        }

        if (remote_fvc_buffer_space < words_to_forward) {
            words_to_forward = remote_fvc_buffer_space;
        }

        // decrement total number of words that need to be sent over ethernet
        // by the amount we are sending in this iteration.
        total_words_to_forward -= words_to_forward;
        uint32_t src_addr = 0;
        uint32_t dest_addr = 0;  // should be second half of fvc buffer.
        uint32_t words_remaining = words_to_forward;
        uint32_t num_words_before_wrap = words_before_buffer_wrap(fvc_out_rdptr);
        uint32_t chunk_to_forward = std::min(num_words_before_wrap, words_remaining);
        while (words_remaining) {
            invalidate_l1_cache();
            src_addr = get_local_buffer_read_addr();
            dest_addr = src_addr - buffer_start + remote_buffer_start;
            if constexpr (barrier) {
                noc_async_read_barrier();
            }
            internal_::eth_send_packet(
                0, src_addr / PACKET_WORD_SIZE_BYTES, dest_addr / PACKET_WORD_SIZE_BYTES, chunk_to_forward);
            advance_out_rdptr(chunk_to_forward);
            words_remaining -= chunk_to_forward;
            chunk_to_forward = words_remaining;
        }

        // send word credits to receiver
        eth_write_remote_reg((uint32_t)words_sent_remote_update, words_to_forward << REMOTE_DEST_BUF_WORDS_FREE_INC);

        return words_to_forward;
    }

    inline uint32_t get_num_words_to_pull(volatile pull_request_t* pull_request) {
        uint32_t num_words_to_pull = num_words_available_to_pull(pull_request);
        uint32_t num_words_before_wrap = words_before_pull_buffer_wrap(pull_request->buffer_size, pull_request->rd_ptr);

        num_words_to_pull = std::min(num_words_to_pull, num_words_before_wrap);
        uint32_t fvc_buffer_space = get_num_words_free();
        num_words_to_pull = std::min(num_words_to_pull, fvc_buffer_space);

        if (num_words_to_pull == 0) {
            return 0;
        }

        uint32_t fvc_space_before_wptr_wrap = words_before_local_buffer_wrap();
        num_words_to_pull = std::min(num_words_to_pull, fvc_space_before_wptr_wrap);

        num_words_to_pull = std::min(num_words_to_pull, buffer_size / 2);

        return num_words_to_pull;
    }
    template <bool packetized = true>
    FORCE_INLINE uint32_t
    pull_data_to_fvc_buffer(volatile pull_request_t* pull_request, volatile pull_request_t* header) {
        if (packet_in_progress == 0) {
            uint32_t size = pull_request->size;
            if constexpr (packetized == false) {
                packet_words_remaining = PACKET_HEADER_SIZE_WORDS + ((size + PACKET_WORD_SIZE_BYTES - 1) >> 4);
                if (move_data_to_fvc_buffer<false>(header)) {
                    packet_in_progress = 1;
                } else {
                    return 0;
                }
            } else {
                packet_words_remaining = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
                packet_in_progress = 1;
            }
        }

        uint32_t num_words_to_pull = get_num_words_to_pull(pull_request);
        if (num_words_to_pull == 0) {
            return 0;
        }

        uint32_t rd_offset = get_rd_ptr_offset_words((pull_request_t*)pull_request);
        uint64_t src_addr = pull_request->buffer_start + (rd_offset * PACKET_WORD_SIZE_BYTES);
        uint32_t fvc_addr = get_local_buffer_pull_addr();

        // pull_data_from_remote();
        noc_async_read(src_addr, fvc_addr, num_words_to_pull * PACKET_WORD_SIZE_BYTES);
        register_pull_data(num_words_to_pull);
        pull_request->rd_ptr = advance_ptr(pull_request->buffer_size, pull_request->rd_ptr, num_words_to_pull);
        pull_request->words_read += num_words_to_pull;

        return num_words_to_pull;
    }

    template <bool packet = true>
    FORCE_INLINE uint32_t move_data_to_fvc_buffer(volatile pull_request_t* pull_request) {
        if constexpr (packet == true) {
            if (packet_in_progress == 0) {
                packet_words_remaining = PACKET_HEADER_SIZE_WORDS;
                packet_in_progress = 1;
            }
        }

        // if fvc does not have enough space, try again later.
        if (get_num_words_free() < PACKET_HEADER_SIZE_WORDS) {
            return 0;
        }

        uint32_t fvc_space_before_wptr_wrap = words_before_local_buffer_wrap();
        uint32_t* fvc_addr = (uint32_t*)get_local_buffer_pull_addr();
        uint32_t* src = (uint32_t*)pull_request;

        switch (fvc_space_before_wptr_wrap) {
            case 1:
                fvc_addr[0] = src[0];
                fvc_addr[1] = src[1];
                fvc_addr[2] = src[2];
                fvc_addr[3] = src[3];
                fvc_addr = (uint32_t*)buffer_start;
                fvc_addr[0] = src[4];
                fvc_addr[1] = src[5];
                fvc_addr[2] = src[6];
                fvc_addr[3] = src[7];
                fvc_addr[4] = src[8];
                fvc_addr[5] = src[9];
                fvc_addr[6] = src[10];
                fvc_addr[7] = src[11];
                break;
            case 2:
                // uint32_t i = 0;
                for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_WORDS - 1) * PACKET_WORD_SIZE_BYTES / 4; i++) {
                    fvc_addr[i] = src[i];
                }
                fvc_addr = (uint32_t*)buffer_start;
                fvc_addr[0] = src[8];
                fvc_addr[1] = src[9];
                fvc_addr[2] = src[10];
                fvc_addr[3] = src[11];
                break;
            default:
                for (uint32_t i = 0; i < PACKET_HEADER_SIZE_BYTES / 4; i++) {
                    fvc_addr[i] = src[i];
                }
        }

        register_move_data(PACKET_HEADER_SIZE_WORDS);
        return PACKET_HEADER_SIZE_WORDS;
    }
};

static_assert(sizeof(fvc_outbound_pull_state_t) % 4 == 0);

enum ProcessingFlags : uint8_t {
    UCAST_DEST = 1,
    MCAST_DEST = 2,
    NOT_DEST = 3,
};
// FVC Producer holds data that needs to be forwarded to other destinations.
// This producer receives data over ethernet from neighboring chip.
// Data in the producer is either destined for local chip, or has to make a noc hop
// to ethernet port enroute to final destination.
// FVC producer buffer issues pull requests to other entities in the fabric node to
// pull data from Producer buffer. Pull requests can be made to next router/consumer buffer in the route
// direction, socket receiver/consumer buffer, center worker/consumer buffer.
// Which ever entity receives the pull request is responsible draining the required amount of data from
// FVC Producer.
struct fvc_inbound_pull_state_t {
    chan_payload_ptr inbound_wrptr;
    chan_payload_ptr inbound_rdptr;
    uint32_t my_id;
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t packet_processing_flags;
    uint8_t mcast_direction;
    uint32_t mcast_router_noc_xy;
    uint32_t words_inbound;
    uint32_t words_cleared;
    uint32_t packet_words_remaining;
    uint32_t hop_words_remaining;
    uint32_t fvc_out_rdptr;
    uint32_t buffer_size;
    uint32_t buffer_size_2x;
    uint32_t buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_to_forward;
    bool curr_packet_valid;
    bool packet_corrupted;
    uint64_t packet_timestamp;
    uint64_t packet_dest;
    uint64_t hop_dest;
    packet_header_t current_packet_header;
    uint32_t* packet_id;
    volatile uint32_t* words_received;
    uint32_t* words_received_local_update;
    uint32_t update_sender_buffer_space;
    uint32_t update_receiver_buffer_space;
    uint64_t pull_req_dest_address;
    bool pull_request_pending;
    uint8_t padding[3];

    inline void reset_words_received() {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        volatile uint32_t* ptr =
            reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        *ptr = 0;
    }

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words) {
        uint32_t words = sizeof(fvc_inbound_pull_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        my_id = routing_table->my_device_id << 16 | routing_table->my_mesh_id;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        buffer_size_2x = data_buf_size_words * 2;
        words_received =
            reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        words_received_local_update =
            reinterpret_cast<uint32_t*>(STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        update_sender_buffer_space = (STREAM_REG_ADDR(1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        update_receiver_buffer_space = (STREAM_REG_ADDR(2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        reset_words_received();
        packet_id = (uint32_t*)&current_packet_header.routing.dst_mesh_id;
        tt::tt_fabric::chan_id_t my_chan = routing_table->intra_mesh_table.dest_entry[routing_table->my_device_id];
        tt::tt_fabric::chan_id_t mcast_channel = 0;
        if (routing_table->port_direction.directions[eth_chan_directions::EAST] == my_chan) {
            mcast_channel = routing_table->port_direction.directions[eth_chan_directions::WEST];
            mcast_direction = 1;
        } else if (routing_table->port_direction.directions[eth_chan_directions::WEST] == my_chan) {
            mcast_channel = routing_table->port_direction.directions[eth_chan_directions::EAST];
            mcast_direction = 0;
        } else if (routing_table->port_direction.directions[eth_chan_directions::NORTH] == my_chan) {
            mcast_channel = routing_table->port_direction.directions[eth_chan_directions::SOUTH];
            mcast_direction = 3;
        } else if (routing_table->port_direction.directions[eth_chan_directions::SOUTH] == my_chan) {
            mcast_channel = routing_table->port_direction.directions[eth_chan_directions::NORTH];
            mcast_direction = 2;
        }
        mcast_router_noc_xy = eth_chan_to_noc_xy[noc_index][mcast_channel];
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr, uint32_t inc) {
        uint32_t temp = ptr + inc;
        if (temp >= buffer_size) {
            temp -= buffer_size;
        }
        return temp;
    }

    inline void advance_local_wrptr(uint32_t num_words) {
        inbound_wrptr.ptr = inc_ptr_with_wrap(inbound_wrptr.ptr, num_words);
        words_inbound += num_words;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size) {
            temp -= buffer_size;
        }
        fvc_out_rdptr = temp;
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            words_inbound -= num_words;
        }
    }

    FORCE_INLINE uint32_t words_before_buffer_wrap(uint32_t ptr) { return buffer_size - ptr; }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE uint32_t get_num_words_available() {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            uint32_t new_words = *words_received;
            if (new_words == 0) {
                return words_inbound;
            }
            *words_received_local_update = (-new_words) << REMOTE_DEST_BUF_WORDS_FREE_INC;
            words_inbound += new_words;
            free_sender_buffer_space(new_words);
            return words_inbound;
        } else {
            return words_inbound;
        }
    }

    FORCE_INLINE
    uint32_t get_num_words_free() { return buffer_size - words_inbound; }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE bool get_curr_packet_valid() {
        if (!curr_packet_valid) {
            if (get_num_words_available<fvc_mode>() >= PACKET_HEADER_SIZE_WORDS) {
                // Wait for a full packet header to arrive before advancing to next packet.
                this->advance_next_packet<fvc_mode>();
            }
        }
        return this->curr_packet_valid;
    }

    FORCE_INLINE uint32_t get_local_buffer_read_addr() {
        return buffer_start + (fvc_out_rdptr * PACKET_WORD_SIZE_BYTES);
    }

    FORCE_INLINE uint32_t get_local_buffer_write_addr() {
        return buffer_start + (inbound_wrptr.ptr * PACKET_WORD_SIZE_BYTES);
    }

    FORCE_INLINE uint32_t words_before_local_buffer_wrap() { return buffer_size - inbound_wrptr.ptr; }

    FORCE_INLINE void free_sender_buffer_space(uint32_t words) {
        // send received word credits to receiver
        eth_write_remote_reg((uint32_t)update_sender_buffer_space, words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void free_receiver_buffer_space(uint32_t words) {
        if constexpr (fvc_mode == FVC_MODE_ROUTER) {
            // send received word credits to receiver
            eth_write_remote_reg((uint32_t)update_receiver_buffer_space, words << REMOTE_DEST_BUF_WORDS_FREE_INC);
        } else {
            words_inbound -= words;
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void advance_next_packet() {
        // The following code makes the following assumptions regarding packet header structure
        // 1 - packet_parameters is always the first word of the packet header, and doesn't span across packet word
        // boundary.
        // 2 - routing is always the last word of the packet header doesn't span across packet word boundary.
        static_assert(
            offsetof(packet_header_t, packet_parameters) == 0 && sizeof(packet_params) <= PACKET_WORD_SIZE_BYTES);
        static_assert(
            offsetof(packet_header_t, routing) >= (PACKET_HEADER_SIZE_BYTES - PACKET_WORD_SIZE_BYTES) &&
            offsetof(packet_header_t, routing) % PACKET_WORD_SIZE_BYTES + sizeof(tt_routing) <= PACKET_WORD_SIZE_BYTES);
        tt_l1_ptr uint32_t* packet_header_ptr = (uint32_t*)&current_packet_header;
        volatile tt_l1_ptr uint32_t* next_header_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(get_local_buffer_read_addr());
        // TODO: Should we just extract the specific field we want here (mcast_params)
        packet_params* next_packet_params_ptr = (packet_params*)(next_header_ptr);
        tt_routing* next_routing_ptr;
        uint32_t words_before_wrap = words_before_buffer_wrap(fvc_out_rdptr);
        uint32_t dwords_to_copy = PACKET_HEADER_SIZE_BYTES / 4;
        if (words_before_wrap < PACKET_HEADER_SIZE_WORDS) {
            // Header spans buffer end.
            // Needs to be copied in two steps.
            uint32_t dwords_before_wrap = words_before_wrap * PACKET_WORD_SIZE_BYTES / 4;
            uint32_t dwords_after_wrap = dwords_to_copy - dwords_before_wrap;
            for (uint32_t i = 0; i < dwords_before_wrap; i++) {
                packet_header_ptr[i] = next_header_ptr[i];
            }
            next_header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(buffer_start);
            for (uint32_t i = 0; i < dwords_after_wrap; i++) {
                packet_header_ptr[i + dwords_before_wrap] = next_header_ptr[i];
            }
            next_routing_ptr =
                (tt_routing*)(next_header_ptr + packet_header_routing_offset_dwords - dwords_before_wrap);
        } else {
#pragma GCC unroll 12
            for (uint32_t i = 0; i < dwords_to_copy; i++) {
                packet_header_ptr[i] = next_header_ptr[i];
            }
            next_routing_ptr = (tt_routing*)(next_header_ptr + packet_header_routing_offset_dwords);
        }

        this->packet_words_remaining =
            (this->current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
        if (tt_fabric_is_header_valid(&current_packet_header)) {
            this->curr_packet_valid = true;
            if (packet_is_for_local_chip()) {
                if (packet_mcast_is_required()) {
                    // If its mcast packet, update the hop count.
                    // Packet arrival on this router accounts for 1 hop.
                    // Decrement respective direction hop count and determine
                    // whether mcast needs further hops.
                    update_mcast_hops(next_packet_params_ptr, next_routing_ptr);
                }
                // After updating mcast hop counts, we check whether the mcast packet still qualifies to be
                // an mcast packet.
                packet_processing_flags =
                    packet_mcast_is_active() ? ProcessingFlags::MCAST_DEST : ProcessingFlags::UCAST_DEST;
            } else {
                if (packet_mcast_is_active()) {
                    // Mcast packets have dest dev/mesh id set to the device where mcast starts.
                    // Hence packet_is_for_local_chip() returns true only for the first mcast target device.
                    // All other devices, need to check for mcast active flag to determine if they should consume
                    // the data or not.
                    // Any device that receives a packet with mcast active flag set consumes the data and forwards
                    // as well if its not the last hop of mcast group.

                    // If mcast is active, update the hop count.
                    // Decrement hop count.
                    update_mcast_hops(next_packet_params_ptr, next_routing_ptr);
                    // After decrementing hop count, check whether mcast is stil active.
                    // If mcast has been deactivated here, that means this chip is the last hop for mcast packet.
                    // If so, we service the last hop as unicast dest.
                    // Otherwise, we handle as mcast dest, which means packet is consumed locally as well as
                    // forwarded to next hop in mcast direction.
                    packet_processing_flags =
                        packet_mcast_is_active() ? ProcessingFlags::MCAST_DEST : ProcessingFlags::UCAST_DEST;
                } else {
                    // We are here for one of 2 reasons.
                    // 1 - Packet is not meant for this chip.
                    // 2 - Packet is not under active mcast.
                    packet_processing_flags = ProcessingFlags::NOT_DEST;
                }
            }
        } else {
            this->packet_corrupted = true;
        }
    }

    inline void copy_header(pull_request_t* req) {
        uint32_t* dst = (uint32_t*)req;
        uint32_t* src = (uint32_t*)&current_packet_header;
        for (uint32_t i = 0; i < sizeof(pull_request_t) / 4; i++) {
            dst[i] = src[i];
        }
    }

    uint32_t get_next_hop_router_noc_xy() {
        uint32_t dst_mesh_id = current_packet_header.routing.dst_mesh_id;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            uint32_t next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        } else {
            uint32_t dst_device_id = current_packet_header.routing.dst_dev_id;
            uint32_t next_port = routing_table->intra_mesh_table.dest_entry[dst_device_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER, bool socket_mode = false>
    inline uint32_t pull_data_from_fvc_buffer() {
        uint32_t words_available = get_num_words_available<fvc_mode>();
        words_available = std::min(words_available, packet_words_remaining);
        bool try_sending_pull_request = false;
        if (packet_in_progress == 0) {
            if (current_packet_header.routing.flags == INLINE_FORWARD) {
                copy_header((pull_request_t*)&local_pull_request->pull_request);
            } else {
                local_pull_request->pull_request.rd_ptr = fvc_out_rdptr;
                local_pull_request->pull_request.size = current_packet_header.routing.packet_size_bytes;
                local_pull_request->pull_request.buffer_size = buffer_size;
                local_pull_request->pull_request.buffer_start = xy_local_addr + buffer_start;
                local_pull_request->pull_request.words_written = 0;
                local_pull_request->pull_request.words_read = 0;
                local_pull_request->pull_request.ack_addr =
                    xy_local_addr + (uint32_t)&local_pull_request->pull_request.words_read;
                local_pull_request->pull_request.flags = FORWARD;
            }

            words_cleared = 0;
            packet_in_progress = 1;

            // issue noc write to noc target of pull request.
            this->pull_req_dest_address =
                socket_mode == false
                    ? get_noc_addr_helper(get_next_hop_router_noc_xy(), FABRIC_ROUTER_REQ_QUEUE_START)
                    : get_noc_addr_helper(
                          current_packet_header.session.target_offset_h, current_packet_header.session.target_offset_l);
            tt_fabric_reserve_pull_request_slot(this->pull_req_dest_address, local_pull_request);
            try_sending_pull_request = true;
        } else if (this->pull_request_pending) {
            try_sending_pull_request = true;
        } else {
            // pull_request.rd_ptr is updated by remote puller when data is read out of producer's local buffer.
            // it is used to determine when it it safe to reclaim local buffer memory for more data.
            uint32_t curr_words_read = local_pull_request->pull_request.words_read;
            uint32_t words_to_clear = curr_words_read - words_cleared;
            if (words_to_clear) {
                free_receiver_buffer_space<fvc_mode>(words_to_clear);
                words_cleared += words_to_clear;
            }
            if (packet_words_remaining) {
                if (words_available) {
                    advance_out_rdptr<fvc_mode>(words_available);
                    // packet_dest is returned by tt_fabric_send_pull_request() as the address of request q entry +
                    // pull_request.words_written.
                    local_pull_request->pull_request.words_written += words_available;
                    noc_inline_dw_write(hop_dest, local_pull_request->pull_request.words_written);
                    packet_words_remaining -= words_available;
                }
            } else if (curr_words_read == local_pull_request->pull_request.words_written) {
                // all data has been pulled and cleared from local buffer
                packet_in_progress = 0;
                curr_packet_valid = false;
            }
        }

        if (try_sending_pull_request) {
            bool can_send_pull_request =
                tt_fabric_check_pull_request_slot<false>(this->pull_req_dest_address, local_pull_request, local_pull_request->wrptr.ptr);
            if (!can_send_pull_request) {
                this->pull_request_pending = true;
                return 0;
            }

            if (current_packet_header.routing.flags == INLINE_FORWARD) {
                words_cleared = words_available;
            } else {
                local_pull_request->pull_request.words_written = words_available;
            }

            hop_dest = tt_fabric_send_pull_request(this->pull_req_dest_address, local_pull_request);
            this->pull_request_pending = false;

            packet_words_remaining -= words_available;
            advance_out_rdptr<fvc_mode>(words_available);

            if (current_packet_header.routing.flags == INLINE_FORWARD) {
                curr_packet_valid = false;
                packet_in_progress = 0;
                flush_async_writes<fvc_mode>();
            }
        }

        return words_available;
    }

    template <bool resample = true>
    FORCE_INLINE void issue_async_write() {
        if constexpr (resample) {
            get_num_words_available();
        }
        uint32_t words_available = words_inbound;
        words_available = std::min(words_available, packet_words_remaining);
        words_available = std::min(words_available, words_before_buffer_wrap(fvc_out_rdptr));
        if (words_available) {
            noc_async_write(get_local_buffer_read_addr(), packet_dest, words_available * PACKET_WORD_SIZE_BYTES);
            packet_words_remaining -= words_available;
            advance_out_rdptr(words_available);
            words_cleared += words_available;
            packet_dest += words_available * PACKET_WORD_SIZE_BYTES;
        }
    }

    FORCE_INLINE bool packet_is_for_local_chip() { return my_id == *packet_id; }

    inline bool packet_mcast_is_active() { return (current_packet_header.routing.flags & MCAST_ACTIVE) != 0; }

    inline bool packet_mcast_is_required() { return (current_packet_header.routing.flags & MCAST_DATA) != 0; }

    inline void update_mcast_hops(packet_params* packet_parameters, tt_routing* routing) {
        uint32_t hop_count = 0;
        if (mcast_direction == 0) {
            hop_count = current_packet_header.packet_parameters.mcast_parameters.east;
            if (hop_count) {
                hop_count--;
                current_packet_header.packet_parameters.mcast_parameters.east = hop_count;
                packet_parameters->mcast_parameters.east = hop_count;
            }
        } else if (mcast_direction == 1) {
            hop_count = current_packet_header.packet_parameters.mcast_parameters.west;
            if (hop_count) {
                hop_count--;
                current_packet_header.packet_parameters.mcast_parameters.west = hop_count;
                packet_parameters->mcast_parameters.west = hop_count;
            }
        } else if (mcast_direction == 2) {
            hop_count = current_packet_header.packet_parameters.mcast_parameters.north;
            if (hop_count) {
                hop_count--;
                current_packet_header.packet_parameters.mcast_parameters.north = hop_count;
                packet_parameters->mcast_parameters.north = hop_count;
            }
        } else if (mcast_direction == 3) {
            hop_count = current_packet_header.packet_parameters.mcast_parameters.south;
            if (hop_count) {
                hop_count--;
                current_packet_header.packet_parameters.mcast_parameters.south = hop_count;
                packet_parameters->mcast_parameters.south = hop_count;
            }
        }
        if (hop_count == 0) {
            // on last hop clear the mcast flag bits.
            // last hop treats packet as normal ucast async write.
            current_packet_header.routing.flags &= ~(MCAST_ACTIVE | MCAST_DATA);
            routing->flags &= ~(MCAST_ACTIVE | MCAST_DATA);
        } else {
            current_packet_header.routing.flags |= MCAST_ACTIVE;
            routing->flags |= MCAST_ACTIVE;
            // calculate new header checksum after mcast updates.
            tt_fabric_add_header_checksum(&current_packet_header);
            // copy new checksum to packet header in fvc buffer.
            packet_parameters->misc_parameters.words[0] =
                current_packet_header.packet_parameters.misc_parameters.words[0];
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline uint32_t process_mcast_packet() {
        uint32_t words_processed = 0;
        uint32_t words_available = get_num_words_available();
        words_available = std::min(words_available, packet_words_remaining);
        bool try_sending_pull_request = false;
        if (current_packet_header.session.command & ASYNC_WR) {
            words_processed = words_available;
            if (packet_in_progress == 0) {
                local_pull_request->pull_request.rd_ptr = fvc_out_rdptr;
                local_pull_request->pull_request.size = current_packet_header.routing.packet_size_bytes;
                local_pull_request->pull_request.buffer_size = buffer_size;
                local_pull_request->pull_request.buffer_start = xy_local_addr + buffer_start;
                local_pull_request->pull_request.words_written = 0;
                local_pull_request->pull_request.words_read = 0;
                words_cleared = 0;
                local_pull_request->pull_request.ack_addr =
                    xy_local_addr + (uint32_t)&local_pull_request->pull_request.words_read;
                local_pull_request->pull_request.flags = FORWARD;

                // subtract the header words. Remaining words are the data to be written to packet_dest.
                // Remember to account for trailing bytes which may not be a full packet word.
                packet_in_progress = 1;

                // issue noc write to noc target of pull request.
                // figure out next hop for mcast forwarding
                this->pull_req_dest_address = get_noc_addr_helper(mcast_router_noc_xy, FABRIC_ROUTER_REQ_QUEUE_START);
                tt_fabric_reserve_pull_request_slot(this->pull_req_dest_address, local_pull_request);
                try_sending_pull_request = true;
            } else if (this->pull_request_pending) {
                try_sending_pull_request = true;
            } else {
                noc_async_writes_flushed();
                // pull_request.rd_ptr is updated by remote puller when data is read out of producer's local buffer.
                // it is used to determine when it it safe to reclaim local buffer memory for more data.
                uint32_t curr_words_read = local_pull_request->pull_request.words_read;
                uint32_t words_to_clear = curr_words_read - words_cleared;
                if (words_to_clear) {
                    free_receiver_buffer_space<fvc_mode>(words_to_clear);
                    words_cleared += words_to_clear;
                }

                if (packet_words_remaining) {
                    if (words_available) {
                        uint32_t local_words_available =
                            std::min(words_available, words_before_buffer_wrap(fvc_out_rdptr));
                        // write available data till end of input buffer
                        noc_async_write(
                            get_local_buffer_read_addr(), packet_dest, local_words_available * PACKET_WORD_SIZE_BYTES);
                        advance_out_rdptr(local_words_available);
                        packet_dest += local_words_available * PACKET_WORD_SIZE_BYTES;
                        local_words_available = words_available - local_words_available;
                        // write remaining available data from beginning of buffer
                        if (local_words_available) {
                            noc_async_write(
                                get_local_buffer_read_addr(),
                                packet_dest,
                                local_words_available * PACKET_WORD_SIZE_BYTES);
                            advance_out_rdptr(local_words_available);
                            packet_dest += local_words_available * PACKET_WORD_SIZE_BYTES;
                        }

                        // hop_dest is returned by tt_fabric_send_pull_request() as the address of request q entry +
                        // pull_request.wr_ptr.
                        local_pull_request->pull_request.words_written += words_available;
                        noc_inline_dw_write(hop_dest, local_pull_request->pull_request.words_written);
                        packet_words_remaining -= words_available;
                    }
                } else if (curr_words_read == local_pull_request->pull_request.words_written) {
                    // for fused command issue the atomic inc before invalidating the current packet
                    if (current_packet_header.session.command & ATOMIC_INC) {
                        uint64_t noc_addr = get_noc_addr_helper(
                            current_packet_header.packet_parameters.async_wr_atomic_parameters.noc_xy,
                            current_packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset);
                        noc_fast_atomic_increment(
                            noc_index,
                            NCRISC_AT_CMD_BUF,
                            noc_addr,
                            NOC_UNICAST_WRITE_VC,
                            current_packet_header.packet_parameters.async_wr_atomic_parameters.increment,
                            31,
                            false);
                    }
                    // all data has been pulled and cleared from local buffer
                    packet_in_progress = 0;
                    curr_packet_valid = false;
                    packet_timestamp = get_timestamp();
                }
            }

            if (try_sending_pull_request) {
                bool can_send_pull_request =
                    tt_fabric_check_pull_request_slot<false>(this->pull_req_dest_address, local_pull_request, local_pull_request->wrptr.ptr);
                if (!can_send_pull_request) {
                    this->pull_request_pending = true;
                    return 0;
                }

                local_pull_request->pull_request.words_written = words_available;
                hop_dest = tt_fabric_send_pull_request(this->pull_req_dest_address, local_pull_request);
                this->pull_request_pending = false;

                packet_words_remaining -= words_available;
                packet_dest = get_noc_addr_helper(
                    current_packet_header.session.target_offset_h, current_packet_header.session.target_offset_l);

                advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                words_available -= PACKET_HEADER_SIZE_WORDS;

                uint32_t local_words_available = std::min(words_available, words_before_buffer_wrap(fvc_out_rdptr));
                // write available data till end of input buffer
                if (local_words_available) {
                    // need to check local_words_available > 0 since it is possible that we only received the packet
                    // header so far, and words_available == 0 after words_available -= PACKET_HEADER_SIZE_WORDS above.
                    noc_async_write(
                        get_local_buffer_read_addr(), packet_dest, local_words_available * PACKET_WORD_SIZE_BYTES);
                    advance_out_rdptr(local_words_available);
                    packet_dest += local_words_available * PACKET_WORD_SIZE_BYTES;
                }
                local_words_available = words_available - local_words_available;
                // write remaining available data from beginning of buffer
                if (local_words_available) {
                    noc_async_write(
                        get_local_buffer_read_addr(), packet_dest, local_words_available * PACKET_WORD_SIZE_BYTES);
                    advance_out_rdptr(local_words_available);
                    packet_dest += local_words_available * PACKET_WORD_SIZE_BYTES;
                }
            }
        }
        return words_processed;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER, uint32_t router_direction = 0>
    inline uint32_t process_inbound_packet() {
        uint32_t words_processed = 0;
        if (packet_processing_flags == ProcessingFlags::UCAST_DEST) {
            if (current_packet_header.routing.flags == FORWARD) {
                if (current_packet_header.session.command & ASYNC_WR) {
                    if (packet_in_progress == 0) {
                        packet_dest = get_noc_addr_helper(
                            current_packet_header.session.target_offset_h,
                            current_packet_header.session.target_offset_l);
                        packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
                        advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                        words_cleared = PACKET_HEADER_SIZE_WORDS;
                        // subtract the header words. Remaining words are the data to be written to packet_dest.
                        // Remember to account for trailing bytes which may not be a full packet word.
                        issue_async_write<false>();
                        if (packet_words_remaining) {
                            packet_in_progress = 1;
                        } else {
                            flush_async_writes();
                            if (current_packet_header.session.command & ATOMIC_INC) {
                                uint64_t noc_addr = get_noc_addr_helper(
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.noc_xy,
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset);
                                noc_fast_atomic_increment(
                                    noc_index,
                                    NCRISC_AT_CMD_BUF,
                                    noc_addr,
                                    NOC_UNICAST_WRITE_VC,
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.increment,
                                    31,
                                    false);
                            }
                            curr_packet_valid = false;
                        }
                    } else {
                        flush_async_writes();
                        if (packet_words_remaining) {
                            issue_async_write<true>();
                        } else {
                            // for fused command issue the atomic inc before invalidating the current packet
                            if (current_packet_header.session.command & ATOMIC_INC) {
                                uint64_t noc_addr = get_noc_addr_helper(
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.noc_xy,
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset);
                                noc_fast_atomic_increment(
                                    noc_index,
                                    NCRISC_AT_CMD_BUF,
                                    noc_addr,
                                    NOC_UNICAST_WRITE_VC,
                                    current_packet_header.packet_parameters.async_wr_atomic_parameters.increment,
                                    31,
                                    false);
                            }
                            packet_in_progress = 0;
                            curr_packet_valid = false;
                        }
                    }
                } else if (current_packet_header.session.command == DSOCKET_WR) {
                    words_processed = pull_data_from_fvc_buffer<fvc_mode, true>();
                }
            } else if (current_packet_header.routing.flags == INLINE_FORWARD) {
                if (current_packet_header.session.command == SOCKET_CLOSE) {
                    words_processed = pull_data_from_fvc_buffer<fvc_mode, true>();
                } else {
                    uint64_t noc_addr = get_noc_addr_helper(
                        current_packet_header.session.target_offset_h, current_packet_header.session.target_offset_l);
                    noc_fast_atomic_increment(
                        noc_index,
                        NCRISC_AT_CMD_BUF,
                        noc_addr,
                        NOC_UNICAST_WRITE_VC,
                        current_packet_header.packet_parameters.atomic_parameters.increment,
                        current_packet_header.packet_parameters.atomic_parameters.wrap_boundary,
                        false);

                    packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
                    advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                    words_processed = PACKET_HEADER_SIZE_WORDS;
                    free_receiver_buffer_space(PACKET_HEADER_SIZE_WORDS);
                    curr_packet_valid = false;
                    packet_timestamp = get_timestamp();
                }
            }
        } else if (packet_processing_flags == ProcessingFlags::MCAST_DEST) {
            words_processed = process_mcast_packet();
        } else {
            pull_data_from_fvc_buffer<fvc_mode, false>();
        }
        return words_processed;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    FORCE_INLINE void flush_async_writes() {
        noc_async_writes_flushed();
        free_receiver_buffer_space<fvc_mode>(words_cleared);
        words_cleared = 0;
    }
};

static_assert(sizeof(fvc_inbound_pull_state_t) % 4 == 0);
struct fvcc_outbound_state_t {
    volatile chan_payload_ptr remote_rdptr;
    uint32_t remote_ptr_update_addr;
    volatile ctrl_chan_msg_buf*
        fvcc_buf;  // fvcc buffer that receives messages that need to be forwarded over ethernet.
    volatile ctrl_chan_sync_buf* fvcc_sync_buf;  // sync buffer to hold pointer updates sent over ethernet
    uint32_t remote_fvcc_buf_start;
    uint32_t out_rdptr;

    // Check if ethernet receiver fvcc on neighboring chip is full.
    bool is_remote_fvcc_full() {
        uint32_t rd_ptr = remote_rdptr.ptr_cleared;
        bool full = false;
        if (out_rdptr != rd_ptr) {
            uint32_t distance = out_rdptr >= rd_ptr ? out_rdptr - rd_ptr : out_rdptr + 2 * FVCC_BUF_SIZE - rd_ptr;
            full = distance >= FVCC_BUF_SIZE;
        }
        return full;
    }

    inline void init(uint32_t buf_start, uint32_t sync_buf_start, uint32_t remote_buf_start, uint32_t ptr_update_addr) {
        uint32_t words = sizeof(fvcc_outbound_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        fvcc_buf = (volatile ctrl_chan_msg_buf*)buf_start;
        fvcc_sync_buf = (volatile ctrl_chan_sync_buf*)sync_buf_start;
        remote_fvcc_buf_start = remote_buf_start;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr) { return (ptr + 1) & FVCC_PTR_MASK; }

    inline void advance_out_rdptr() { out_rdptr = inc_ptr_with_wrap(out_rdptr); }

    inline void advance_fvcc_rdptr() {
        uint32_t rd_ptr = remote_rdptr.ptr;
        while (rd_ptr != fvcc_buf->rdptr.ptr) {
            invalidate_l1_cache();
            uint32_t msg_index = fvcc_buf->rdptr.ptr & FVCC_SIZE_MASK;
            fvcc_buf->msg_buf[msg_index].packet_header.routing.flags = 0;
            fvcc_buf->rdptr.ptr = inc_ptr_with_wrap(fvcc_buf->rdptr.ptr);
        }
    }

    template <bool live = true>
    inline uint32_t forward_data_from_fvcc_buffer() {
        // If receiver ethernet fvcc is full, we cannot send more messages.
        if (is_remote_fvcc_full()) {
            return 0;
        }

        if (fvcc_buf->wrptr.ptr != out_rdptr) {
            // There are new messages to forward.
            uint32_t msg_index = out_rdptr & FVCC_SIZE_MASK;
            volatile packet_header_t* msg = &fvcc_buf->msg_buf[msg_index].packet_header;
            bool msg_valid = msg->routing.flags != 0;
            if (!msg_valid) {
                return 0;
            }

            uint32_t dest_addr =
                remote_fvcc_buf_start + offsetof(ctrl_chan_msg_buf, msg_buf) + msg_index * sizeof(packet_header_t);
            internal_::eth_send_packet(
                0,
                (uint32_t)msg / PACKET_WORD_SIZE_BYTES,
                dest_addr / PACKET_WORD_SIZE_BYTES,
                PACKET_HEADER_SIZE_WORDS);
            advance_out_rdptr();

            uint32_t* sync_ptr = (uint32_t*)&fvcc_sync_buf->ptr[msg_index];
            sync_ptr[0] = out_rdptr;
            sync_ptr[1] = 0;
            sync_ptr[2] = 0;
            sync_ptr[3] = out_rdptr;
            internal_::eth_send_packet(
                0, ((uint32_t)sync_ptr) / PACKET_WORD_SIZE_BYTES, remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES, 1);
        }

        return PACKET_HEADER_SIZE_WORDS;
    }

    inline void fvcc_handler() {
        forward_data_from_fvcc_buffer();
        advance_fvcc_rdptr();
    }
};

static_assert(sizeof(fvcc_outbound_state_t) % 4 == 0);

// Fabric Virtual Control Channel (FVCC) Producer receives control/sync packets over ethernet from neighboring chip.
// Data in the producer is either destined for local chip, or has to make a noc hop
// to next outgoing ethernet port enroute to final destination.
// Control packets are forwarded to next fvcc consumer buffer in the route
// direction, if not meant for local device.
// If control packet is addressed to local device, FVCC producer can process the packet locally if
// it is a read/write ack, or forward the packet to Gatekeeper for further local processing.
struct fvcc_inbound_state_t {
    volatile chan_payload_ptr inbound_wrptr;
    volatile chan_payload_ptr inbound_rdptr;
    uint32_t remote_ptr_update_addr;
    volatile ctrl_chan_msg_buf* fvcc_buf;  // fvcc buffer that receives incoming control messages over ethernet.
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t pad1;
    uint8_t pad2;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    bool curr_packet_valid;
    bool packet_corrupted;
    uint64_t packet_timestamp;
    uint64_t gk_fvcc_buf_addr;  // fvcc buffer in gatekeeper.
    volatile packet_header_t* current_packet_header;

    inline void init(uint32_t buf_start, uint32_t ptr_update_addr, uint64_t gk_fvcc_buf_start) {
        uint32_t words = sizeof(fvcc_inbound_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        fvcc_buf = (volatile ctrl_chan_msg_buf*)buf_start;
        gk_fvcc_buf_addr = gk_fvcc_buf_start;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr, uint32_t inc) { return (ptr + inc) & FVCC_PTR_MASK; }

    inline void advance_local_wrptr(uint32_t inc) { inbound_wrptr.ptr = inc_ptr_with_wrap(inbound_wrptr.ptr, inc); }

    inline void advance_out_wrptr(uint32_t inc) { fvc_out_wrptr = inc_ptr_with_wrap(fvc_out_wrptr, inc); }

    inline void advance_out_rdptr(uint32_t inc) { fvc_out_rdptr = inc_ptr_with_wrap(fvc_out_rdptr, inc); }

    inline uint32_t get_num_msgs_available() const {
        uint32_t wrptr = inbound_wrptr.ptr;
        uint32_t msgs = 0;
        if (fvc_out_rdptr != wrptr) {
            msgs = wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : wrptr + 2 * FVCC_BUF_SIZE - fvc_out_rdptr;
        }
        return msgs;
    }

    inline uint32_t get_num_msgs_free() { return FVCC_BUF_SIZE - get_num_msgs_available(); }

    inline uint32_t words_before_local_buffer_wrap() {
        if (inbound_wrptr.ptr >= FVCC_BUF_SIZE) {
            return (FVCC_BUF_SIZE * 2 - inbound_wrptr.ptr) * PACKET_HEADER_SIZE_WORDS;
        } else {
            return (FVCC_BUF_SIZE - inbound_wrptr.ptr) * PACKET_HEADER_SIZE_WORDS;
        }
    }

    inline bool get_curr_packet_valid() {
        if (!curr_packet_valid && (get_num_msgs_available() >= 1)) {
            uint32_t msg_index = fvc_out_rdptr & FVCC_SIZE_MASK;
            bool msg_valid = fvcc_buf->msg_buf[msg_index].packet_header.routing.flags != 0;
            if (msg_valid) {
                current_packet_header = (volatile packet_header_t*)&fvcc_buf->msg_buf[msg_index];
                if (tt_fabric_is_header_valid((packet_header_t*)current_packet_header)) {
                    this->curr_packet_valid = true;
                } else {
                    this->packet_corrupted = true;
                }
            }
        }
        return this->curr_packet_valid;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = (uint32_t)&fvcc_buf->msg_buf[fvc_out_rdptr & FVCC_SIZE_MASK];
        return addr;
    }

    inline uint32_t get_local_buffer_write_addr() {
        uint32_t addr = (uint32_t)&fvcc_buf->msg_buf[inbound_wrptr.ptr & FVCC_SIZE_MASK];
        ;
        return addr;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void update_remote_rdptr_sent() {
        if (inbound_wrptr.ptr_cleared != inbound_rdptr.ptr) {
            inbound_rdptr.ptr = inbound_wrptr.ptr_cleared;
            if constexpr (fvc_mode == FVC_MODE_ROUTER) {
                internal_::eth_send_packet(
                    0,
                    ((uint32_t)&inbound_rdptr) / PACKET_WORD_SIZE_BYTES,
                    remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES,
                    1);
            }
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void update_remote_rdptr_cleared() {
        if (fvc_out_rdptr != inbound_rdptr.ptr_cleared) {
            inbound_rdptr.ptr_cleared = fvc_out_rdptr;
            if constexpr (fvc_mode == FVC_MODE_ROUTER) {
                internal_::eth_send_packet(
                    0,
                    ((uint32_t)&inbound_rdptr) / PACKET_WORD_SIZE_BYTES,
                    remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES,
                    1);
            }
        }
    }

    uint32_t get_next_hop_router_noc_xy() {
        uint32_t dst_mesh_id = current_packet_header->routing.dst_mesh_id;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            uint32_t next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        } else {
            uint32_t dst_device_id = current_packet_header->routing.dst_dev_id;
            uint32_t next_port = routing_table->intra_mesh_table.dest_entry[dst_device_id];
            ASSERT(next_port != INVALID_DIRECTION);
            return eth_chan_to_noc_xy[noc_index][next_port];
        }
    }

    inline bool packet_is_for_local_chip() {
        return (current_packet_header->routing.dst_mesh_id == routing_table->my_mesh_id) &&
               (current_packet_header->routing.dst_dev_id == routing_table->my_device_id);
    }

    // issue a pull request.
    // currently blocks till the request queue has space.
    // This needs to be non blocking, so that if one fvc pull request queue is full,
    // we can process other fvcs and come back to check status of this pull request later.
    inline void forward_message(uint64_t dest_addr) {
        uint64_t noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, wrptr);
        noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
            noc_index,
            NCRISC_AT_CMD_BUF,
            noc_addr,
            NOC_UNICAST_WRITE_VC,
            1,
            FVCC_BUF_LOG_SIZE,
            false,
            false,
            (uint32_t)&fvcc_buf->wrptr.ptr);
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
        uint32_t wrptr = fvcc_buf->wrptr.ptr;
        noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, rdptr);
        while (1) {
            invalidate_l1_cache();
            noc_async_read_one_packet(noc_addr, (uint32_t)(&fvcc_buf->rdptr.ptr), 4);
            noc_async_read_barrier();
            if (!fvcc_buf_ptrs_full(wrptr, fvcc_buf->rdptr.ptr)) {
                break;
            }
#if defined(COMPILE_FOR_ERISC)
            else {
                // Consumer pull request buffer is full
                // Context switch to enable base firmware routing
                // as it might be handling slow dispatch traffic
                internal_::risc_context_switch();
            }
#endif
        }
        uint32_t dest_wr_index = wrptr & FVCC_SIZE_MASK;
        noc_addr = dest_addr + offsetof(ctrl_chan_msg_buf, msg_buf) + dest_wr_index * sizeof(packet_header_t);
        noc_async_write_one_packet((uint32_t)(current_packet_header), noc_addr, sizeof(packet_header_t), noc_index);
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void process_inbound_packet() {
        if (packet_is_for_local_chip()) {
            if (current_packet_header->routing.flags == SYNC) {
                if (current_packet_header->session.command == SOCKET_OPEN or
                    current_packet_header->session.command == SOCKET_CONNECT) {
                    // forward socket related messages to gatekeeper.
                    forward_message(gk_fvcc_buf_addr);
                } else if (current_packet_header->session.command == ASYNC_WR_RESP) {
                    // Write response. Decrement transaction count for respective transaction id.
                    uint64_t noc_addr = get_noc_addr_helper(
                        current_packet_header->session.target_offset_h, current_packet_header->session.target_offset_l);
                    noc_fast_atomic_increment(
                        noc_index, NCRISC_AT_CMD_BUF, noc_addr, NOC_UNICAST_WRITE_VC, -1, 31, false);
                }
            }
        } else {
            // Control message is not meant for local chip. Forward to next router enroute to destination.
            uint64_t dest_addr = get_noc_addr_helper(get_next_hop_router_noc_xy(), FVCC_OUT_BUF_START);
            forward_message(dest_addr);
        }
        curr_packet_valid = false;
        advance_out_wrptr(1);
        advance_out_rdptr(1);
        noc_async_write_barrier();
        update_remote_rdptr_cleared<fvc_mode>();
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void fvcc_handler() {
        if (get_curr_packet_valid()) {
            process_inbound_packet<fvc_mode>();
        }
        update_remote_rdptr_sent<fvc_mode>();
    }
};

struct socket_reader_state_t {
    volatile chan_payload_ptr remote_rdptr;
    uint8_t packet_in_progress;

    uint32_t packet_words_remaining;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    uint32_t fvc_pull_wrptr;
    uint32_t buffer_size;
    uint32_t buffer_start;
    uint32_t remote_buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_since_last_sync;

    uint32_t get_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_pull_wrptr != rd_ptr) {
            words_occupied =
                fvc_pull_wrptr > rd_ptr ? fvc_pull_wrptr - rd_ptr : buffer_size * 2 + fvc_pull_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    uint32_t get_remote_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr_cleared;
        uint32_t words_occupied = 0;
        if (fvc_out_wrptr != rd_ptr) {
            words_occupied = fvc_out_wrptr > rd_ptr ? fvc_out_wrptr - rd_ptr : buffer_size * 2 + fvc_out_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words) {
        uint32_t words = sizeof(socket_reader_state_t) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        remote_buffer_start = data_buf_start + buffer_size * PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t words_before_local_buffer_wrap() {
        if (fvc_pull_wrptr >= buffer_size) {
            return buffer_size * 2 - fvc_pull_wrptr;
        } else {
            return buffer_size - fvc_pull_wrptr;
        }
    }

    inline uint32_t get_local_buffer_pull_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_pull_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_out_rdptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_remote_buffer_write_addr() {
        uint32_t addr = remote_buffer_start;
        uint32_t offset = fvc_out_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline void advance_pull_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_pull_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_pull_wrptr = temp;
    }

    inline void advance_out_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_out_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_out_wrptr = temp;
    }

    inline void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_out_rdptr = temp;
    }

    inline void register_pull_data(uint32_t num_words_to_pull) {
        pull_words_in_flight += num_words_to_pull;
        advance_pull_wrptr(num_words_to_pull);
        words_since_last_sync += num_words_to_pull;
        packet_words_remaining -= num_words_to_pull;
    }

    inline uint32_t get_num_words_to_pull(volatile pull_request_t* pull_request) {
        uint32_t num_words_to_pull = num_words_available_to_pull(pull_request);
        uint32_t num_words_before_wrap = words_before_pull_buffer_wrap(pull_request->buffer_size, pull_request->rd_ptr);

        num_words_to_pull = std::min(num_words_to_pull, num_words_before_wrap);
        uint32_t socket_buffer_space = get_num_words_free();
        num_words_to_pull = std::min(num_words_to_pull, socket_buffer_space);

        if (num_words_to_pull == 0) {
            return 0;
        }

        uint32_t space_before_wptr_wrap = words_before_local_buffer_wrap();
        num_words_to_pull = std::min(num_words_to_pull, space_before_wptr_wrap);

        return num_words_to_pull;
    }

    inline uint32_t pull_socket_data(volatile pull_request_t* pull_request) {
        volatile uint32_t* temp = (volatile uint32_t*)0xffb2010c;
        if (packet_in_progress == 0) {
            uint32_t size = pull_request->size;
            packet_words_remaining = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            packet_in_progress = 1;
        }

        uint32_t num_words_to_pull = get_num_words_to_pull(pull_request);
        if (num_words_to_pull == 0) {
            temp[0] = 0xdead1111;
            return 0;
        }

        uint32_t rd_offset = get_rd_ptr_offset_words((pull_request_t*)pull_request);
        uint64_t src_addr = pull_request->buffer_start + (rd_offset * PACKET_WORD_SIZE_BYTES);
        uint32_t local_addr = get_local_buffer_pull_addr();

        // pull_data_from_remote();
        noc_async_read(src_addr, local_addr, num_words_to_pull * PACKET_WORD_SIZE_BYTES);
        register_pull_data(num_words_to_pull);
        pull_request->rd_ptr = advance_ptr(pull_request->buffer_size, pull_request->rd_ptr, num_words_to_pull);

        return num_words_to_pull;
    }

    template <bool live = true>
    inline uint32_t push_socket_data() {
        uint32_t total_words_to_forward = 0;
        uint32_t wrptr = fvc_pull_wrptr;

        total_words_to_forward =
            wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : buffer_size * 2 + wrptr - fvc_out_rdptr;

        uint32_t remote_fvc_buffer_space = get_remote_num_words_free();
        total_words_to_forward = std::min(total_words_to_forward, remote_fvc_buffer_space);
        if (total_words_to_forward == 0) {
            return 0;
        }

        if (packet_words_remaining and (words_since_last_sync < FVC_SYNC_THRESHOLD)) {
            // not enough data to forward.
            // wait for more data.
            return 0;
        }

        if constexpr (live == true) {
            uint32_t src_addr = 0;
            uint32_t dest_addr = 0;  // should be second half of fvc buffer.
            uint32_t words_remaining = total_words_to_forward;
            while (words_remaining) {
                invalidate_l1_cache();
                uint32_t num_words_before_local_wrap = words_before_pull_buffer_wrap(buffer_size, fvc_out_rdptr);
                uint32_t num_words_before_remote_wrap = words_before_pull_buffer_wrap(buffer_size, fvc_out_wrptr);
                uint32_t words_to_forward = std::min(num_words_before_local_wrap, num_words_before_remote_wrap);
                words_to_forward = std::min(words_to_forward, words_remaining);
                // max 8K bytes
                words_to_forward = std::min(words_to_forward, DEFAULT_MAX_NOC_SEND_WORDS);
                src_addr = get_local_buffer_read_addr();
                dest_addr = get_remote_buffer_write_addr();

                // TODO: Issue a noc write here to forward data.

                advance_out_rdptr(words_to_forward);
                advance_out_wrptr(words_to_forward);
                words_remaining -= words_to_forward;
            }
        } else {
            advance_out_rdptr(total_words_to_forward);
            advance_out_wrptr(total_words_to_forward);
            remote_rdptr.ptr = fvc_out_rdptr;
            remote_rdptr.ptr_cleared = fvc_out_wrptr;
        }
        words_since_last_sync -= total_words_to_forward;
        return total_words_to_forward;
    }
};

static_assert(sizeof(socket_reader_state_t) % 4 == 0);

struct router_state_t {
    uint32_t sync_in;
    uint32_t padding_in[3];
    uint32_t sync_out;
    uint32_t padding_out[3];
    uint32_t scratch[4];
};

inline uint64_t get_timestamp_32b() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes / 4; i++) {
        buf[i] = 0;
    }
}

static FORCE_INLINE void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE void write_kernel_status(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index + 1] = val & 0xFFFFFFFF;
    }
}

inline void req_buf_ptr_advance(chan_ptr* ptr) { ptr->ptr = (ptr->ptr + 1) & CHAN_REQ_BUF_PTR_MASK; }

inline void req_buf_advance_wrptr(chan_req_buf* req_buf) { req_buf_ptr_advance(&(req_buf->wrptr)); }

inline void req_buf_advance_rdptr(chan_req_buf* req_buf) {
    // clear valid before incrementing read pointer.
    // PACK and FORWARD requests take 2 entries in request buffer.
    // First entry is pull reqeust itself, second entry is packet header.
    uint32_t rd_index = req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
    if (req_buf->chan_req[rd_index].pull_request.flags == PACK_N_FORWARD) {
        req_buf->chan_req[rd_index].pull_request.flags = 0;
        req_buf_ptr_advance(&(req_buf->rdptr));
        rd_index = (rd_index + 1) & CHAN_REQ_BUF_SIZE_MASK;
    }
    req_buf->chan_req[rd_index].pull_request.flags = 0;
    req_buf_ptr_advance(&(req_buf->rdptr));
}

inline bool req_buf_ptrs_empty(uint32_t wrptr, uint32_t rdptr) { return (wrptr == rdptr); }

inline bool req_buf_ptrs_full(uint32_t wrptr, uint32_t rdptr) {
    uint32_t distance = wrptr >= rdptr ? wrptr - rdptr : wrptr + 2 * CHAN_REQ_BUF_SIZE - rdptr;
    return !req_buf_ptrs_empty(wrptr, rdptr) && (distance >= CHAN_REQ_BUF_SIZE);
}

inline bool fvc_req_buf_is_empty(const volatile chan_req_buf* req_buf) {
    return req_buf_ptrs_empty(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_buf_is_full(const volatile chan_req_buf* req_buf) {
    return req_buf_ptrs_full(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_valid(const volatile chan_req_buf* req_buf) {
    uint32_t rd_index = req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
    return req_buf->chan_req[rd_index].pull_request.flags != 0;
}

inline uint32_t num_words_available_to_pull(volatile pull_request_t* pull_request) {
    return pull_request->words_written - pull_request->words_read;
}

inline uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words) {
    uint32_t temp = ptr + inc_words;
    if (temp >= buffer_size) {
        temp -= buffer_size;
    }
    return temp;
}

inline uint32_t words_before_pull_buffer_wrap(uint32_t buffer_size, uint32_t rd_ptr) { return buffer_size - rd_ptr; }

inline uint32_t get_rd_ptr_offset_words(pull_request_t* pull_request) { return pull_request->rd_ptr; }

inline void update_pull_request_words_cleared(pull_request_t* pull_request) {
    noc_inline_dw_write(pull_request->ack_addr, pull_request->words_read);
}

/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(volatile router_state_t* router_state, uint32_t timeout_cycles = 0) {
    bool src_ready = false;
    bool dest_ready = false;

    uint32_t iters = 0;

    uint32_t start_timestamp = get_timestamp_32b();
    uint32_t sync_in_addr = ((uint32_t)&router_state->sync_in) / PACKET_WORD_SIZE_BYTES;
    uint32_t sync_out_addr = ((uint32_t)&router_state->sync_out) / PACKET_WORD_SIZE_BYTES;

    uint32_t scratch_addr = ((uint32_t)&router_state->scratch) / PACKET_WORD_SIZE_BYTES;
    router_state->scratch[0] = 0xAA;

    while (!src_ready or !dest_ready) {
        invalidate_l1_cache();
        if (router_state->sync_out != 0xAA) {
            internal_::eth_send_packet(0, scratch_addr, sync_in_addr, 1);
        } else {
            dest_ready = true;
        }

        if (!src_ready && router_state->sync_in == 0xAA) {
            internal_::eth_send_packet(0, sync_in_addr, sync_out_addr, 1);
            src_ready = true;
        }

        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }

#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            // if timeout is disabled, context switch every 4096 iterations.
            // this is necessary to allow ethernet routing layer to operate.
            // this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}

// reserve a slot in the req queue for sending the pull request
inline void tt_fabric_reserve_pull_request_slot(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request, uint32_t num_slots) {
    uint64_t noc_addr = dest_addr + offsetof(chan_req_buf, wrptr);
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        noc_addr,
        NOC_UNICAST_WRITE_VC,
        num_slots,
        CHAN_REQ_BUF_LOG_SIZE,
        false,
        false,
        (uint32_t)&local_pull_request->wrptr.ptr);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
}

// check if the pull request can be sent
// issuing this in a blocking mode on routers can result in deadlocks, use carefully
template <bool blocking_mode = false>
inline bool tt_fabric_check_pull_request_slot(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request, uint32_t wrptr) {
    uint64_t noc_addr = dest_addr + offsetof(chan_req_buf, rdptr);
    do {
        invalidate_l1_cache();
        noc_async_read_one_packet(noc_addr, (uint32_t)(&local_pull_request->rdptr.ptr), 4);
        noc_async_read_barrier();
        if (!req_buf_ptrs_full(wrptr, local_pull_request->rdptr.ptr)) {
            return true;
        }
    } while (blocking_mode);

    return false;
}

// issue a pull request.
inline uint64_t tt_fabric_send_pull_request(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request) {
    uint32_t dest_wr_index = (local_pull_request->wrptr.ptr) & CHAN_REQ_BUF_SIZE_MASK;
    uint64_t noc_addr = dest_addr + offsetof(chan_req_buf, chan_req) + dest_wr_index * sizeof(pull_request_t);
    noc_async_write_one_packet(
        (uint32_t)(&local_pull_request->pull_request), noc_addr, sizeof(pull_request_t), noc_index);

    // compute the address to send write pointer updates to consumer buffer.
    // This will happen, if the producer did not have all the availale data in its buffer when
    // the pull request was first issued. In this case, as the producer gets more data in its buffer,
    // it updates write pointer in the consumer request buffer pull request entry.
    uint64_t words_written_addr = noc_addr + offsetof(pull_request_t, words_written);
    return words_written_addr;
}

inline void tt_fabric_init() { xy_local_addr = get_noc_addr(0); }
