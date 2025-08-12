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
#include "hostdevcommon/fabric_common.h"
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
extern volatile tt_l1_ptr fabric_router_l1_config_t* routing_table;
extern chan_payload_ptr inbound_rdptr_ack;
extern volatile chan_payload_ptr remote_rdptr;

uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words);

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
            noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);

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
        noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
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
        noc_inline_dw_write<InlineWriteDst::DEFAULT>(push_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);

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

enum ProcessingFlags : uint8_t {
    UCAST_DEST = 1,
    MCAST_DEST = 2,
    NOT_DEST = 3,
};

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

inline bool req_buf_ptrs_empty(uint32_t wrptr, uint32_t rdptr) { return (wrptr == rdptr); }

inline bool req_buf_ptrs_full(uint32_t wrptr, uint32_t rdptr) {
    uint32_t distance = wrptr >= rdptr ? wrptr - rdptr : wrptr + 2 * CHAN_REQ_BUF_SIZE - rdptr;
    return !req_buf_ptrs_empty(wrptr, rdptr) && (distance >= CHAN_REQ_BUF_SIZE);
}

inline uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words) {
    uint32_t temp = ptr + inc_words;
    if (temp >= buffer_size) {
        temp -= buffer_size;
    }
    return temp;
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

inline void tt_fabric_init() { xy_local_addr = get_noc_addr(0); }
