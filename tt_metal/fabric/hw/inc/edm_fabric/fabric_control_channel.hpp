// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "fabric_edm_types.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "edm_fabric_flow_control_helpers.hpp"

// TODO: get these from compile args
#define HOST_BUFFER_DEPTH 2
#define ETH_BUFFER_DEPTH 4
#define NUM_ROUTERS 16

namespace tt::tt_fabric {

// maybe have a Nestedbuffer type for the local group
// i.e. array of buffers, each with its own rd/wr ptrs and depth
// for V1, we can start with a single nested buffer. Later we can extend to multiple buffers for QoS
// one buffer for local producers and one for remote producers (forwarded packets from host and eth groups)

template <typename T, uint8_t NUM_BUFFERS>
class FabricControlChannelQueue {
public:
    explicit FabricControlChannelQueue() = default;

    FORCE_INLINE void init() {
        rd_ptr = BufferIndex(0);
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            slots[i] = T{};
        }
    }

    void push(const T& data, const BufferIndex& buffer_index) { slots[buffer_index] = data; }

    T pop() {
        T data = slots[rd_ptr];
        rd_ptr = BufferIndex{wrap_increment<NUM_BUFFERS>(rd_ptr.get())};
        return data;
    }

private:
    std::array<T, NUM_BUFFERS> slots;
    BufferIndex rd_ptr = BufferIndex(0);
};

template <uint8_t NUM_BUFFERS>
class FabricControlChannelBuffer {
public:
    explicit FabricControlChannelBuffer() = default;

    FORCE_INLINE void init(size_t channel_base_address) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] = channel_base_address + i * sizeof(ControlPacketHeader);
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
            for (size_t j = 0; j < sizeof(ControlPacketHeader) / sizeof(uint32_t); j++) {
                reinterpret_cast<volatile uint32_t*>(this->buffer_addresses[i])[j] = 0;
            }
        }
    }

    FabricControlChannelBuffer(size_t channel_base_address) { init(channel_base_address); }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const BufferIndex& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;
};

struct FSMContext {
    bool active = false;
    uint32_t state = 0;
};

template <uint8_t NUM_BUFFERS>
struct FabricControlChannelEthBufferPointers {
    uint32_t num_free_slots;
    BufferIndex remote_receiver_buffer_index;

    FORCE_INLINE void init() {
        this->num_free_slots = NUM_BUFFERS;
        this->remote_receiver_buffer_index = BufferIndex(0);
    }

    FORCE_INLINE bool has_space_for_packet() const { return this->num_free_slots > 0; }
};

template <uint8_t NUM_ETH_CHANNELS>
struct FabricControlChannelLocalBufferPointers {
    std::array<uint8_t, NUM_ETH_CHANNELS> num_free_slots;

    // no need to track buffer indices since they are fixed

    FORCE_INLINE void init() { num_free_slots.fill(0); }

    FORCE_INLINE bool router_has_space_for_packet(uint8_t channel_id) const { return num_free_slots[channel_id] > 0; }

    FORCE_INLINE bool routers_have_space_for_packet(uint32_t router_eth_chans_mask) const {
        uint32_t remaining_channels = router_eth_chans_mask;
        while (remaining_channels) {
            uint32_t chan = __builtin_ctz(remaining_channels);
            if (num_free_slots[chan] == 0) {
                return false;
            }
            remaining_channels &= ~(0x1 << chan);
        }

        return true;
    }
};

struct FabricControlChannel {
    FSMContext fsm_context;

    FabricControlChannelBuffer<HOST_BUFFER_DEPTH> host_buffer;
    FabricControlChannelBuffer<ETH_BUFFER_DEPTH> eth_buffer;
    FabricControlChannelBuffer<NUM_ROUTERS> local_buffer;
    FabricControlChannelQueue<uint8_t, NUM_ROUTERS> local_arrival_queue;  // arrival queue for ordering and QoS

    // dont need remote host buffer credit tracking since only host will be pushing to the host buffer

    FabricControlChannelEthBufferPointers<ETH_BUFFER_DEPTH> remote_eth_buffer_ptrs;
    FabricControlChannelLocalBufferPointers<NUM_ROUTERS> remote_local_buffer_ptrs;

    // TODO:
    // 1. need queue ptrs for remote increment via noc_atomic_inc - should this be a connection object?

    void init() {
        fsm_context.active = true;
        host_buffer.init(0);
        eth_buffer.init(0);
        local_buffer.init(0);
        local_arrival_queue.init();

        remote_eth_buffer_ptrs.init();
        remote_local_buffer_ptrs.init();
    }

    void forward_control_packet() {
        // forward to eth or local based on destination
        // update relevant buffer ptrs and counters
        // if local, push to arrival queue
    }

    void process_control_packet(tt_l1_ptr ControlPacketHeader* packet_start) {
        // process the control packet based on its type and source
        // update the relevant state machines and buffer pointers

        // we may need to forward packets to other routers -> either over eth or in the local queue
        // we will need to check if there is space in the relevant buffers before forwarding
        // for local routers, we should be able to check space against a mask
    }

    void process() {
        if (fsm_context.active) {
            // Process the control channel
            fsm_context.state++;
        }

        // process host buffer
        // process eth buffer
        // process local buffer

        // each of the above should call process_control_packet()

        // do not make any state transitions until we are able to forward packets to all the
        // routers. we do not want to have an outbound queue so we would rather leave the
        // packets marked unprocessed in the inbound buffers and process them again the next time
    }

    void teardown(uint32_t dummy_addr) {
        fsm_context.active = false;
        host_buffer.init(0);
        eth_buffer.init(0);
        local_buffer.init(0);

        auto status_ptr = reinterpret_cast<volatile uint32_t*>(dummy_addr);
        *status_ptr = fsm_context.state;
    }
};

}  // namespace tt::tt_fabric
