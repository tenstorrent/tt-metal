// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "fabric_edm_types.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "edm_fabric_flow_control_helpers.hpp"
#include "fabric_erisc_router_ct_args.hpp"
#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"
#include "eth_chan_noc_mapping.h"

// TODO: get these from compile args
#define HOST_BUFFER_SLOTS 2
#define ETH_BUFFER_SLOTS 4
#define LOCAL_ROUTER_SLOTS 4
#define NUM_ROUTERS 16

namespace tt::tt_fabric {

namespace control_channel {

// maybe have a Nestedbuffer type for the local group
// i.e. array of buffers, each with its own rd/wr ptrs and depth
// for V1, we can start with a single nested buffer. Later we can extend to multiple buffers for QoS
// one buffer for local producers and one for remote producers (forwarded packets from host and eth groups)

template <uint8_t NUM_SLOTS>
class FabricControlChannelBuffer {
public:
    explicit FabricControlChannelBuffer() = default;

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const uint8_t slot_index) const {
        return base_address_ + slot_index * slot_size_;
    }

    FORCE_INLINE void init() {
        for (uint8_t i = 0; i < NUM_SLOTS; i++) {
            size_t slot_address = get_buffer_address(i);
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
            for (size_t j = 0; j < slot_size_ / sizeof(uint32_t); j++) {
                reinterpret_cast<volatile uint32_t*>(slot_address)[j] = 0;
            }
        }
    }

    FabricControlChannelBuffer(uint32_t channel_base_address) :
        base_address_(channel_base_address), slot_size_(sizeof(ControlPacketHeader)) {
        init();
    }

    // for optimization, we can store buffer addresses instead of computing them each time
private:
    size_t base_address_;
    size_t slot_size_;
};

template <uint8_t NUM_GROUPS, uint8_t NUM_SLOTS_PER_GROUP>
class FabricControlChannelNestedBuffer {
public:
    explicit FabricControlChannelNestedBuffer() = default;

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const uint8_t group_index, const uint8_t slot_index) const {
        return base_address_ + (group_index * NUM_SLOTS_PER_GROUP + slot_index) * slot_size_;
    }

    FORCE_INLINE void init() {
        for (uint8_t g = 0; g < NUM_GROUPS; g++) {
            for (uint8_t s = 0; s < NUM_SLOTS_PER_GROUP; s++) {
                size_t slot_address = get_buffer_address(g, s);
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
                for (size_t j = 0; j < slot_size_ / sizeof(uint32_t); j++) {
                    reinterpret_cast<volatile uint32_t*>(slot_address)[j] = 0;
                }
            }
        }
    }

    FabricControlChannelNestedBuffer(uint32_t channel_base_address) :
        base_address_(channel_base_address), slot_size_(sizeof(ControlPacketHeader)) {
        init();
    }

    // for optimization, we can store buffer addresses instead of computing them each time
private:
    size_t base_address_;
    size_t slot_size_;
};

template <uint8_t NUM_BUFFER_SLOTS>
class HostBufferConsumerInterface {
public:
    explicit HostBufferConsumerInterface() = default;

    FORCE_INLINE void init(
        uint32_t channel_base_address, uint32_t from_remote_write_counter_addr, uint32_t to_remote_read_counter_addr) {
        host_buffer_ = FabricControlChannelBuffer<NUM_BUFFER_SLOTS>(channel_base_address);
        from_remote_write_counter_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_write_counter_addr);
        to_remote_read_counter_ = reinterpret_cast<tt_l1_ptr uint32_t*>(to_remote_read_counter_addr);
        local_read_counter_.reset();
    }

    FORCE_INLINE bool has_new_packet() const {
        invalidate_l1_cache();
        return *from_remote_write_counter_ - local_read_counter_.counter < NUM_BUFFER_SLOTS;
    }

    FORCE_INLINE size_t get_next_packet() const {
        return host_buffer_.get_buffer_address(local_read_counter_.get_buffer_index().get());
    }

    FORCE_INLINE void advance() {
        update_remote_read_counter();
        update_local_read_counter();
    }

    HostBufferConsumerInterface(
        uint32_t channel_base_address, uint32_t from_remote_write_counter_addr, uint32_t to_remote_read_counter_addr) {
        this->init(channel_base_address, from_remote_write_counter_addr, to_remote_read_counter_addr);
    };

private:
    FORCE_INLINE void update_local_read_counter() { local_read_counter_.increment(); }

    FORCE_INLINE void update_remote_read_counter() const {
        // Notify host of read pointer update
        // No noc operations needed, since we are updating an address on our own L1 that host reads back
        *to_remote_read_counter_ = local_read_counter_.counter;
    }

    FabricControlChannelBuffer<NUM_BUFFER_SLOTS> host_buffer_;
    volatile tt_l1_ptr uint32_t* from_remote_write_counter_;
    tt_l1_ptr uint32_t* to_remote_read_counter_;
    ChannelCounter<NUM_BUFFER_SLOTS> local_read_counter_;
};

template <uint8_t NUM_BUFFER_SLOTS>
class EthBufferConsumerInterface {
public:
    explicit EthBufferConsumerInterface() = default;

    FORCE_INLINE void init(
        uint32_t channel_base_address,
        uint32_t from_remote_write_counter_addr,
        uint32_t to_remote_read_counter_addr,
        uint32_t local_read_counter_addr) {
        eth_buffer_ = FabricControlChannelBuffer<NUM_BUFFER_SLOTS>(channel_base_address);
        from_remote_write_counter_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_write_counter_addr);
        to_remote_read_counter_addr_ = to_remote_read_counter_addr;
        local_read_counter_ptr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_read_counter_addr);
        local_read_counter_.reset();
    }

    FORCE_INLINE bool has_new_packet() const {
        invalidate_l1_cache();
        return *from_remote_write_counter_ - local_read_counter_.counter < NUM_BUFFER_SLOTS;
    }

    FORCE_INLINE size_t get_next_packet() const {
        return eth_buffer_.get_buffer_address(local_read_counter_.get_buffer_index().get());
    }

    FORCE_INLINE void advance() {
        update_remote_read_counter();
        update_local_read_counter();
    }

    EthBufferConsumerInterface(
        uint32_t channel_base_address,
        uint32_t from_remote_write_counter_addr,
        uint32_t to_remote_read_counter_addr,
        uint32_t local_read_counter_addr) {
        this->init(
            channel_base_address, from_remote_write_counter_addr, to_remote_read_counter_addr, local_read_counter_addr);
    };

private:
    FORCE_INLINE void update_local_read_counter() { local_read_counter_.increment(); }

    FORCE_INLINE void update_remote_read_counter() const {
        // Push pointer updates over eth
        *local_read_counter_ptr_ = local_read_counter_.counter;
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            reinterpret_cast<uint32_t>(local_read_counter_ptr_),
            to_remote_read_counter_addr_,
            ETH_WORD_SIZE_BYTES);
    }

    FabricControlChannelBuffer<NUM_BUFFER_SLOTS> eth_buffer_;
    volatile tt_l1_ptr uint32_t* from_remote_write_counter_;
    uint32_t to_remote_read_counter_addr_;
    volatile tt_l1_ptr uint32_t* local_read_counter_ptr_;
    ChannelCounter<NUM_BUFFER_SLOTS> local_read_counter_;
};

template <uint8_t NUM_GROUPS, uint8_t NUM_SLOTS_PER_GROUP>
class LocalBufferConsumerInterface {
public:
    LocalBufferConsumerInterface() = default;

    FORCE_INLINE void init(
        uint32_t channel_base_address,
        uint32_t from_remote_write_counters_base_addr,
        uint32_t to_remote_read_counters_base_addr) :
        from_remote_write_counters_base_addr_(from_remote_write_counters_base_addr),
        to_remote_read_counters_base_addr_(to_remote_read_counters_base_addr) {
        local_buffer_ = FabricControlChannelNestedBuffer<NUM_GROUPS, NUM_SLOTS_PER_GROUP>(channel_base_address);

// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
        for (uint8_t g = 0; g < NUM_GROUPS; g++) {
            local_read_counters_[g].reset();
        }
    }

    FORCE_INLINE bool has_new_packet() const {
        // before checking L1, see if we already have unprocessed packets
        if (unprocessed_packet_mask_ != 0) {
            return true;
        }

// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
        for (uint8_t g = 0; g < NUM_GROUPS; g++) {
            if (group_has_new_packet(g)) {
                unprocessed_packet_mask_ |= (0x1 << g);
            }
        }

        // set current_group_index_ to the first group with new packets
        current_group_index_ = __builtin_ctz(unprocessed_packet_mask_);

        return unprocessed_packet_mask_ != 0;
    }

    FORCE_INLINE size_t get_next_packet() const {
        return local_buffer_.get_buffer_address(
            current_group_index_, local_read_counters_[current_group_index_].get_buffer_index().get());
    }

    FORCE_INLINE void advance() {
        update_remote_read_counter();
        update_local_read_counter();
    }

    LocalBufferConsumerInterface(
        uint32_t channel_base_address,
        uint32_t from_remote_write_counters_base_addr,
        uint32_t to_remote_read_counters_base_addr) {
        this->init(channel_base_address, from_remote_write_counters_base_addr, to_remote_read_counters_base_addr);
    }

private:
    FORCE_INLINE uint32_t get_group_addr(const uint8_t group_index, const uint32_t base_address) const {
        // we are not caching any addresses to save stack space, so we need to compute them on the fly
        // as future optimization, we can cache these addresses if needed
        return base_address + group_index * sizeof(uint32_t);
    }

    FORCE_INLINE bool group_has_new_packet(const uint8_t group_index) const {
        invalidate_l1_cache();
        uint32_t remote_write_counter_addr = get_group_addr(group_index, from_remote_write_counters_base_addr_);
        volatile tt_l1_ptr uint32_t* remote_write_counter =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_write_counter_addr);
        return *remote_write_counter - local_read_counters_[group_index].counter < NUM_SLOTS_PER_GROUP;
    }

    FORCE_INLINE void advance_group_index() {
        // advance current_group_index_ to the next group with new packets
        unprocessed_packet_mask_ &= ~(0x1 << current_group_index_);
        if (unprocessed_packet_mask_ != 0) {
            current_group_index_ = __builtin_ctz(unprocessed_packet_mask_);
        }
    }

    FORCE_INLINE void update_local_read_counter() {
        // MAKE SURE THAT THIS IS CALLED AFTER UPDATING REMOTE READ COUNTER
        local_read_counters_[current_group_index_].increment();
        advance_group_index();
    }

    FORCE_INLINE void update_remote_read_counter() const {
        // need to prepare noc address using the target router's noc coords
        uint64_t noc_addr = get_noc_addr_helper(
            eth_chan_to_noc_xy[noc_index][current_group_index_],
            get_group_addr(current_group_index_, to_remote_read_counters_base_addr_));
        noc_inline_dw_write(noc_addr, local_read_counters_[current_group_index_].counter);
    }

    FabricControlChannelNestedBuffer<NUM_GROUPS, NUM_SLOTS_PER_GROUP> local_buffer_;
    std::array<ChannelCounter<NUM_SLOTS_PER_GROUP>, NUM_GROUPS> local_read_counters_;
    uint32_t from_remote_write_counters_base_addr_;
    uint32_t to_remote_read_counters_base_addr_;
    uint32_t unprocessed_packet_mask_ = 0;
    uint8_t current_group_index_ = 0;
};

template <uint8_t NUM_BUFFER_SLOTS>
class EthProducerInterface {
public:
    explicit EthProducerInterface() = default;

    FORCE_INLINE void init(
        uint32_t remote_buffer_base_addr,
        uint32_t from_remote_read_counter_addr,
        uint32_t to_remote_write_counter_addr,
        uint32_t local_write_counter_addr) {
        remote_buffer_base_addr_ = remote_buffer_base_addr;
        from_remote_read_counter_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_read_counter_addr);
        to_remote_write_counter_addr_ = to_remote_write_counter_addr;
        local_write_counter_ptr_ = reinterpret_cast<tt_l1_ptr uint32_t*>(local_write_counter_addr);
        local_write_counter_.reset();
    }

    FORCE_INLINE bool remote_has_space_for_packet() const {
        return local_write_counter_.counter - *from_remote_read_counter_ < NUM_BUFFER_SLOTS;
    }

    // SHOULD ONLY BE CALLED IF REMOTE HAS SPACE
    FORCE_INLINE void forward_packet(uint32_t* packet_start) {
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            reinterpret_cast<uint32_t>(packet_start),
            get_remote_buffer_slot_address(),
            sizeof(ControlPacketHeader));

        update_local_write_counter();
        update_remote_write_counter();
    }

    EthProducerInterface(
        uint32_t remote_buffer_base_addr,
        uint32_t from_remote_read_counter_addr,
        uint32_t to_remote_write_counter_addr,
        uint32_t local_write_counter_addr) {
        this->init(
            remote_buffer_base_addr,
            from_remote_read_counter_addr,
            to_remote_write_counter_addr,
            local_write_counter_addr);
    }

private:
    FORCE_INLINE uint32_t get_remote_buffer_slot_address() const {
        return remote_buffer_base_addr_ + (local_write_counter_.get_buffer_index().get() * sizeof(ControlPacketHeader));
    }

    FORCE_INLINE void update_local_write_counter() { local_write_counter_.increment(); }

    FORCE_INLINE void update_remote_write_counter() const {
        *local_write_counter_ptr_ = local_write_counter_.counter;
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            reinterpret_cast<uint32_t>(local_write_counter_ptr_),
            to_remote_write_counter_addr_,
            sizeof(uint32_t));
    }

    tt_l1_ptr uint32_t* local_write_counter_ptr_;  // needed to get an address for eth send
    ChannelCounter<NUM_BUFFER_SLOTS> local_write_counter_;
    volatile tt_l1_ptr uint32_t* from_remote_read_counter_;
    uint32_t to_remote_write_counter_addr_;
    uint32_t remote_buffer_base_addr_;  // could be constexpr if needed
};

template <uint8_t NUM_GROUPS, uint8_t NUM_SLOTS_PER_GROUP>
class LocalProducerInterface {
public:
    explicit LocalProducerInterface() = default;

    FORCE_INLINE void init(
        uint32_t remote_buffer_base_addr,
        uint32_t from_remote_read_counters_base_addr,
        uint32_t to_remote_write_counters_base_addr) {
        remote_buffer_base_addr_ = remote_buffer_base_addr;
        from_remote_read_counters_base_addr_ = from_remote_read_counters_base_addr;
        to_remote_write_counters_base_addr_ = to_remote_write_counters_base_addr;

        // Initialize all local write counters
        for (uint8_t g = 0; g < NUM_GROUPS; g++) {
            local_write_counters_[g].reset();
        }
    }

    FORCE_INLINE bool remote_has_space_for_packet(uint32_t router_mask) const {
        uint32_t remaining_routers = router_mask;
        while (remaining_routers) {
            uint32_t channel_id = __builtin_ctz(remaining_routers);
            if (!remote_router_has_space_for_packet(channel_id)) {
                return false;
            }
            remaining_routers &= ~(1 << channel_id);
        }

        return true;
    }

    // SHOULD ONLY BE CALLED IF REMOTE HAS SPACE
    FORCE_INLINE void forward_packet(uint32_t* packet_start, uint32_t router_mask) const {
        uint32_t remaining_routers = router_mask;
        while (remaining_routers) {
            uint32_t channel_id = __builtin_ctz(remaining_routers);
            forward_packet_to_remote_router(packet_start, channel_id);
            remaining_routers &= ~(1 << channel_id);
        }
    }

    LocalProducerInterface(
        uint32_t remote_buffer_base_addr,
        uint32_t from_remote_read_counters_base_addr,
        uint32_t to_remote_write_counters_base_addr) {
        this->init(remote_buffer_base_addr, from_remote_read_counters_base_addr, to_remote_write_counters_base_addr);
    };

private:
    FORCE_INLINE uint32_t get_group_addr(const uint8_t group_index, const uint32_t base_address) const {
        // we are not caching any addresses to save stack space, so we need to compute them on the fly
        // as future optimization, we can cache these addresses if needed
        return base_address + group_index * sizeof(uint32_t);
    }

    FORCE_INLINE uint32_t get_remote_buffer_slot_address(uint8_t channel_id) const {
        uint32_t offset_in_remote_buffer =
            channel_id * NUM_SLOTS_PER_GROUP + local_write_counters_[channel_id].get_buffer_index().get();
        return remote_buffer_base_addr_ + (offset_in_remote_buffer * sizeof(ControlPacketHeader));
    }

    FORCE_INLINE bool remote_router_has_space_for_packet(uint8_t channel_id) const {
        // Check if remote router has space by comparing our write counter with their read counter
        invalidate_l1_cache();
        uint32_t remote_read_counter_addr = get_group_addr(channel_id, from_remote_read_counters_base_addr_);
        volatile tt_l1_ptr uint32_t* remote_read_counter =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_read_counter_addr);
        return local_write_counters_[channel_id].counter - *remote_read_counter < NUM_SLOTS_PER_GROUP;
    }

    FORCE_INLINE void forward_packet_to_remote_router(uint32_t* packet_start, uint8_t channel_id) const {
        // need to prepare noc address using the target router's noc coords
        uint64_t noc_addr =
            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][channel_id], get_remote_buffer_slot_address(channel_id));
        // TODO: fix api usage here
        noc_async_write_one_packet(noc_addr, *packet_start);
        update_local_write_counter(channel_id);
        update_remote_write_counter(channel_id);
    }

    FORCE_INLINE void update_local_write_counter(uint8_t channel_id) { local_write_counters_[channel_id].increment(); }

    FORCE_INLINE void update_remote_write_counter(uint8_t channel_id) const {
        // Update remote router's write counter to notify them of new packet
        uint32_t remote_write_counter_addr = get_group_addr(channel_id, to_remote_write_counters_base_addr_);
        uint64_t noc_addr = get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][channel_id], remote_write_counter_addr);
        noc_inline_dw_write(noc_addr, local_write_counters_[channel_id].counter);
    }

    std::array<ChannelCounter<NUM_SLOTS_PER_GROUP>, NUM_GROUPS> local_write_counters_;
    uint32_t remote_buffer_base_addr_;  // could be constexpr if needed since its same across all routers
    uint32_t from_remote_read_counters_base_addr_;
    uint32_t to_remote_write_counters_base_addr_;
};

struct FSMContext {
    bool active = false;
    uint32_t state = 0;
};

class FabricControlChannel {
public:
    void init() {
        fsm_context.active = true;
        // TODO: Initialize inbound and outbound interfaces with proper addresses
        // host_inbound_interface.init(...);
        // eth_inbound_interface.init(...);
        // local_inbound_interface.init(...);
        // eth_outbound_interface.init(...);
        // local_outbound_interface.init(...);
    }

    void process() {
        if (fsm_context.active) {
            // Process the control channel
            fsm_context.state++;
        }

        // in a single step, process 1 packet from each inbound interface

        if (host_inbound_interface_.has_new_packet()) {
            process_control_packet(host_inbound_interface_.get_next_packet());
        }

        if (eth_inbound_interface_.has_new_packet()) {
            process_control_packet(eth_inbound_interface_.get_next_packet());
        }

        if (local_inbound_interface_.has_new_packet()) {
            process_control_packet(local_inbound_interface_.get_next_packet());
        }
    }

    void teardown(uint32_t dummy_addr) {
        fsm_context.active = false;
        host_buffer.init(0);
        eth_buffer.init(0);
        local_buffer.init(0);

        auto status_ptr = reinterpret_cast<volatile uint32_t*>(dummy_addr);
        *status_ptr = fsm_context.state;
    }

private:
    FSMContext fsm_context_;

    // Buffer consumer interfaces
    HostBufferConsumerInterface<HOST_BUFFER_SLOTS> host_inbound_interface_;
    EthBufferConsumerInterface<ETH_BUFFER_SLOTS> eth_inbound_interface_;
    LocalBufferConsumerInterface<NUM_ROUTERS, LOCAL_ROUTER_SLOTS> local_inbound_interface_;

    // Buffer producer interfaces
    EthProducerInterface<ETH_BUFFER_SLOTS> eth_outbound_interface_;
    LocalProducerInterface<NUM_ROUTERS, LOCAL_ROUTER_SLOTS> local_outbound_interface_;

    void forward_control_packet() {
        // forward to eth or local based on destination
        // update relevant buffer ptrs and counters
        // if local, push to arrival queue
    }

    void process_control_packet(uint32_t* packet_start) {
        // process the control packet based on its type and source
        // update the relevant state machines and buffer pointers

        // we may need to forward packets to other routers -> either over eth or in the local queue
        // we will need to check if there is space in the relevant buffers before forwarding
        // for local routers, we should be able to check space against a mask
    }
};

}  // namespace control_channel

}  // namespace tt::tt_fabric
