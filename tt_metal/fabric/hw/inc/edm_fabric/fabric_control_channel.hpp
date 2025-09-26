// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "hostdevcommon/fabric_common.h"
#include "debug/dprint.h"

namespace tt::tt_fabric {

namespace control_channel {

template <uint8_t NUM_SLOTS>
class FabricControlChannelBuffer {
public:
    explicit FabricControlChannelBuffer() = default;

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const uint8_t slot_index) const {
        return base_address_ + slot_index * slot_size_;
    }

    void init() {
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

    void init() {
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
    void init() {
        host_buffer_ = FabricControlChannelBuffer<NUM_BUFFER_SLOTS>(channel_base_address_);
        local_read_counter_.reset();
    }

    template <typename ProcessorFn>
    FORCE_INLINE bool process_packet(ProcessorFn processor_fn) {
        if (!has_new_packet()) {
            return false;
        }

        uint32_t packet_address = get_next_packet();
        // TODO: need to check if we care about the return value here
        bool success = processor_fn(reinterpret_cast<ControlPacketHeader*>(packet_address));
        if (success) {
            advance();
        }
        return true;
    }

    HostBufferConsumerInterface() { this->init(); };

private:
    FORCE_INLINE bool has_new_packet() const {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_write_counter_addr_) -
                   local_read_counter_.counter >
               0;
    }

    FORCE_INLINE size_t get_next_packet() const {
        return host_buffer_.get_buffer_address(local_read_counter_.get_buffer_index().get());
    }

    FORCE_INLINE void advance() {
        update_remote_read_counter();
        update_local_read_counter();
    }

    FORCE_INLINE void update_local_read_counter() { local_read_counter_.increment(); }

    FORCE_INLINE void update_remote_read_counter() const {
        // Notify host of read pointer update
        // No noc operations needed, since we are updating an address on our own L1 that host reads back
        *reinterpret_cast<tt_l1_ptr uint32_t*>(to_remote_read_counter_addr_) = local_read_counter_.counter;
    }

    FabricControlChannelBuffer<NUM_BUFFER_SLOTS> host_buffer_;
    ChannelCounter<NUM_BUFFER_SLOTS> local_read_counter_;

    static constexpr size_t channel_base_address_ = control_channel_host_buffer_base_address;
    static constexpr size_t from_remote_write_counter_addr_ = control_channel_host_buffer_remote_write_counter_address;
    static constexpr size_t to_remote_read_counter_addr_ = control_channel_host_buffer_remote_read_counter_address;
};

template <uint8_t NUM_BUFFER_SLOTS>
class EthBufferConsumerInterface {
public:
    void init() {
        eth_buffer_ = FabricControlChannelBuffer<NUM_BUFFER_SLOTS>(channel_base_address_);
        local_read_counter_.reset();
    }

    template <typename ProcessorFn>
    FORCE_INLINE bool process_packet(ProcessorFn processor_fn) {
        if (!has_new_packet()) {
            return false;
        }

        uint32_t packet_address = get_next_packet();
        // TODO: need to check if we care about the return value here
        bool success = processor_fn(reinterpret_cast<ControlPacketHeader*>(packet_address));
        if (success) {
            advance();
        }

        return true;
    }

    EthBufferConsumerInterface() { this->init(); }

private:
    FORCE_INLINE bool has_new_packet() const {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_write_counter_addr_) -
                   local_read_counter_.counter >
               0;
    }

    FORCE_INLINE size_t get_next_packet() const {
        return eth_buffer_.get_buffer_address(local_read_counter_.get_buffer_index().get());
    }

    FORCE_INLINE void advance() {
        update_remote_read_counter();
        update_local_read_counter();
    }

    FORCE_INLINE void update_local_read_counter() { local_read_counter_.increment(); }

    FORCE_INLINE void update_remote_read_counter() const {
        // Push pointer updates over eth
        *reinterpret_cast<tt_l1_ptr uint32_t*>(local_read_counter_addr_) = local_read_counter_.counter;
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            local_read_counter_addr_,
            to_remote_read_counter_addr_,
            ETH_WORD_SIZE_BYTES);
    }

    FabricControlChannelBuffer<NUM_BUFFER_SLOTS> eth_buffer_;
    ChannelCounter<NUM_BUFFER_SLOTS> local_read_counter_;

    static constexpr size_t channel_base_address_ = control_channel_eth_buffer_base_address;
    static constexpr size_t from_remote_write_counter_addr_ = control_channel_eth_buffer_remote_write_counter_address;
    static constexpr size_t to_remote_read_counter_addr_ = control_channel_eth_buffer_remote_read_counter_address;
    static constexpr size_t local_read_counter_addr_ = control_channel_eth_buffer_local_read_counter_address;
};

template <uint8_t NUM_GROUPS, uint8_t NUM_SLOTS_PER_GROUP>
class LocalBufferConsumerInterface {
public:
    void init() {
        local_buffer_ = FabricControlChannelNestedBuffer<NUM_GROUPS, NUM_SLOTS_PER_GROUP>(channel_base_address_);

// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
        for (uint8_t g = 0; g < NUM_GROUPS; g++) {
            local_read_counters_[g].reset();
        }
    }

    template <typename ProcessorFn>
    FORCE_INLINE bool process_packet(ProcessorFn processor_fn) {
        if (!has_new_packet()) {
            return false;
        }

        uint32_t packet_address = get_next_packet();
        // TODO: need to check if we care about the return value here
        bool success = processor_fn(reinterpret_cast<ControlPacketHeader*>(packet_address));
        if (success) {
            advance();
        }

        return true;
    }

    LocalBufferConsumerInterface() { this->init(); }

private:
    FORCE_INLINE bool has_new_packet() {
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

    FORCE_INLINE uint32_t get_group_addr(const uint8_t group_index, const uint32_t base_address) const {
        // we are not caching any addresses to save stack space, so we need to compute them on the fly
        // as future optimization, we can cache these addresses if needed
        return base_address + group_index * sizeof(uint32_t);
    }

    FORCE_INLINE bool group_has_new_packet(const uint8_t group_index) const {
        invalidate_l1_cache();
        uint32_t remote_write_counter_addr = get_group_addr(group_index, from_remote_write_counters_base_addr_);
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_write_counter_addr) -
                   local_read_counters_[group_index].counter >
               0;
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
    uint32_t unprocessed_packet_mask_ = 0;
    uint8_t current_group_index_ = 0;

    static constexpr size_t channel_base_address_ = control_channel_local_buffer_base_address;
    static constexpr size_t from_remote_write_counters_base_addr_ =
        control_channel_local_buffer_remote_write_counter_base_address;
    static constexpr size_t to_remote_read_counters_base_addr_ =
        control_channel_local_buffer_remote_read_counter_base_address;
};

template <uint8_t NUM_BUFFER_SLOTS>
class EthProducerInterface {
public:
    void init() { local_write_counter_.reset(); }

    FORCE_INLINE bool remote_has_space_for_packet() const {
        return local_write_counter_.counter -
                   *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from_remote_read_counter_addr_) <
               NUM_BUFFER_SLOTS;
    }

    // SHOULD ONLY BE CALLED IF REMOTE HAS SPACE
    FORCE_INLINE void forward_packet(uint32_t packet_start_address) {
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            packet_start_address,
            get_remote_buffer_slot_address(),
            sizeof(ControlPacketHeader));

        update_local_write_counter();
        update_remote_write_counter();
    }

    EthProducerInterface() { this->init(); }

private:
    FORCE_INLINE uint32_t get_remote_buffer_slot_address() const {
        return remote_buffer_base_addr_ + (local_write_counter_.get_buffer_index().get() * sizeof(ControlPacketHeader));
    }

    FORCE_INLINE void update_local_write_counter() { local_write_counter_.increment(); }

    FORCE_INLINE void update_remote_write_counter() const {
        *reinterpret_cast<tt_l1_ptr uint32_t*>(local_write_counter_addr_) = local_write_counter_.counter;
        internal_::eth_send_packet_bytes_unsafe(
            0, /* Does this matter for control channel? */
            local_write_counter_addr_,
            to_remote_write_counter_addr_,
            sizeof(uint32_t));
    }

    ChannelCounter<NUM_BUFFER_SLOTS> local_write_counter_;

    static constexpr size_t local_write_counter_addr_ = control_channel_eth_buffer_local_write_counter_address;
    static constexpr size_t from_remote_read_counter_addr_ = control_channel_eth_buffer_remote_read_counter_address;
    static constexpr size_t to_remote_write_counter_addr_ = control_channel_eth_buffer_remote_write_counter_address;
    static constexpr size_t remote_buffer_base_addr_ = control_channel_eth_buffer_base_address;
};

template <uint8_t NUM_GROUPS, uint8_t NUM_SLOTS_PER_GROUP>
class LocalProducerInterface {
public:
    void init() {
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
    FORCE_INLINE void forward_packet(uint32_t packet_start_address, uint32_t router_mask) {
        uint32_t remaining_routers = router_mask;
        while (remaining_routers) {
            uint32_t channel_id = __builtin_ctz(remaining_routers);
            forward_packet_to_remote_router(packet_start_address, channel_id);
            remaining_routers &= ~(1 << channel_id);
        }
    }

    LocalProducerInterface() { this->init(); }

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
        return local_write_counters_[channel_id].counter -
                   *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_read_counter_addr) <
               NUM_SLOTS_PER_GROUP;
    }

    FORCE_INLINE void forward_packet_to_remote_router(uint32_t packet_start_address, uint8_t channel_id) {
        // need to prepare noc address using the target router's noc coords
        uint64_t noc_addr =
            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][channel_id], get_remote_buffer_slot_address(channel_id));
        noc_async_write_one_packet(packet_start_address, noc_addr, sizeof(ControlPacketHeader));
        update_local_write_counter(channel_id);
        update_remote_write_counter(channel_id);
        noc_async_writes_flushed();
    }

    FORCE_INLINE void update_local_write_counter(uint8_t channel_id) { local_write_counters_[channel_id].increment(); }

    FORCE_INLINE void update_remote_write_counter(uint8_t channel_id) const {
        // Update remote router's write counter to notify them of new packet
        uint32_t remote_write_counter_addr = get_group_addr(channel_id, to_remote_write_counters_base_addr_);
        uint64_t noc_addr = get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][channel_id], remote_write_counter_addr);
        noc_inline_dw_write(noc_addr, local_write_counters_[channel_id].counter);
    }

    std::array<ChannelCounter<NUM_SLOTS_PER_GROUP>, NUM_GROUPS> local_write_counters_;

    static constexpr size_t remote_buffer_base_addr_ = control_channel_local_buffer_base_address;
    static constexpr size_t from_remote_read_counters_base_addr_ =
        control_channel_local_buffer_remote_read_counter_base_address;
    static constexpr size_t to_remote_write_counters_base_addr_ =
        control_channel_local_buffer_remote_write_counter_base_address;
};

template <typename Derived>
class BaseFSMContext {
public:
    BaseFSMContext() = default;

    void init() { static_cast<Derived*>(this)->init_impl(); }

    void process(ControlPacketHeader* packet_header, uint32_t staging_packet_address) {
        static_cast<Derived*>(this)->process_impl(packet_header, staging_packet_address);
    }

    bool has_outbound_packet() const { return has_outbound_packet_; }

    void clear_outbound_packet() { has_outbound_packet_ = false; }

    bool is_active() const { return has_outbound_packet_ || static_cast<const Derived*>(this)->is_active_impl(); }

protected:
    bool has_outbound_packet_ = false;
};

class RemoteHeartbeatFSMContext : public BaseFSMContext<RemoteHeartbeatFSMContext> {
public:
    RemoteHeartbeatFSMContext() {
        auto* log_ptr = reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_);
        log_ptr->counter = 0;
        log_ptr->current_state = static_cast<uint8_t>(RemoteHeartbeatFSMState::INVALID);
    }

    void init_impl() {
        state_ = RemoteHeartbeatFSMState::INIT;
        log_state();
    }

    void process_impl(ControlPacketHeader* packet_header, uint32_t staging_packet_address) {
        // drop any non-heartbeat packets
        if (packet_header->type != ControlPacketType::REMOTE_HEARTBEAT) {
            return;
        }

        switch (state_) {
            case RemoteHeartbeatFSMState::INIT:
                if (packet_header->sub_type == ControlPacketSubType::INIT) {
                    prepare_request_packet(packet_header, staging_packet_address);
                    state_ = RemoteHeartbeatFSMState::WAITING_FOR_HEARTBEAT;
                } else if (packet_header->sub_type == ControlPacketSubType::ACK_REQUEST) {
                    prepare_response_packet(packet_header, staging_packet_address);
                    state_ = RemoteHeartbeatFSMState::COMPLETED;
                }  // Drop any other packets

                break;
            case RemoteHeartbeatFSMState::WAITING_FOR_HEARTBEAT:
                // Drop any non-ack response packets
                if (packet_header->sub_type != ControlPacketSubType::ACK_RESPONSE) {
                    break;
                }

                pending_heartbeat_mask_ &= ~(1 << packet_header->src_channel_id);
                if (pending_heartbeat_mask_ == 0) {
                    state_ = RemoteHeartbeatFSMState::COMPLETED;
                    log_counter();
                }

                break;
            default:
                // do nothing, silently drop the packet
                break;
        }

        log_state();
    }

    bool is_active_impl() const { return state_ != RemoteHeartbeatFSMState::COMPLETED; }

private:
    static constexpr uint32_t log_address_ = control_channel_heartbeat_fsm_log_address;

    FORCE_INLINE void log_state() const {
        reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_)->current_state = static_cast<uint8_t>(state_);
    }

    FORCE_INLINE void log_counter() const { reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_)->counter++; }

    void prepare_request_packet(ControlPacketHeader* packet_header, uint32_t staging_packet_address) {
        auto* request_packet = reinterpret_cast<tt_l1_ptr ControlPacketHeader*>(staging_packet_address);
        request_packet->type = ControlPacketType::REMOTE_HEARTBEAT;
        request_packet->sub_type = ControlPacketSubType::ACK_REQUEST;
        request_packet->src_node_id = packet_header->dst_node_id;
        request_packet->dst_node_id = packet_header->context.remote_heartbeat_packet_context.target_node_id;
        request_packet->src_channel_id = packet_header->dst_channel_id;
        request_packet->dst_channel_id = packet_header->context.remote_heartbeat_packet_context.target_channel_id;
        request_packet->sequence_id = packet_header->sequence_id;

        pending_heartbeat_mask_ = 1 << packet_header->context.remote_heartbeat_packet_context.target_channel_id;
        has_outbound_packet_ = true;
    }

    void prepare_response_packet(ControlPacketHeader* packet_header, uint32_t staging_packet_address) {
        auto* response_packet = reinterpret_cast<tt_l1_ptr ControlPacketHeader*>(staging_packet_address);
        response_packet->type = ControlPacketType::REMOTE_HEARTBEAT;
        response_packet->sub_type = ControlPacketSubType::ACK_RESPONSE;
        response_packet->src_node_id = packet_header->dst_node_id;
        response_packet->dst_node_id = packet_header->src_node_id;
        response_packet->src_channel_id = packet_header->dst_channel_id;
        response_packet->dst_channel_id = packet_header->src_channel_id;
        response_packet->sequence_id = packet_header->sequence_id;

        has_outbound_packet_ = true;
    }

    RemoteHeartbeatFSMState state_;
    uint32_t pending_heartbeat_mask_ = 0;
};

class RerouteFSMContext : public BaseFSMContext<RerouteFSMContext> {
public:
    RerouteFSMContext() {
        auto* log_ptr = reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_);
        log_ptr->counter = 0;
        log_ptr->current_state = static_cast<uint8_t>(RerouteFSMState::INVALID);
    }

    void init_impl() {
        state_ = RerouteFSMState::INIT;
        log_state();
    }

    void process_impl(ControlPacketHeader* packet_header, uint32_t staging_packet_address) {
        state_ = RerouteFSMState::COMPLETED;
        log_state();
    }

    bool is_active_impl() const { return state_ != RerouteFSMState::COMPLETED; }

private:
    static constexpr uint32_t log_address_ = control_channel_reroute_fsm_log_address;

    FORCE_INLINE void log_state() const {
        reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_)->current_state = static_cast<uint8_t>(state_);
    }

    FORCE_INLINE void log_counter() const { reinterpret_cast<tt_l1_ptr FSMLog*>(log_address_)->counter++; }

    RerouteFSMState state_;
};

class FSMManager {
public:
    FSMManager() = default;

    void init() {
        reset_active_fsm_type();
        auto* log_ptr = reinterpret_cast<tt_l1_ptr CommonFSMLog*>(common_fsm_log_address_);
        log_ptr->current_fsm_type = FSMType::NONE;
        log_ptr->last_processed_sequence_id = 0;
    }

    FORCE_INLINE bool is_any_fsm_active() const { return active_fsm_type_ != FSMType::NONE; }

    void activate_fsm(ControlPacketType packet_type) {
        // TODO: do we really need to check here if any FSM is active?
        if (is_any_fsm_active()) {
            return;
        }

        switch (packet_type) {
            case ControlPacketType::REMOTE_HEARTBEAT:
                active_fsm_type_ = FSMType::REMOTE_HEARTBEAT;
                remote_heartbeat_fsm_context_.init();
                break;
            case ControlPacketType::REROUTE:
                active_fsm_type_ = FSMType::REROUTE;
                reroute_fsm_context_.init();
                break;
            default: __builtin_unreachable();
        }
    }

    void process(ControlPacketHeader* packet_header) {
        switch (active_fsm_type_) {
            case FSMType::REMOTE_HEARTBEAT:
                remote_heartbeat_fsm_context_.process(packet_header, staging_packet_address_);
                break;
            case FSMType::REROUTE: reroute_fsm_context_.process(packet_header, staging_packet_address_); break;
            default: __builtin_unreachable();
        }
    }

    bool has_outbound_packet() const {
        switch (active_fsm_type_) {
            case FSMType::REMOTE_HEARTBEAT: return remote_heartbeat_fsm_context_.has_outbound_packet();
            case FSMType::REROUTE: return reroute_fsm_context_.has_outbound_packet();
            default: __builtin_unreachable();
        }
    }

    void clear_outbound_packet() {
        switch (active_fsm_type_) {
            case FSMType::REMOTE_HEARTBEAT: remote_heartbeat_fsm_context_.clear_outbound_packet(); break;
            case FSMType::REROUTE: reroute_fsm_context_.clear_outbound_packet(); break;
            default: __builtin_unreachable();
        }
    }

    void check_and_log_fsm_completion(ControlPacketHeader* packet_header) {
        bool fsm_completed = false;
        switch (active_fsm_type_) {
            case FSMType::REMOTE_HEARTBEAT: fsm_completed = !remote_heartbeat_fsm_context_.is_active(); break;
            case FSMType::REROUTE: fsm_completed = !reroute_fsm_context_.is_active(); break;
            default: __builtin_unreachable();
        }

        if (fsm_completed) {
            post_fsm_completion_actions(packet_header);
        } else {
            log_current_fsm_type();
        }
    }

    FORCE_INLINE uint32_t get_outbound_packet_address() const { return staging_packet_address_; }

private:
    static constexpr uint32_t staging_packet_address_ = control_channel_staging_packet_buffer_address;
    static constexpr uint32_t common_fsm_log_address_ = control_channel_common_fsm_log_address;

    FSMType active_fsm_type_ = FSMType::NONE;
    RemoteHeartbeatFSMContext remote_heartbeat_fsm_context_;
    RerouteFSMContext reroute_fsm_context_;

    FORCE_INLINE void log_current_fsm_type() const {
        reinterpret_cast<tt_l1_ptr CommonFSMLog*>(common_fsm_log_address_)->current_fsm_type = active_fsm_type_;
    }

    FORCE_INLINE void log_completed_sequence_id(uint32_t sequence_id) const {
        reinterpret_cast<tt_l1_ptr CommonFSMLog*>(common_fsm_log_address_)->last_processed_sequence_id = sequence_id;
    }

    FORCE_INLINE void reset_active_fsm_type() { active_fsm_type_ = FSMType::NONE; }

    void post_fsm_completion_actions(ControlPacketHeader* packet_header) {
        reset_active_fsm_type();
        log_current_fsm_type();
        log_completed_sequence_id(packet_header->sequence_id);
    }
};

class FabricControlChannel {
public:
    void init() {
        clear_flow_control_counters();

        fsm_manager_.init();

        host_inbound_interface_.init();
        eth_inbound_interface_.init();
        local_inbound_interface_.init();

        eth_outbound_interface_.init();
        local_outbound_interface_.init();

        next_inbound_interface_to_process_ = InterfaceType::HOST;
    }

    void local_handshake() {
        // local handshake with all local routers to ensure that the inbound and outbound interfaces are ready
    }

    // In a single step, we handle 1 outbound packet (if any)and 1 inbound packet
    void run_control_channel_step() {
        handle_outbound_packet();

        process_next_inbound_interface();
    }

    void teardown(uint32_t dummy_addr) {
        // fsm_context.active = false;
        // host_buffer.init(0);
        // eth_buffer.init(0);
        // local_buffer.init(0);

        // auto status_ptr = reinterpret_cast<volatile uint32_t*>(dummy_addr);
        // *status_ptr = fsm_context.state;
    }

private:
    static constexpr uint32_t MY_ETH_CHANNEL_MASK = 1 << MY_ETH_CHANNEL;

    enum class InterfaceType : uint8_t {
        HOST = 0,
        ETH = 1,
        LOCAL = 2,
        NONE = 3,
    };

    // tracks the next inbound interface to process
    InterfaceType next_inbound_interface_to_process_ = InterfaceType::NONE;

    // Finite State Machine Manager
    FSMManager fsm_manager_;

    // Buffer consumer interfaces
    HostBufferConsumerInterface<control_channel_num_host_buffer_slots> host_inbound_interface_;
    EthBufferConsumerInterface<control_channel_num_eth_buffer_slots> eth_inbound_interface_;
    LocalBufferConsumerInterface<control_channel_max_num_eth_cores, control_channel_num_local_buffer_slots>
        local_inbound_interface_;

    // Buffer producer interfaces
    EthProducerInterface<control_channel_num_eth_buffer_slots> eth_outbound_interface_;
    LocalProducerInterface<control_channel_max_num_eth_cores, control_channel_num_local_buffer_slots>
        local_outbound_interface_;

    FORCE_INLINE void clear_flow_control_counters() {
        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_host_buffer_remote_write_counter_address) = 0;
        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_host_buffer_remote_read_counter_address) = 0;

        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_eth_buffer_remote_write_counter_address) = 0;
        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_eth_buffer_remote_read_counter_address) = 0;
        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_eth_buffer_local_write_counter_address) = 0;
        *reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_eth_buffer_local_read_counter_address) = 0;

        auto* control_channel_local_buffer_remote_write_counter_base_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_local_buffer_remote_write_counter_base_address);
        auto* control_channel_local_buffer_remote_read_counter_base_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(control_channel_local_buffer_remote_read_counter_base_address);
        for (size_t i = 0; i < control_channel_max_num_eth_cores; i++) {
            control_channel_local_buffer_remote_write_counter_base_ptr[i] = 0;
            control_channel_local_buffer_remote_read_counter_base_ptr[i] = 0;
        }
    }

    FORCE_INLINE void handle_outbound_packet() {
        if (!fsm_manager_.has_outbound_packet()) {
            return;
        }

        auto packet_address = fsm_manager_.get_outbound_packet_address();
        bool success = forward_control_packet(reinterpret_cast<ControlPacketHeader*>(packet_address));
        if (success) {
            fsm_manager_.clear_outbound_packet();
            fsm_manager_.check_and_log_fsm_completion(reinterpret_cast<ControlPacketHeader*>(packet_address));
        }
    }

    FORCE_INLINE void process_next_inbound_interface() {
        switch (next_inbound_interface_to_process_) {
            case InterfaceType::HOST:
                host_inbound_interface_.process_packet(
                    [this](ControlPacketHeader* packet_header) { return this->forward_control_packet(packet_header); });
                next_inbound_interface_to_process_ = InterfaceType::ETH;
                break;
            case InterfaceType::ETH:
                eth_inbound_interface_.process_packet(
                    [this](ControlPacketHeader* packet_header) { return this->forward_control_packet(packet_header); });
                next_inbound_interface_to_process_ = InterfaceType::LOCAL;
                break;
            case InterfaceType::LOCAL:
                local_inbound_interface_.process_packet(
                    [this](ControlPacketHeader* packet_header) { return this->forward_control_packet(packet_header); });
                next_inbound_interface_to_process_ = InterfaceType::HOST;
                break;
            default: __builtin_unreachable();
        }
    }

    FORCE_INLINE bool forward_packet_via_eth_interface(ControlPacketHeader* packet_header) {
        if (!eth_outbound_interface_.remote_has_space_for_packet()) {
            return false;
        }

        eth_outbound_interface_.forward_packet(reinterpret_cast<uint32_t>(packet_header));
        return true;
    }

    FORCE_INLINE bool forward_packet_via_local_interface(ControlPacketHeader* packet_header, uint32_t channel_mask) {
        if (!local_outbound_interface_.remote_has_space_for_packet(channel_mask)) {
            return false;
        }

        local_outbound_interface_.forward_packet(reinterpret_cast<uint32_t>(packet_header), channel_mask);
        return true;
    }

    FORCE_INLINE void consume_local_control_packet(ControlPacketHeader* packet_header) {
        if (!fsm_manager_.is_any_fsm_active()) {
            // need to peel the packet type to initialize the appropriate FSM?
            fsm_manager_.activate_fsm(packet_header->type);
        }

        fsm_manager_.process(packet_header);
        fsm_manager_.check_and_log_fsm_completion(packet_header);
    }

    InterfaceType get_outbound_interface_type(ControlPacketHeader* packet_header) {
        const auto dst_chip_id = packet_header->dst_node_id.chip_id;
        const auto dst_mesh_id = packet_header->dst_node_id.mesh_id;
        const auto dst_channel_id = packet_header->dst_channel_id;

        const tt_l1_ptr fabric_router_l1_config_t* routing_table =
            reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);

        uint8_t downstream_channel_id = INVALID_DIRECTION;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            // inter-mesh routing
            downstream_channel_id = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
        } else if (dst_chip_id != routing_table->my_device_id) {
            // intra-mesh routing
            downstream_channel_id = routing_table->intra_mesh_table.dest_entry[dst_chip_id];
        } else if (dst_channel_id != MY_ETH_CHANNEL) {
            // local routing
            downstream_channel_id = dst_channel_id;
        }  // else the packet is destined for the same router

        InterfaceType outbound_interface_type = InterfaceType::NONE;
        if (downstream_channel_id == MY_ETH_CHANNEL) {
            outbound_interface_type = InterfaceType::ETH;
        } else if (downstream_channel_id != INVALID_DIRECTION) {
            outbound_interface_type = InterfaceType::LOCAL;
        }

        return outbound_interface_type;
    }

    bool forward_control_packet(ControlPacketHeader* packet_header) {
        const auto outbound_interface_type = get_outbound_interface_type(packet_header);
        bool success = true;

        switch (outbound_interface_type) {
            case InterfaceType::NONE: consume_local_control_packet(packet_header); break;
            case InterfaceType::ETH: success = forward_packet_via_eth_interface(packet_header); break;
            case InterfaceType::LOCAL: {
                // since the dst_channel is specified, the mask will contain only one entry
                const uint32_t channel_mask = 1 << packet_header->dst_channel_id;
                success = forward_packet_via_local_interface(packet_header, channel_mask);
                break;
            }
            default: __builtin_unreachable();
        }

        return success;
    }
};

}  // namespace control_channel

}  // namespace tt::tt_fabric
