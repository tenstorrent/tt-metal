// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "dataflow_api.h"
#if defined(COMPILE_FOR_ERISC)
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#endif
#include "tt_metal/hw/inc/utils/utils.h"
#include "risc_attribs.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "fabric_edm_types.hpp"
#include "edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {
/* Ethernet channel structure is as follows (for both sender and receiver):
              &header->  |----------------|\  <-  channel_base_address
                         |    header      | \
             &payload->  |----------------|  \
                         |                |   |- repeated n times
                         |    payload     |  /
                         |                | /
                         |----------------|/
*/

template <typename T>
FORCE_INLINE auto wrap_increment(T val, size_t max) {
    return (val == max - 1) ? 0 : val + 1;
}

// A base sender channel interface class that will be specialized for different
// channel architectures (e.g. static vs elastic sizing)
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS, typename DERIVED_T>
class SenderEthChannelInterface {
public:
    explicit SenderEthChannelInterface() = default;

    FORCE_INLINE void init(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        static_cast<DERIVED_T*>(this)->init_impl(
            channel_base_address, max_eth_payload_size_in_bytes, header_size_bytes);
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const {
        return static_cast<const DERIVED_T*>(this)->get_cached_next_buffer_slot_addr_impl();
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr() {
        static_cast<DERIVED_T*>(this)->advance_to_next_cached_buffer_slot_addr_impl();
    }
};

// This class implements the interface for static sized sender channels.
// Static sized sender channels have a fixed number of buffer slots, defined
// at router initialization, and persistent for the lifetime of the router.
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class StaticSizedSenderEthChannel : public SenderEthChannelInterface<
                                        HEADER_TYPE,
                                        NUM_BUFFERS,
                                        StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>> {
public:
    explicit StaticSizedSenderEthChannel() = default;

    FORCE_INLINE void init_impl(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        this->next_packet_buffer_index = BufferIndex{0};
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] = channel_base_address + i * max_eth_payload_size_in_bytes;
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
            for (size_t j = 0; j < sizeof(HEADER_TYPE) / sizeof(uint32_t); j++) {
                reinterpret_cast<volatile uint32_t*>(this->buffer_addresses[i])[j] = 0;
            }
        }
        if constexpr (NUM_BUFFERS) {
            cached_next_buffer_slot_addr = this->buffer_addresses[0];
        }
    }

    StaticSizedSenderEthChannel(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) :
        SenderEthChannelInterface<HEADER_TYPE, NUM_BUFFERS, StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>>() {
        this->init(channel_base_address, buffer_size_bytes, header_size_bytes);
    }

    // For sender channel, only need a get_next_packet style
    [[nodiscard]] FORCE_INLINE size_t get_buffer_address_impl() const {
        return this->buffer_addresses[next_packet_buffer_index.get()];
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const { return this->cached_next_buffer_slot_addr; }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr_impl() {
        next_packet_buffer_index = BufferIndex{wrap_increment<NUM_BUFFERS>(next_packet_buffer_index.get())};
        this->cached_next_buffer_slot_addr = this->buffer_addresses[next_packet_buffer_index.get()];
    }

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;
    std::size_t cached_next_buffer_slot_addr;
    BufferIndex next_packet_buffer_index;
};

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using SenderEthChannel = StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>;

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class EthChannelBuffer final {
public:
    // The channel structure is as follows:
    //              &header->  |----------------| channel_base_address
    //                         |    header      |
    //             &payload->  |----------------|
    //                         |                |
    //                         |    payload     |
    //                         |                |
    //                         |----------------|

    explicit EthChannelBuffer() = default;

    FORCE_INLINE void init(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) {
        buffer_size_in_bytes = buffer_size_bytes;
        max_eth_payload_size_in_bytes = buffer_size_in_bytes;

        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] = channel_base_address + i * this->max_eth_payload_size_in_bytes;
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
            for (size_t j = 0; j < sizeof(HEADER_TYPE) / sizeof(uint32_t); j++) {
                reinterpret_cast<volatile uint32_t*>(this->buffer_addresses[i])[j] = 0;
            }
        }
        if constexpr (NUM_BUFFERS) {
            set_cached_next_buffer_slot_addr(this->buffer_addresses[0]);
        }
    }

    EthChannelBuffer(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) {
        init(channel_base_address, buffer_size_bytes, header_size_bytes);
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const BufferIndex& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE volatile T* get_packet_header(const BufferIndex& buffer_index) const {
        return reinterpret_cast<volatile T*>(this->buffer_addresses[buffer_index]);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE size_t get_payload_size(const BufferIndex& buffer_index) const {
        return get_packet_header<T>(buffer_index)->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes(const BufferIndex& buffer_index) const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size() const { return this->max_eth_payload_size_in_bytes; }

#if defined(COMPILE_FOR_ERISC)
    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed(const BufferIndex& buffer_index) const {
        return eth_is_receiver_channel_send_acked(buffer_index) || eth_is_receiver_channel_send_done(buffer_index);
    }
#endif

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const { return this->cached_next_buffer_slot_addr; }

    FORCE_INLINE void set_cached_next_buffer_slot_addr(size_t next_buffer_slot_addr) {
        this->cached_next_buffer_slot_addr = next_buffer_slot_addr;
    }

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;

    // header + payload regions only
    std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    std::size_t max_eth_payload_size_in_bytes;
    std::size_t cached_next_buffer_slot_addr;
};

template <template <typename, size_t> class ChannelBase, typename HEADER_TYPE, size_t... BufferSizes>
struct ChannelTuple {
    std::tuple<ChannelBase<HEADER_TYPE, BufferSizes>...> channel_buffers;

    explicit ChannelTuple() = default;

    void init(
        const size_t channel_base_address[],
        const size_t buffer_size_bytes,
        const size_t header_size_bytes,
        const size_t channel_base_id) {
        size_t idx = 0;

        std::apply(
            [&](auto&... chans) {
                ((chans.init(channel_base_address[idx], buffer_size_bytes, header_size_bytes), ++idx), ...);
            },
            channel_buffers);
    }

    template <size_t I>
    auto& get() {
        return std::get<I>(channel_buffers);
    }
};

// Specific aliases
template <typename HEADER_TYPE, size_t... BufferSizes>
using EthChannelBufferTuple = ChannelTuple<tt::tt_fabric::EthChannelBuffer, HEADER_TYPE, BufferSizes...>;

template <typename HEADER_TYPE, size_t... BufferSizes>
using SenderEthChannelTuple = ChannelTuple<tt::tt_fabric::SenderEthChannel, HEADER_TYPE, BufferSizes...>;

// Generic template for channel buffers helpers
template <template <typename, size_t> class ChannelBase, typename HEADER_TYPE, auto& ChannelBuffers>
struct ChannelBuffersHelper {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        return ChannelTuple<ChannelBase, HEADER_TYPE, ChannelBuffers[Is]...>{};
    }
};

// Specific aliases for helpers
template <typename HEADER_TYPE, auto& ChannelBuffers>
using EthChannelBuffers = ChannelBuffersHelper<tt::tt_fabric::EthChannelBuffer, HEADER_TYPE, ChannelBuffers>;

template <typename HEADER_TYPE, auto& ChannelBuffers>
using SenderEthChannelBuffers = ChannelBuffersHelper<tt::tt_fabric::SenderEthChannel, HEADER_TYPE, ChannelBuffers>;

// Base class for channel worker interfaces using CRTP (Curiously Recurring Template Pattern).
// This base class contains members and methods that don't depend on NUM_BUFFERS.
// Derived classes implement specific counter management strategies.
template <uint8_t WORKER_HANDSHAKE_NOC, typename DERIVED>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(nullptr),
        sender_sync_noc_cmd_buf(write_at_cmd_buf) {}

    EdmChannelWorkerInterface(
        // TODO: PERF: See if we can make this non-volatile and then only
        // mark it volatile when we know we need to reload it (i.e. after we receive a
        // "done" message from sender)
        // Have a volatile update function that only triggers after reading the volatile
        // completion field so that way we don't have to do a volatile read for every
        // packet... Then we'll also be able to cache the uint64_t addr of the worker
        // semaphore directly (saving on regenerating it each time)
        volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t* const connection_live_semaphore,
        uint8_t sender_sync_noc_cmd_buf,
        uint8_t edm_read_counter_initial_value) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(connection_live_semaphore),
        sender_sync_noc_cmd_buf(sender_sync_noc_cmd_buf) {
        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_read_counter)) = edm_read_counter_initial_value;
        static_cast<DERIVED*>(this)->reset_counters();
    }

    // Flow control methods
    //
    // local_wrptr trails from_remote_wrptr
    // we have new data if they aren't equal

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }

    // Only used for persistent connections (i.e. upstream is EDM)
    template <bool enable_deadlock_avoidance>
    FORCE_INLINE void notify_persistent_connection_of_free_space(int32_t inc_val) {
        static_cast<DERIVED*>(this)
            ->template notify_persistent_connection_of_free_space_impl<enable_deadlock_avoidance>(inc_val);
    }

    template <bool enable_noc_flush = true>
    FORCE_INLINE void notify_worker_of_read_counter_update() {
        static_cast<DERIVED*>(this)->template notify_worker_of_read_counter_update_impl<enable_noc_flush>();
    }

    FORCE_INLINE void increment_local_read_counter(int32_t inc_val) {
        static_cast<DERIVED*>(this)->increment_read_counter_impl(inc_val);
    }

    FORCE_INLINE void copy_read_counter_to_worker_location_info() const {
        worker_location_info_ptr->edm_read_counter = static_cast<const DERIVED*>(this)->get_local_read_counter();
    }

    // Connection management methods
    //
    template <bool posted = false>
    FORCE_INLINE void teardown_worker_connection() const {
        invalidate_l1_cache();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x,
            (uint32_t)worker_info.worker_xy.y,
            worker_info.worker_teardown_semaphore_address);

        // Set connection to unused so it's available for next worker
        *this->connection_live_semaphore = tt::tt_fabric::connection_interface::unused_connection_value;

        this->copy_read_counter_to_worker_location_info();

        noc_semaphore_inc<posted>(worker_semaphore_address, 1, WORKER_HANDSHAKE_NOC);
    }

    template <uint8_t MY_ETH_CHANNEL = USE_DYNAMIC_CREDIT_ADDR>
    FORCE_INLINE void cache_producer_noc_addr() {
        invalidate_l1_cache();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address;
        worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        invalidate_l1_cache();
        return *connection_live_semaphore == tt::tt_fabric::connection_interface::close_connection_request_value;
    }
    [[nodiscard]] FORCE_INLINE bool connection_is_live() const {
        invalidate_l1_cache();
        return *connection_live_semaphore == tt::tt_fabric::connection_interface::open_connection_value;
    }

    volatile tt_l1_ptr EDMChannelWorkerLocationInfo* worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t* const connection_live_semaphore;
    uint8_t sender_sync_noc_cmd_buf;
};

// Derived class for static-sized sender channels with fixed number of buffer slots.
// This implements the interface for channels with a known, fixed NUM_BUFFERS at compile time.
template <uint8_t WORKER_HANDSHAKE_NOC, uint8_t NUM_BUFFERS>
struct StaticSizedSenderChannelWorkerInterface
    : public EdmChannelWorkerInterface<
          WORKER_HANDSHAKE_NOC,
          StaticSizedSenderChannelWorkerInterface<WORKER_HANDSHAKE_NOC, NUM_BUFFERS>> {
    using Base = EdmChannelWorkerInterface<
        WORKER_HANDSHAKE_NOC,
        StaticSizedSenderChannelWorkerInterface<WORKER_HANDSHAKE_NOC, NUM_BUFFERS>>;

    static constexpr uint8_t num_buffers = NUM_BUFFERS;

    StaticSizedSenderChannelWorkerInterface() : Base(), read_counter_update_src_address(0) {}

    StaticSizedSenderChannelWorkerInterface(
        volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t* const remote_producer_write_counter,
        volatile tt_l1_ptr uint32_t* const connection_live_semaphore,
        uint8_t sender_sync_noc_cmd_buf,
        uint8_t edm_read_counter_initial_value,
        uint32_t read_counter_update_src_address = 0) :
        Base(
            worker_location_info_ptr,
            connection_live_semaphore,
            sender_sync_noc_cmd_buf,
            edm_read_counter_initial_value),
        read_counter_update_src_address(read_counter_update_src_address) {}

    // CRTP implementation methods
    FORCE_INLINE void reset_counters() {
        local_write_counter.reset();
        local_read_counter.reset();
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_local_read_counter() const { return local_read_counter.counter; }

    FORCE_INLINE void increment_read_counter_impl(int32_t inc_val) { local_read_counter.counter += inc_val; }

    template <bool enable_noc_flush = true>
    FORCE_INLINE void notify_worker_of_read_counter_update_impl() {
        noc_inline_dw_write<InlineWriteDst::L1, true, enable_noc_flush>(
            this->cached_worker_semaphore_address,
            local_read_counter.counter,
            0xf,
            WORKER_HANDSHAKE_NOC,
            NOC_UNICAST_WRITE_VC,
            read_counter_update_src_address);
    }

    template <bool enable_deadlock_avoidance>
    FORCE_INLINE void notify_persistent_connection_of_free_space_impl(int32_t inc_val) {
        auto packed_val = pack_value_for_inc_on_write_stream_reg_write(inc_val);
        noc_inline_dw_write<InlineWriteDst::REG, true>(
            this->cached_worker_semaphore_address, packed_val, 0xf, WORKER_HANDSHAKE_NOC);
    }

    // Write counter management methods
    template <bool SKIP_CONNECTION_LIVENESS_CHECK>
    FORCE_INLINE void update_write_counter_for_send() {
        if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
            // For persistent connections, only update buffer index
            local_write_counter.index = BufferIndex{wrap_increment<NUM_BUFFERS>(local_write_counter.index.get())};
        } else {
            // For non-persistent connections, increment full counter
            local_write_counter.increment();
        }
    }

    [[nodiscard]] FORCE_INLINE BufferIndex get_write_buffer_index() const { return local_write_counter.index; }

    uint32_t read_counter_update_src_address;
    ChannelCounter<NUM_BUFFERS> local_write_counter;
    ChannelCounter<NUM_BUFFERS> local_read_counter;
};

// Stub for elastic-sized sender channels. This is a placeholder for future implementation
// where channel buffer allocation can change dynamically at runtime.
template <uint8_t WORKER_HANDSHAKE_NOC>
struct ElasticSenderChannelWorkerInterface : public EdmChannelWorkerInterface<
                                                 WORKER_HANDSHAKE_NOC,
                                                 ElasticSenderChannelWorkerInterface<WORKER_HANDSHAKE_NOC>> {
    using Base =
        EdmChannelWorkerInterface<WORKER_HANDSHAKE_NOC, ElasticSenderChannelWorkerInterface<WORKER_HANDSHAKE_NOC>>;

    ElasticSenderChannelWorkerInterface() : Base() {}

    // CRTP implementation methods (stubs)
    FORCE_INLINE void reset_counters() {
        // TODO: Implement elastic counter management
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_local_read_counter() const {
        // TODO: Implement elastic counter management
        return 0;
    }

    FORCE_INLINE void increment_read_counter_impl(int32_t inc_val) {
        // TODO: Implement elastic counter management
    }

    template <bool enable_noc_flush = true>
    FORCE_INLINE void notify_worker_of_read_counter_update_impl() {
        // TODO: Implement elastic counter management
    }

    template <bool enable_deadlock_avoidance>
    FORCE_INLINE void notify_persistent_connection_of_free_space_impl(int32_t inc_val) {
        // TODO: Implement elastic counter management
    }

    // Write counter management methods (stubs)
    template <bool SKIP_CONNECTION_LIVENESS_CHECK>
    FORCE_INLINE void update_write_counter_for_send() {
        // TODO: Implement elastic counter management
    }

    [[nodiscard]] FORCE_INLINE BufferIndex get_write_buffer_index() const {
        // TODO: Implement elastic counter management
        return BufferIndex{0};
    }
};

// A tuple of EDM channel worker interfaces (using static-sized implementation)
template <uint8_t WORKER_HANDSHAKE_NOC, size_t... BufferSizes>
struct EdmChannelWorkerInterfaceTuple {
    // tuple of StaticSizedSenderChannelWorkerInterface<BufferSizes>...
    std::tuple<tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<WORKER_HANDSHAKE_NOC, BufferSizes>...>
        channel_worker_interfaces;

    template <size_t I>
    auto& get() {
        return std::get<I>(channel_worker_interfaces);
    }
};

template <uint8_t WORKER_HANDSHAKE_NOC, auto& ChannelBuffers>
struct EdmChannelWorkerInterfaces {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        return EdmChannelWorkerInterfaceTuple<WORKER_HANDSHAKE_NOC, ChannelBuffers[Is]...>{};
    }
};

}  // namespace tt::tt_fabric
