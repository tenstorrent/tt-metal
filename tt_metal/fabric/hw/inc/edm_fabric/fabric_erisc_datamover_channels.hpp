// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#if defined(COMPILE_FOR_ERISC)
#include "internal/ethernet/tunneling.h"
#endif
#include "api/alignment.h"
#include "internal/risc_attribs.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "fabric_static_channels_ct_args.hpp"
#include "edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/datastructures/outbound_channel.hpp"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"

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

// Configuration flag to select channel buffer implementation
// For now, hardcoded to use static sized buffers
// In the future, this could come from compile-time args or build configuration
constexpr bool USE_STATIC_SIZED_CHANNEL_BUFFERS = true;

// A base Ethernet channel buffer interface class that will be specialized for different
// channel architectures (e.g. static vs elastic sizing)
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS, typename DERIVED_T>
class EthChannelBufferInterface {
public:
    explicit EthChannelBufferInterface() = default;

    FORCE_INLINE void init(size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        static_cast<DERIVED_T*>(this)->init_impl(channel_base_address, max_eth_payload_size_in_bytes, header_size_bytes);
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->get_buffer_address_impl(buffer_index);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE volatile T* get_packet_header(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->template get_packet_header_impl<T>(buffer_index);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE size_t get_payload_size(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->template get_payload_size_impl<T>(buffer_index);
    }

    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->get_channel_buffer_max_size_in_bytes_impl(buffer_index);
    }

    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size() const {
        return static_cast<const DERIVED_T*>(this)->get_max_eth_payload_size_impl();
    }

#if defined(COMPILE_FOR_ERISC)
    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed(const BufferIndex& buffer_index) const {
        return static_cast<const DERIVED_T*>(this)->eth_is_acked_or_completed_impl(buffer_index);
    }
#endif

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const {
        return static_cast<const DERIVED_T*>(this)->get_cached_next_buffer_slot_addr_impl();
    }

    FORCE_INLINE void set_cached_next_buffer_slot_addr(size_t next_buffer_slot_addr) {
        static_cast<DERIVED_T*>(this)->set_cached_next_buffer_slot_addr_impl(next_buffer_slot_addr);
    }
};

// Elastic sender channel implementation (stub for now)
// Issue #26311
template <typename HEADER_TYPE>
class ElasticSenderEthChannel : public SenderEthChannelInterface<HEADER_TYPE, 0, ElasticSenderEthChannel<HEADER_TYPE>> {
public:
    explicit ElasticSenderEthChannel() = default;

    FORCE_INLINE void init_impl(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        // TODO: Issue #26311
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const {
        // TODO: Issue #26311
        return 0;
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr_impl() {
        // TODO: Issue #26311
    }
};

// Elastic channel buffer implementation (stub for now)
// Issue #26311
template <typename HEADER_TYPE>
class ElasticEthChannelBuffer : public EthChannelBufferInterface<HEADER_TYPE, 0, ElasticEthChannelBuffer<HEADER_TYPE>> {
public:
    explicit ElasticEthChannelBuffer() = default;

    FORCE_INLINE void init_impl(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) {
        // TODO: Issue #26311
        // Dynamic buffer allocation logic would go here
    }

    FORCE_INLINE size_t get_buffer_address_impl(const BufferIndex& buffer_index) const {
        // TODO: Issue #26311
        return 0;
    }

    FORCE_INLINE size_t get_max_eth_payload_size_impl() const {
        // TODO: Issue #26311
        return 0;
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const {
        // TODO: Issue #26311
        return 0;
    }

    FORCE_INLINE void set_cached_next_buffer_slot_addr_impl(size_t addr) {
        // TODO: Issue #26311
    }
};

// This class implements the interface for static sized receiver/Ethernet channels.
// Static sized channels have a fixed number of buffer slots, defined
// at router initialization, and persistent for the lifetime of the router.
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class StaticSizedEthChannelBuffer : public EthChannelBufferInterface<
                                        HEADER_TYPE,
                                        NUM_BUFFERS,
                                        StaticSizedEthChannelBuffer<HEADER_TYPE, NUM_BUFFERS>> {
public:
    // The channel structure is as follows:
    //              &header->  |----------------| channel_base_address
    //                         |    header      |
    //             &payload->  |----------------|
    //                         |                |
    //                         |    payload     |
    //                         |                |
    //                         |----------------|

    explicit StaticSizedEthChannelBuffer() = default;

    FORCE_INLINE void init_impl(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) {
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
            set_cached_next_buffer_slot_addr_impl(this->buffer_addresses[0]);
        }
    }

    StaticSizedEthChannelBuffer(size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes) {
        this->init(channel_base_address, buffer_size_bytes, header_size_bytes);
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address_impl(const BufferIndex& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE volatile T* get_packet_header_impl(const BufferIndex& buffer_index) const {
        return reinterpret_cast<volatile T*>(this->buffer_addresses[buffer_index]);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE size_t get_payload_size_impl(const BufferIndex& buffer_index) const {
        return get_packet_header_impl<T>(buffer_index)->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes_impl(const BufferIndex& buffer_index) const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size_impl() const { return this->max_eth_payload_size_in_bytes; }

#if defined(COMPILE_FOR_ERISC)
    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed_impl(const BufferIndex& buffer_index) const {
        return eth_is_receiver_channel_send_acked(buffer_index) || eth_is_receiver_channel_send_done(buffer_index);
    }
#endif

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const { return this->cached_next_buffer_slot_addr; }

    FORCE_INLINE void set_cached_next_buffer_slot_addr_impl(size_t next_buffer_slot_addr) {
        this->cached_next_buffer_slot_addr = next_buffer_slot_addr;
    }

    FORCE_INLINE uint32_t channel_base_address() const {
        return static_cast<uint32_t>(this->buffer_addresses[0]);
    }

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;
    std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    std::size_t max_eth_payload_size_in_bytes;
    std::size_t cached_next_buffer_slot_addr;
};

// Helper to select channel buffer implementation based on compile-time flag
template <bool UseStatic, typename HEADER_TYPE, uint8_t NUM_BUFFERS>
struct ChannelBufferTypeSelector {
    using type = std::conditional_t<
        UseStatic,
        StaticSizedEthChannelBuffer<HEADER_TYPE, NUM_BUFFERS>,
        ElasticEthChannelBuffer<HEADER_TYPE>>;
};

template <bool UseStatic, typename HEADER_TYPE, uint8_t NUM_BUFFERS>
struct SenderChannelTypeSelector {
    using type = std::conditional_t<
        UseStatic,
        StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>,
        ElasticSenderEthChannel<HEADER_TYPE>
    >;
};

// For backward compatibility until Issue #26311 completed
// - when NUM_BUFFERS is known at compile time, else can provide a dummy value
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using EthChannelBuffer = typename ChannelBufferTypeSelector<
    USE_STATIC_SIZED_CHANNEL_BUFFERS, HEADER_TYPE, NUM_BUFFERS>::type;

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using SenderEthChannel = typename SenderChannelTypeSelector<
    USE_STATIC_SIZED_CHANNEL_BUFFERS, HEADER_TYPE, NUM_BUFFERS>::type;

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

template <template <typename, size_t> class ChannelBase, typename HEADER_TYPE, auto& ChannelBuffers>
struct StaticSizedChannelBuffersHelper {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        return ChannelTuple<ChannelBase, HEADER_TYPE, ChannelBuffers[Is]...>{};
    }
};

template <template <typename, size_t> class ChannelBase, typename HEADER_TYPE>
struct ChannelBuffersBase {
    // Generic interface that derived types will specialize
};

// Static-sized variants using the new naming
template <typename HEADER_TYPE, auto& ChannelBuffers>
using StaticSizedEthChannelBuffers =
    StaticSizedChannelBuffersHelper<StaticSizedEthChannelBuffer, HEADER_TYPE, ChannelBuffers>;

template <typename HEADER_TYPE, auto& ChannelBuffers>
using StaticSizedSenderEthChannelBuffers =
    StaticSizedChannelBuffersHelper<StaticSizedSenderEthChannel, HEADER_TYPE, ChannelBuffers>;

// Elastic channel buffer helpers that ignore the compile-time buffer array
template <typename HEADER_TYPE, auto& ChannelBuffers>
struct ElasticEthChannelBuffersHelper {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        // Create a tuple of ElasticEthChannelBuffer instances
        // The number of channels is determined by the index sequence size
        // Use parameter pack expansion with comma operator to create N instances
        return std::tuple<decltype((void)Is, ElasticEthChannelBuffer<HEADER_TYPE>{})...>{
            ((void)Is, ElasticEthChannelBuffer<HEADER_TYPE>{})...
        };
    }
};

template <typename HEADER_TYPE, auto& ChannelBuffers>
struct ElasticSenderEthChannelBuffersHelper {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        // Create a tuple of ElasticSenderEthChannel instances
        // Use parameter pack expansion with comma operator to create N instances
        return std::tuple<decltype((void)Is, ElasticSenderEthChannel<HEADER_TYPE>{})...>{
            ((void)Is, ElasticSenderEthChannel<HEADER_TYPE>{})...
        };
    }
};

// Elastic channel buffer aliases
template <typename HEADER_TYPE, auto& ChannelBuffers>
using ElasticEthChannelBuffers = ElasticEthChannelBuffersHelper<HEADER_TYPE, ChannelBuffers>;

template <typename HEADER_TYPE, auto& ChannelBuffers>
using ElasticSenderEthChannelBuffers = ElasticSenderEthChannelBuffersHelper<HEADER_TYPE, ChannelBuffers>;


template <typename HEADER_TYPE, auto& ChannelBuffers>
using EthChannelBuffers = std::conditional_t<
    USE_STATIC_SIZED_CHANNEL_BUFFERS,
    StaticSizedEthChannelBuffers<HEADER_TYPE, ChannelBuffers>,
    ElasticEthChannelBuffers<HEADER_TYPE, ChannelBuffers>
>;

// Generic channel type selector - works for ANY channel type (sender or receiver)
template <typename HEADER_TYPE, FabricChannelPoolType PoolType, size_t NumBuffers,
          template<typename, size_t> class StaticType, typename ElasticType>
struct GenericChannelTypeSelector {
    using type = std::conditional_t<
        PoolType == FabricChannelPoolType::STATIC,
        StaticType<HEADER_TYPE, NumBuffers>,
        ElasticType
    >;
};

// Backward compatibility alias for existing code
template <typename HEADER_TYPE, FabricChannelPoolType PoolType, size_t NumBuffers>
using ChannelTypeSelector = GenericChannelTypeSelector<
    HEADER_TYPE, PoolType, NumBuffers,
    StaticSizedSenderEthChannel, ElasticSenderEthChannel<HEADER_TYPE>
>;

template <typename HEADER_TYPE, typename ChannelPoolCollection,
          template<typename, size_t> class StaticChannelType, typename ElasticChannelType,
          auto& ChannelToPoolIndex,
          typename IndexSequence>
struct HeterogeneousChannelBuilder {
    using type = std::tuple<>;
};

template <typename HEADER_TYPE, typename ChannelPoolCollection,
          template<typename, size_t> class StaticChannelType, typename ElasticChannelType,
          auto& ChannelToPoolIndex,
          size_t... Indices>
struct HeterogeneousChannelBuilder<HEADER_TYPE, ChannelPoolCollection,
                                   StaticChannelType, ElasticChannelType,
                                   ChannelToPoolIndex,
                                   std::index_sequence<Indices...>> {
    // Extract pool type for each channel using provided mapping
    template <size_t ChannelIdx>
    FORCE_INLINE static constexpr FabricChannelPoolType get_channel_pool_type() {
        constexpr size_t pool_idx = ChannelToPoolIndex[ChannelIdx];
        return static_cast<FabricChannelPoolType>(ChannelPoolCollection::channel_pool_types[pool_idx]);
    }

    // Get buffer count for static channels by extracting from pool data
    template <size_t ChannelIdx>
    FORCE_INLINE static constexpr size_t get_buffer_count() {
        constexpr auto pool_type = get_channel_pool_type<ChannelIdx>();
        if constexpr (pool_type == FabricChannelPoolType::STATIC) {
            // Get the pool index for this channel
            constexpr size_t pool_idx = ChannelToPoolIndex[ChannelIdx];

            // Extract the actual pool from the PoolsTuple
            using PoolType = std::tuple_element_t<pool_idx, typename ChannelPoolCollection::PoolsTuple>;

            // Return the actual num_slots from the pool
            return PoolType::num_slots;
        } else {
            return 0; // Elastic channels don't use compile-time buffer counts
        }
    }

    // Build the heterogeneous tuple using the generic selector
    using type = std::tuple<typename GenericChannelTypeSelector<
        HEADER_TYPE,
        get_channel_pool_type<Indices>(),
        get_buffer_count<Indices>(),
        StaticChannelType,
        ElasticChannelType>::type...>;
};

// Generic heterogeneous channel tuple wrapper - works for any channel mapping
template <typename HEADER_TYPE, typename ChannelPoolCollection, typename ChannelTypes, auto& ChannelToPoolIndex>
struct HeterogeneousChannelTuple {
    static constexpr size_t const num_channels = std::tuple_size_v<ChannelTypes>;

    ChannelTypes channel_buffers;

    explicit HeterogeneousChannelTuple() = default;

    template <typename PoolCollection>
    FORCE_INLINE void init(size_t buffer_size_bytes, size_t header_size_bytes) {
        init_impl<PoolCollection>(buffer_size_bytes, header_size_bytes, std::make_index_sequence<std::tuple_size_v<ChannelTypes>>());
    }

    template <size_t I>
    FORCE_INLINE auto& get() {
        return std::get<I>(channel_buffers);
    }

private:
    template <typename PoolCollection, size_t... ChannelIndices>
    FORCE_INLINE void init_impl(
        size_t buffer_size_bytes, size_t header_size_bytes, std::index_sequence<ChannelIndices...>) {
        (init_single_channel<PoolCollection, ChannelIndices>(
             std::get<ChannelIndices>(channel_buffers), buffer_size_bytes, header_size_bytes),
         ...);
    }

    template <typename PoolCollection, size_t ChannelIdx, typename Channel>
    FORCE_INLINE void init_single_channel(Channel& chan, size_t buffer_size_bytes, size_t header_size_bytes) {
        // Get pool index for this channel using provided mapping
        constexpr size_t pool_idx = ChannelToPoolIndex[ChannelIdx];
        constexpr auto pool_type = static_cast<FabricChannelPoolType>(
            PoolCollection::channel_pool_types[pool_idx]);

        if constexpr (pool_type == FabricChannelPoolType::STATIC) {
            // Static channel: get base address from the pool
            using PoolType = std::tuple_element_t<pool_idx, typename PoolCollection::PoolsTuple>;
            chan.init(PoolType::base_address, buffer_size_bytes, header_size_bytes);
        } else {
            // Elastic channel: get address from elastic pool chunks
            using PoolType = std::tuple_element_t<pool_idx, typename PoolCollection::PoolsTuple>;
            // For now, use first chunk address (elastic channel logic TBD)
            constexpr size_t chunk_address = PoolType::chunk_base_addresses[0];
            chan.init(chunk_address, buffer_size_bytes, header_size_bytes);
        }
    }
};

template <
    typename HEADER_TYPE,
    typename ChannelPoolCollection,
    auto& PoolTypes,  // Keep for validation
    template <typename, size_t> class StaticChannelType,
    typename ElasticChannelType,
    auto& ChannelToPoolIndex>
struct MultiPoolChannelBuffers {
    static constexpr size_t num_channels = ChannelToPoolIndex.size();

    // Compile-time validation
    static_assert(num_channels > 0, "Must have at least one channel");
    static_assert(num_channels == PoolTypes.size(), "PoolTypes array size must match number of channels");

    using ChannelTypes = typename HeterogeneousChannelBuilder<
        HEADER_TYPE,
        ChannelPoolCollection,
        StaticChannelType,
        ElasticChannelType,
        ChannelToPoolIndex,
        std::make_index_sequence<num_channels>>::type;

    template <size_t... Is>
    FORCE_INLINE static auto make(std::index_sequence<Is...>) {
        return HeterogeneousChannelTuple<HEADER_TYPE, ChannelPoolCollection, ChannelTypes, ChannelToPoolIndex>{};
    }

    FORCE_INLINE static auto make() { return make(std::make_index_sequence<num_channels>{}); }

    // Access individual channels by index
    template <size_t Index>
    FORCE_INLINE static constexpr auto get_channel_type() {
        static_assert(Index < num_channels, "Channel index out of bounds");
        return std::tuple_element_t<Index, ChannelTypes>{};
    }
};

// Template aliases for specific channel types
template <typename HEADER_TYPE, typename ChannelPoolCollection, auto& PoolTypes, auto& ChannelToPoolIndex>
using MultiPoolSenderEthChannelBuffers = MultiPoolChannelBuffers<
    HEADER_TYPE, ChannelPoolCollection, PoolTypes,
    StaticSizedSenderEthChannel, ElasticSenderEthChannel<HEADER_TYPE>,
    ChannelToPoolIndex>;

template <typename HEADER_TYPE, typename ChannelPoolCollection, auto& PoolTypes, auto& ChannelToPoolIndex>
using MultiPoolEthChannelBuffers = MultiPoolChannelBuffers<
    HEADER_TYPE, ChannelPoolCollection, PoolTypes,
    StaticSizedEthChannelBuffer, ElasticEthChannelBuffer<HEADER_TYPE>,
    ChannelToPoolIndex>;

// Backward compatibility: keep the old interface for existing code
template <typename HEADER_TYPE, auto& ChannelBuffers>
using SenderEthChannelBuffers = std::conditional_t<
    USE_STATIC_SIZED_CHANNEL_BUFFERS,
    StaticSizedSenderEthChannelBuffers<HEADER_TYPE, ChannelBuffers>,
    ElasticSenderEthChannelBuffers<HEADER_TYPE, ChannelBuffers>
>;

// Base class for channel worker interfaces
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
    template <bool posted = false, bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE void teardown_worker_connection() const {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
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

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t MY_ETH_CHANNEL = USE_DYNAMIC_CREDIT_ADDR>
    FORCE_INLINE void cache_producer_noc_addr() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address;
        worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        return *connection_live_semaphore == tt::tt_fabric::connection_interface::close_connection_request_value;
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
        // TODO: Issue #26311
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_local_read_counter() const {
        // TODO: Issue #26311
        return 0;
    }

    FORCE_INLINE void increment_read_counter_impl(int32_t inc_val) {
        // TODO: Issue #26311
    }

    template <bool enable_noc_flush = true>
    FORCE_INLINE void notify_worker_of_read_counter_update_impl() {
        // TODO: Issue #26311
    }

    template <bool enable_deadlock_avoidance>
    FORCE_INLINE void notify_persistent_connection_of_free_space_impl(int32_t inc_val) {
        // TODO: Issue #26311
    }

    template <bool SKIP_CONNECTION_LIVENESS_CHECK>
    FORCE_INLINE void update_write_counter_for_send() {
        // TODO: Issue #26311
    }

    [[nodiscard]] FORCE_INLINE BufferIndex get_write_buffer_index() const {
        // TODO: Issue #26311
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
