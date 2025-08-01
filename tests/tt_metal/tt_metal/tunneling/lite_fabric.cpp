// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "lite_fabric_constants.hpp"
#include "risc_common.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "firmware_common.h"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "blackhole/dev_mem_map.h"
#include "blackhole/noc_nonblocking_api.h"
#include "tt_metal/hw/inc/blackhole/core_config.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tests/tt_metal/tt_metal/tunneling/fabric_lite.hpp"
#include "debug/dprint.h"
#include "init-fsm-basic.h"

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

#if !defined(NUM_NOCS)
#define NUM_NOCS 2
#endif

#if !defined(NOC_INDEX)
#define NOC_INDEX 0
#endif

// Define METAL_LAUNCH if lite_fabric is being launched by Metal
#if !defined(METAL_LAUNCH)
#define IF_NOT_METAL_LAUNCH(x) x
#define IF_METAL_LAUNCH(x)

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere yet
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

#else
#define IF_NOT_METAL_LAUNCH(x)
#define IF_METAL_LAUNCH(x) x
#endif

// Define a main function for lite_fabric being compiled as firmware or kernel
#define BEGIN_MAIN_FUNCTION()                                                           \
    IF_NOT_METAL_LAUNCH(int main())                                                     \
    IF_METAL_LAUNCH(void kernel_main()) {                                               \
        IF_NOT_METAL_LAUNCH(configure_csr();)                                           \
        IF_NOT_METAL_LAUNCH(do_crt1((uint32_t*)MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH);) \
        IF_NOT_METAL_LAUNCH(noc_bank_table_init(MEM_AERISC_BANK_TO_NOC_SCRATCH);)       \
        IF_NOT_METAL_LAUNCH(risc_init();)                                               \
        IF_NOT_METAL_LAUNCH(noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);)                     \
        IF_NOT_METAL_LAUNCH(for (uint32_t n = 0; n < NUM_NOCS; n++) { noc_local_state_init(n); })

// End the main function
#define END_MAIN_FUNCTION()        \
    IF_NOT_METAL_LAUNCH(return 0;) \
    IF_METAL_LAUNCH(return;)       \
    }

constexpr uint32_t SOFT_RESET_ADDR = 0xFFB121B0;

// Interface to the connected RISC1 processor via ethernet
struct ConnectedRisc1Interface {
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;
    static constexpr uint32_t k_ResetPcAddr = SUBORDINATE_AERISC_RESET_PC;

    template <uint32_t TXQ = 0>
    inline void send_packet(uint32_t src_addr, uint32_t dst_addr, uint32_t n) const {
        do {
            invalidate_l1_cache();
        } while (internal_::eth_txq_is_busy(TXQ));
        internal_::eth_send_packet(TXQ, src_addr >> 4, dst_addr >> 4, n);
    }

    // Put the connected RISC1 into reset
    inline void assert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0x47000;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    // Take the connected RISC1 out of reset
    inline void deassert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    inline void set_pc(uint32_t pc) const { internal_::eth_write_remote_reg(0, k_ResetPcAddr, pc); }
};

constexpr uint32_t local_sender_0_channel_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_channel_buffer);
static_assert(local_sender_0_channel_address % 16 == 0);

constexpr uint32_t remote_receiver_0_channel_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, receiver_channel_buffer);
static_assert(remote_receiver_0_channel_address % 16 == 0);

constexpr uint32_t local_receiver_0_channel_address = remote_receiver_0_channel_address;

constexpr uint32_t local_sender_channel_0_connection_info_addr =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_location_info);
static_assert(local_sender_channel_0_connection_info_addr % 16 == 0);

constexpr uint32_t sender_flow_control_semaphore_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_flow_control_semaphore);
static_assert(sender_flow_control_semaphore_address % 16 == 0);

constexpr uint32_t sender_connection_live_semaphore_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_connection_live_semaphore);
static_assert(sender_connection_live_semaphore_address % 16 == 0);

BEGIN_MAIN_FUNCTION() {
    invalidate_l1_cache();
    // Test values
    auto dptr = reinterpret_cast<volatile uint32_t*>(0x20000);
    dptr[0] = 0xdeadbeef;
    dptr[1] = 0xcafecafe;

    auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(MEM_AERISC_FABRIC_LITE_CONFIG);

    // Initialize stream registers
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_ptr_val<to_receiver_packets_sent_streams[1]>(0);
    init_ptr_val<to_sender_packets_acked_streams[0]>(0);
    init_ptr_val<to_sender_packets_acked_streams[1]>(0);
    init_ptr_val<to_sender_packets_acked_streams[2]>(0);
    init_ptr_val<to_sender_packets_completed_streams[0]>(0);
    init_ptr_val<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<to_sender_packets_completed_streams[2]>(0);
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(CHANNEL_BUFFER_SLOTS);  // LOCAL
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(CHANNEL_BUFFER_SLOTS);  // EAST
    init_ptr_val<sender_channel_free_slots_stream_ids[2]>(CHANNEL_BUFFER_SLOTS);  // WEST
    init_ptr_val<receiver_channel_0_free_slots_from_east_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_west_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_north_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_south_stream_id>(CHANNEL_BUFFER_SLOTS);

    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS_ARRAY[0], NUM_TRANSACTION_IDS, 0> receiver_channel_0_trid_tracker;

    auto local_receiver_channels = tt::tt_fabric::EthChannelBuffers<RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto remote_receiver_channels = tt::tt_fabric::EthChannelBuffers<RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels = tt::tt_fabric::EthChannelBuffers<SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& local_receiver_buffer_addresses = {
        local_receiver_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        remote_receiver_0_channel_address};

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_flow_control_semaphore_addresses = {
        sender_flow_control_semaphore_address};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses = {
        sender_connection_live_semaphore_address};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        local_sender_channel_0_connection_info_addr};

    std::array<uint32_t, NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids_ordered;

    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }

    // populate_local_sender_channel_free_slots_stream_id_ordered_map
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_free_slots_stream_ids_ordered[i] = sender_channel_free_slots_stream_ids[i];
    }

    local_receiver_channels.init(
        local_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(PACKET_HEADER_TYPE),
        RECEIVER_CHANNEL_BASE_ID);

    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(PACKET_HEADER_TYPE),
        RECEIVER_CHANNEL_BASE_ID);

    local_sender_channels.init(
        local_sender_buffer_addresses.data(), CHANNEL_BUFFER_SIZE, sizeof(PACKET_HEADER_TYPE), SENDER_CHANNEL_BASE_ID);

    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    tt::tt_fabric::init_local_sender_channel_worker_interfaces<NUM_SENDER_CHANNELS>(
        local_sender_connection_live_semaphore_addresses,
        local_sender_connection_info_addresses,
        local_sender_channel_worker_interfaces,
        local_sender_flow_control_semaphore_addresses);

    lite_fabric::routing_init(&structs->config);

    while (structs->config.routing_enabled) {
        invalidate_l1_cache();
    }

    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
END_MAIN_FUNCTION()
