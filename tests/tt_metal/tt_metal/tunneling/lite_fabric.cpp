// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include <utility>

#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include "blackhole/dev_mem_map.h"
#include "blackhole/noc_nonblocking_api.h"
#include "dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.hpp"
#include "firmware_common.h"
#include "init-fsm-basic.h"
#include "lite_fabric_constants.hpp"
#include "lite_fabric_channels.hpp"
#include "lite_fabric_channel_util.hpp"
#include "lite_fabric_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#define BEGIN_MAIN_FUNCTION()                                                                \
    IF_NOT_METAL_LAUNCH(int main())                                                          \
    IF_METAL_LAUNCH(void kernel_main()) {                                                    \
        IF_NOT_METAL_LAUNCH(configure_csr();)                                                \
        IF_NOT_METAL_LAUNCH(noc_index = NOC_INDEX;)                                          \
        IF_NOT_METAL_LAUNCH(do_crt1((uint32_t*)MEM_LITE_FABRIC_INIT_LOCAL_L1_BASE_SCRATCH);) \
        IF_NOT_METAL_LAUNCH(noc_bank_table_init(MEM_LITE_FABRIC_BANK_TO_NOC_SCRATCH);)       \
        IF_NOT_METAL_LAUNCH(risc_init();)                                                    \
        IF_NOT_METAL_LAUNCH(noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);)                          \
        IF_NOT_METAL_LAUNCH(for (uint32_t n = 0; n < NUM_NOCS; n++) { noc_local_state_init(n); })

// End the main function
#define END_MAIN_FUNCTION()        \
    IF_NOT_METAL_LAUNCH(return 0;) \
    IF_METAL_LAUNCH(return;)       \
    }

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

// Define METAL_LAUNCH if lite_fabric is being launched by Metal
#if !defined(METAL_LAUNCH)
#define IF_NOT_METAL_LAUNCH(x) x
#define IF_METAL_LAUNCH(x)

uint8_t noc_index __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[tt::tt_metal::NumHalProgrammableCoreTypes] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere yet
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

#else

#include "debug/dprint.h"

#define IF_NOT_METAL_LAUNCH(x)
#define IF_METAL_LAUNCH(x) x
#endif

BEGIN_MAIN_FUNCTION() {
    invalidate_l1_cache();

    auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(MEM_LITE_FABRIC_CONFIG_BASE);

    volatile lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>& host_interface =
        structs->host_interface;

    const uint32_t lf_local_sender_0_channel_address = (uintptr_t)&structs->sender_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_info_addr = (uintptr_t)&structs->sender_location_info;
    const uint32_t lf_remote_receiver_0_channel_buffer_address = (uintptr_t)&structs->receiver_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_semaphore_addr =
        (uintptr_t)&structs->sender_connection_live_semaphore;
    auto lf_sender0_worker_semaphore_ptr =
        reinterpret_cast<volatile uint32_t*>((uintptr_t)&structs->sender_flow_control_semaphore);

    // One send buffer and one receiver buffer
    init_ptr_val<to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<to_sender_0_pkts_acked_id>(0);
    init_ptr_val<to_sender_0_pkts_completed_id>(0);

    auto remote_receiver_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    // use same addr space for host to lite fabric edm connection
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        RECEIVER_CHANNEL_BASE_ID);
    lite_fabric::init_receiver_headers(remote_receiver_channels);

    // initialize the local sender channel worker interfaces
    local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        SENDER_CHANNEL_BASE_ID);

    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        0,
        lite_fabric::edm_to_local_chip_noc,
        lite_fabric::edm_to_downstream_noc>
        receiver_channel_0_trid_tracker;

    auto outbound_to_receiver_channel_pointers = lite_fabric::
        ChannelPointersTuple<lite_fabric::OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto outbound_to_receiver_channel_pointer_ch0 = outbound_to_receiver_channel_pointers.template get<0>();

    auto receiver_channel_pointers =
        lite_fabric::ChannelPointersTuple<lite_fabric::ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

    // Must initialize to match what's on the host
    structs->host_interface.init();

    bool on_mmio_chip = structs->config.is_mmio;
    DPRINT << "Routing Enabled " << structs->config.routing_enabled
           << " Init on MMIO = " << (uint32_t)structs->config.is_mmio << " Host IF at 0x" << HEX()
           << (uint32_t)&host_interface << DEC() << ENDL();

    lite_fabric::routing_init(&structs->config);

    volatile uint32_t* status_addr = (volatile uint32_t*)(0x1c);

    while (structs->config.routing_enabled) {
        status_addr[0]++;
        invalidate_l1_cache();

        lite_fabric::run_sender_channel_step(
            local_sender_channels.template get<0>(),
            host_interface,
            outbound_to_receiver_channel_pointer_ch0,
            remote_receiver_channels.template get<0>(),
            on_mmio_chip);

        invalidate_l1_cache();
        lite_fabric::run_receiver_channel_step(
            remote_receiver_channels.template get<0>(),
            receiver_channel_pointers_ch0,
            receiver_channel_0_trid_tracker,
            structs,
            local_sender_channels.template get<0>(),
            on_mmio_chip);
    }

    status_addr[0] = 0xdeadbeef;

    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    DPRINT << "lite_fabric: out " << (uint32_t)structs->config.routing_enabled << ENDL();

#if !defined(METAL_LAUNCH)
    lite_fabric::ConnectedRisc1Interface::assert_connected_dm1_reset();
#endif

    structs->config.current_state = lite_fabric::InitState::TERMINATED;
}
END_MAIN_FUNCTION()
