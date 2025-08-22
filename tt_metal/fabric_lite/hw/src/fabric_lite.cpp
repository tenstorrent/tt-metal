// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include <utility>

#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include "blackhole/noc_nonblocking_api.h"
#include "dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "firmware_common.h"
#include "tt_metal/fabric_lite/hw/inc/host_interface.hpp"
#include "tt_metal/fabric_lite/hw/inc/init-fsm-basic.hpp"
#include "tt_metal/fabric_lite/hw/inc/constants.hpp"
#include "tt_metal/fabric_lite/hw/inc/channels.hpp"
#include "tt_metal/fabric_lite/hw/inc/channel_util.hpp"
#include "tt_metal/fabric_lite/hw/inc/header.hpp"
#include "tt_metal/fabric_lite/hw/inc/types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

/////////////////////
// Metal globals
/////////////////////
uint8_t noc_index __attribute__((used));

extern uint32_t __ldm_bss_start[];
extern uint32_t __ldm_bss_end[];
extern uint32_t __ldm_data_start[];
extern uint32_t __ldm_data_end[];

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

/////////////////////
// Lite Fabric globals
/////////////////////
namespace fabric_lite {

// Global variable definitions matching extern declarations in fabric_lite_channels.hpp
RemoteReceiverChannelsType remote_receiver_channels __attribute__((used));

LocalSenderChannelsType local_sender_channels __attribute__((used));

// These are used by the other files
bool on_mmio_chip __attribute__((used));

volatile HostInterface* host_interface __attribute__((used));

WriteTridTracker receiver_channel_0_trid_tracker __attribute__((used));

OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple __attribute__((used));

ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple __attribute__((used));

// object_init is expected to be called before this
__attribute__((noinline)) void service_fabric_lite() {
    invalidate_l1_cache();
    if (!reinterpret_cast<volatile fabric_lite::FabricLiteMemoryMap*>(FABRIC_LITE_CONFIG_START)
             ->config.routing_enabled) {
        return;
    }
    reinterpret_cast<uint32_t*>(0x20000)[1]++;
    reinterpret_cast<uint32_t*>(0x20000)[2] = (uint32_t)host_interface;
    reinterpret_cast<uint32_t*>(0x20000)[3] = (uint32_t)&host_interface;
    fabric_lite::run_sender_channel_step();
    fabric_lite::run_receiver_channel_step();
}

inline void object_init(volatile fabric_lite::FabricLiteMemoryMap* mem_map) {
    local_sender_channels =
        tt::tt_fabric::EthChannelBuffers<fabric_lite::FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});
    remote_receiver_channels =
        tt::tt_fabric::EthChannelBuffers<fabric_lite::FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});
    outbound_to_receiver_channel_pointers_tuple = OutboundReceiverChannelPointersTuple::make();
    receiver_channel_pointers_tuple = ReceiverChannelPointersTuple::make();

    const uint32_t lf_local_sender_0_channel_address = (uintptr_t)&mem_map->sender_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_info_addr = (uintptr_t)&mem_map->sender_location_info;
    const uint32_t lf_remote_receiver_0_channel_buffer_address = (uintptr_t)&mem_map->receiver_channel_buffer;
    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    const uint32_t lf_local_sender_channel_0_connection_semaphore_addr =
        (uintptr_t)&mem_map->sender_connection_live_semaphore;
    auto lf_sender0_worker_semaphore_ptr =
        reinterpret_cast<volatile uint32_t*>((uintptr_t)&mem_map->sender_flow_control_semaphore);

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    // Note: Do not use stream 17
    init_ptr_val<to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<to_sender_0_pkts_acked_id>(0);
    init_ptr_val<to_sender_0_pkts_completed_id>(0);

    fabric_lite::remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(fabric_lite::FabricLiteHeader),
        RECEIVER_CHANNEL_BASE_ID);
    fabric_lite::init_receiver_headers(fabric_lite::remote_receiver_channels);

    fabric_lite::local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(fabric_lite::FabricLiteHeader),
        SENDER_CHANNEL_BASE_ID);

    (fabric_lite::receiver_channel_pointers_tuple.template get<0>()).reset();
    fabric_lite::on_mmio_chip = mem_map->config.is_mmio;
    fabric_lite::host_interface = &mem_map->host_interface;
    mem_map->service_fabric_lite_addr = reinterpret_cast<uint32_t>(&service_fabric_lite);
    fabric_lite::host_interface->init();
}

inline void data_init() {
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    wzerorange(__ldm_bss_start, __ldm_bss_end);
}

inline void teardown(volatile fabric_lite::FabricLiteMemoryMap* mem_map) {
    fabric_lite::receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    fabric_lite::ConnectedRisc1Interface::assert_connected_dm1_reset();

    mem_map->config.current_state = fabric_lite::InitState::READY;
}

}  // namespace fabric_lite

int main() {
    invalidate_l1_cache();
    configure_csr();
    noc_index = NOC_INDEX;
    fabric_lite::data_init();
    noc_bank_table_init(FABRIC_LITE_INIT_BANK_TO_NOC_SCRATCH);
    risc_init();
    noc_init(MEM_FABRIC_LITE_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    auto structs = reinterpret_cast<volatile fabric_lite::FabricLiteMemoryMap*>(FABRIC_LITE_CONFIG_START);
    fabric_lite::object_init(structs);
    fabric_lite::routing_init(&structs->config);

    invalidate_l1_cache();
    while (true) {
        fabric_lite::service_fabric_lite();
    }

    fabric_lite::teardown(structs);

    return 0;
}
