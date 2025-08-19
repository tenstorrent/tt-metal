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
#include "lite_fabric.hpp"
#include "firmware_common.h"
#include "init-fsm-basic.h"
#include "lite_fabric_constants.hpp"
#include "lite_fabric_channels.hpp"
#include "lite_fabric_channel_util.hpp"
#include "lite_fabric_header.hpp"
#include "lite_fabric_types.hpp"
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
namespace lite_fabric {

// Global variable definitions matching extern declarations in lite_fabric_channels.hpp
RemoteReceiverChannelsType remote_receiver_channels __attribute__((used));

LocalSenderChannelsType local_sender_channels __attribute__((used));

// These are used by the other files
bool on_mmio_chip __attribute__((used));

volatile HostInterface* host_interface __attribute__((used));

WriteTridTracker receiver_channel_0_trid_tracker __attribute__((used));

OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple __attribute__((used));

ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple __attribute__((used));

// object_init is expected to be called before this
__attribute__((noinline)) void service_lite_fabric() {
    invalidate_l1_cache();
    lite_fabric::run_sender_channel_step();

    invalidate_l1_cache();
    lite_fabric::run_receiver_channel_step();
}

inline void object_init(volatile lite_fabric::LiteFabricMemoryMap* mem_map) {
    local_sender_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});
    remote_receiver_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
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

    lite_fabric::remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        RECEIVER_CHANNEL_BASE_ID);
    lite_fabric::init_receiver_headers(lite_fabric::remote_receiver_channels);

    lite_fabric::local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        SENDER_CHANNEL_BASE_ID);

    (lite_fabric::receiver_channel_pointers_tuple.template get<0>()).reset();
    lite_fabric::on_mmio_chip = mem_map->config.is_mmio;
    lite_fabric::host_interface = &mem_map->host_interface;
    mem_map->service_lite_fabric_addr = reinterpret_cast<uint32_t>(&service_lite_fabric);
    lite_fabric::host_interface->init();
}

inline void data_init() {
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    wzerorange(__ldm_bss_start, __ldm_bss_end);
}

inline void teardown(volatile lite_fabric::LiteFabricMemoryMap* mem_map) {
    lite_fabric::receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    lite_fabric::ConnectedRisc1Interface::assert_connected_dm1_reset();

    mem_map->config.current_state = lite_fabric::InitState::READY;
}

}  // namespace lite_fabric

int main() {
    invalidate_l1_cache();
    configure_csr();
    noc_index = NOC_INDEX;
    lite_fabric::data_init();
    noc_bank_table_init(LITE_FABRIC_INIT_BANK_TO_NOC_SCRATCH);
    risc_init();
    noc_init(MEM_LITE_FABRIC_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(LITE_FABRIC_CONFIG_START);
    lite_fabric::object_init(structs);
    lite_fabric::routing_init(&structs->config);

    invalidate_l1_cache();
    while (structs->config.routing_enabled) {
        lite_fabric::service_lite_fabric();
    }

    lite_fabric::teardown(structs);

    return 0;
}
