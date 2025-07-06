// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

struct unicast_mode {
    uint8_t distance;
};
struct mcast_mode {
    uint8_t distance;
    uint8_t range;
};

union transmit_config {
    unicast_mode unicast;
    mcast_mode mcast;
};

// Worker core - Data Movement Writer -> Sends to Erisc Data Mover (sender side).
// -> takes input from local cb and pushes to erisc L1
void kernel_main() {
    // Test doesn't support multiple pages per send yet since we are writing
    // to interleaved which will never have subsequent pages on the same core
    // (and hence, able to share a packet header)
    constexpr uint32_t total_pages_to_send = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    // constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);
    constexpr bool dest_is_dram = get_compile_time_arg_val(2) != 0;
    constexpr bool mcast_mode = get_compile_time_arg_val(3) == 1;
    constexpr bool write_scatter_mode = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_pages_per_send = (write_scatter_mode ? 2 : 1);

    DPRINT << "sws: args " << "\n\tnum_pages_to_send=" << total_pages_to_send << "\n\tpage_size="
           << page_size
           //    << "\n\tnum_buffers_per_channel=" << num_buffers_per_channel
           << "\n\tdest_is_dram=" << (dest_is_dram ? "T" : "F") << "\n\tmcast_mode=" << (mcast_mode ? "T" : "F")
           << "\n\twrite_scatter_mode=" << (write_scatter_mode ? "T" : "F") << "\n";

    size_t arg_idx = 0;
    // Nearly all of the following arguments are needed to establish a connection with
    // EDM.
    // FUTURE WORK to make the connection info more compact. This will include:
    // 1. packing EDM noc x/y into one RT arg
    // 2. packing all semaphores as IDs and those IDs into the same RT arg
    //    We should be able to comfortably fit 4 into a single arg
    // 3. All other fields should be derivable from an EDM channel ID,
    //    which can then be used to statically compute offsets into EDM unreserved L1
    //    according to the static EDM L1 allocation scheme.
    //    This should let us get away with describing the full connection in 3-4 args total
    // const uint32_t eth_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    // // erisc l1 semaphore address
    // const uint32_t eth_sender_l1_sem_id = get_arg_val<uint32_t>(arg_idx++);
    // volatile uint32_t* const writer_send_sem_addr =
    //     reinterpret_cast<volatile uint32_t* const>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    // volatile uint32_t* const worker_teardown_sem_addr =
    //     reinterpret_cast<volatile uint32_t* const>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    // const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    // const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    // const uint32_t num_buffers_per_edm_channel = get_arg_val<uint32_t>(arg_idx++);
    // size_t edm_connection_handshake_id = get_arg_val<uint32_t>(arg_idx++);
    // size_t edm_worker_location_info_addr = get_arg_val<uint32_t>(arg_idx++);
    // size_t edm_buffer_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    size_t dest_addr = get_arg_val<uint32_t>(arg_idx++);
    // For global semaphore, we get the address directly (not a semaphore ID)
    volatile uint32_t* const last_message_semaphore_address =
        reinterpret_cast<volatile uint32_t* const>(get_arg_val<uint32_t>(arg_idx++));
    // Note: Semaphore initialization moved to receiver kernel
    // auto worker_buffer_index_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    // TODO: move to semaphore
    // auto edm_buffer_index_sem_id = get_arg_val<uint32_t>(arg_idx++);
    // auto edm_buffer_index_id = edm_buffer_index_sem_id;
    // ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(writer_send_sem_addr));
    // ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(worker_teardown_sem_addr));
    // ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(last_message_semaphore_address));
    auto packet_header_buffer_cb_id = get_arg_val<uint32_t>(arg_idx++);

    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    transmit_config config;
    if (mcast_mode) {
        config.mcast.distance = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        config.mcast.range = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    } else {
        config.unicast.distance = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    }

    // Get receiver NOC coordinates for global semaphore signaling
    const uint32_t receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const InterleavedAddrGen<dest_is_dram> dest_addr_gen = {
        .bank_base_address = dest_addr, .page_size = page_size};

    // ASSERT(num_buffers_per_channel > 0);
    // auto sender = tt::tt_fabric::WorkerToFabricEdmSender(
    //     true,  // persistent fabric (always true)
    //     0,
    //     eth_sender_noc_x,
    //     eth_sender_noc_y,
    //     eth_l1_base_addr,
    //     num_buffers_per_channel,
    //     eth_sender_l1_sem_id,

    //     edm_connection_handshake_id,
    //     edm_worker_location_info_addr,
    //     edm_buffer_size_bytes + sizeof(PACKET_HEADER_TYPE),
    //     edm_buffer_index_id,
    //     writer_send_sem_addr,
    //     worker_teardown_sem_addr,
    //     worker_buffer_index_semaphore_addr,
    //     tt::tt_fabric::WorkerToFabricEdmSenderImpl<0>::sender_channel_0_free_slots_stream_id,
    //     StreamId{std::numeric_limits<uint32_t>::max()});

    sender.open();

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    // We need to normalize all noc addresses to be for a consistent noc ID
    // so the remote sender core can correctly send the packet. In the future
    // we can decide if it's better for the noc index to be embedded in the packet
    // header (for now we don't do that)
    constexpr size_t NORMALIZED_NOC_INDEX = 0;

    uint32_t buffer_index = 0;
    cb_wait_front(cb_id_in0, 1);

    cb_reserve_back(packet_header_buffer_cb_id, 1);

    auto packet_header_addr = get_write_ptr(packet_header_buffer_cb_id);
    auto* packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_addr);
    for (uint32_t p = 0; p < total_pages_to_send; p += num_pages_per_send) {
        uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - p);

        sender.wait_for_empty_write_slot();

        cb_wait_front(cb_id_in0, pages_to_send);

        // bit of a hack to extract X/Y
        const auto dest_noc_address = get_noc_addr(p, dest_addr_gen, 0, NORMALIZED_NOC_INDEX);
        auto payload_addr = get_read_ptr(cb_id_in0);
        if constexpr (mcast_mode) {
            packet_header
                ->to_chip_multicast(
                    tt::tt_fabric::MulticastRoutingCommandHeader{config.mcast.distance, config.mcast.range})
                ->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{dest_noc_address}, (pages_to_send * page_size));
        } else {
#ifdef ARCH_WORMHOLE
            if (write_scatter_mode && pages_to_send == 2) {
                uint64_t dest_noc_address2 = get_noc_addr(p + 1, dest_addr_gen, 0, NORMALIZED_NOC_INDEX);
                packet_header->to_chip_unicast(config.unicast.distance)
                    ->to_noc_unicast_scatter_write(
                        tt::tt_fabric::NocUnicastScatterCommandHeader{
                            {dest_noc_address, dest_noc_address2}, (uint16_t)page_size},
                        (pages_to_send * page_size));
            } else
#endif
            {
                packet_header->to_chip_unicast(config.unicast.distance)
                    ->to_noc_unicast_write(
                        tt::tt_fabric::NocUnicastCommandHeader{dest_noc_address}, (pages_to_send * page_size));
            }
        }

        sender.send_payload_without_header_non_blocking_from_address(payload_addr, pages_to_send * page_size);
        sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

        noc_async_writes_flushed();
        cb_pop_front(cb_id_in0, pages_to_send);
    }

    if constexpr (!mcast_mode) {
        sender.wait_for_empty_write_slot();

        // Send completion signal to receiver on remote device
        // Note: We no longer initialize or wait for the semaphore here
        // The receiver will initialize and wait for it
        // Compute the NOC address of the global semaphore on the receiver device
        uint64_t last_message_semaphore_noc0_addr =
            safe_get_noc_addr(receiver_noc_x, receiver_noc_y, (uint32_t)last_message_semaphore_address, 0);
        packet_header->to_chip_unicast(config.unicast.distance);
        packet_header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(last_message_semaphore_noc0_addr, 1, 32));

        sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    sender.close();
}
