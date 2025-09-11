// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

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
    size_t dest_addr = get_arg_val<uint32_t>(arg_idx++);
    // For global semaphore, we get the address directly (not a semaphore ID)
    volatile uint32_t* const last_message_semaphore_address =
        reinterpret_cast<volatile uint32_t* const>(get_arg_val<uint32_t>(arg_idx++));

    ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(writer_send_sem_addr));
    ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(worker_teardown_sem_addr));
    ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(last_message_semaphore_address));

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

    constexpr auto dst_args = TensorAccessorArgs<5>();
    const auto dest_addr_gen = TensorAccessor(dst_args, dest_addr, page_size);

    sender.open<true>();

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    // We need to normalize all noc addresses to be for a consistent noc ID
    // so the remote sender core can correctly send the packet. In the future
    // we can decide if it's better for the noc index to be embedded in the packet
    // header (for now we don't do that)
    constexpr size_t NORMALIZED_NOC_INDEX = 0;

    uint32_t buffer_index = 0;
    cb_wait_front(cb_id_in0, 1);

    auto* packet_header = PacketHeaderPool::allocate_header();
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
            if (write_scatter_mode && pages_to_send == 2) {
                uint64_t dest_noc_address2 = get_noc_addr(p + 1, dest_addr_gen, 0, NORMALIZED_NOC_INDEX);
                packet_header->to_chip_unicast(config.unicast.distance)
                    ->to_noc_unicast_scatter_write(
                        tt::tt_fabric::NocUnicastScatterCommandHeader{
                            {dest_noc_address, dest_noc_address2}, (uint16_t)page_size},
                        (pages_to_send * page_size));
            } else {
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

    // Send completion signal to receiver on remote device
    // Note: We no longer initialize or wait for the semaphore here
    // The receiver will initialize and wait for it
    // Compute the NOC address of the global semaphore on the receiver device
    uint64_t last_message_semaphore_noc0_addr =
        safe_get_noc_addr(receiver_noc_x, receiver_noc_y, (uint32_t)last_message_semaphore_address, 0);
    if constexpr (!mcast_mode) {
        packet_header->to_chip_unicast(config.unicast.distance);
    } else {
        packet_header->to_chip_unicast(config.mcast.distance + config.mcast.range - 1);
    }
    packet_header->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(last_message_semaphore_noc0_addr, 1, 32));

    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

    sender.close();
}
