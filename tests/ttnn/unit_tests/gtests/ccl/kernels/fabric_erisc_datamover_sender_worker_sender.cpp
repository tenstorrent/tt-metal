// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"

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
    constexpr uint32_t num_pages_per_send = 1;  // get_compile_time_arg_val(0);
    constexpr uint32_t total_pages_to_send = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);
    constexpr bool dest_is_dram = get_compile_time_arg_val(4) != 0;
    constexpr bool mcast_mode = get_compile_time_arg_val(5) == 1;

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
    const uint32_t eth_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    // erisc l1 semaphore address
    const uint32_t eth_sender_l1_sem_id = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* const writer_send_sem_addr =
        reinterpret_cast<volatile uint32_t* const>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_buffers_per_edm_channel = get_arg_val<uint32_t>(arg_idx++);
    size_t edm_connection_handshake_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    size_t edm_worker_location_info_addr = get_arg_val<uint32_t>(arg_idx++);
    size_t edm_buffer_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    size_t dest_addr = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* const last_message_semaphore_address =
        reinterpret_cast<volatile uint32_t* const>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    *last_message_semaphore_address = 0;
    auto worker_buffer_index_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    // TODO: move to semaphore
    auto edm_buffer_index_sem_id = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(edm_buffer_index_sem_id < 8);
    auto edm_buffer_index_address = get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_buffer_index_sem_id);
    ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(writer_send_sem_addr));
    ASSERT(worker_buffer_index_semaphore_addr != reinterpret_cast<size_t>(last_message_semaphore_address));

    transmit_config config;
    if (mcast_mode) {
        config.mcast.distance = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        config.mcast.range = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    } else {
        config.unicast.distance = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    }

    const InterleavedAddrGen<dest_is_dram> dest_addr_gen = {.bank_base_address = dest_addr, .page_size = page_size};

    ASSERT(num_buffers_per_channel > 0);
    auto sender = tt::fabric::WorkerToFabricEdmSender(
        eth_sender_noc_x,
        eth_sender_noc_y,
        eth_l1_base_addr,
        num_buffers_per_channel,
        eth_sender_l1_sem_id,

        edm_connection_handshake_addr,
        edm_worker_location_info_addr,
        edm_buffer_size_bytes,
        edm_buffer_index_address,
        writer_send_sem_addr,
        worker_buffer_index_semaphore_addr);

    sender.open();

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    // We need to normalize all noc addresses to be for a consistent noc ID
    // so the remote sender core can correctly send the packet. In the future
    // we can decide if it's better for the noc index to be embedded in the packet
    // header (for now we don't do that)
    constexpr size_t NORMALIZED_NOC_INDEX = 0;

    uint32_t buffer_index = 0;
    cb_wait_front(cb_id_in0, 1);
    auto a_packet_header_addr = get_read_ptr(cb_id_in0);
    for (uint32_t p = 0; p < total_pages_to_send; p += num_pages_per_send) {
        uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - p);
        sender.wait_for_empty_write_slot();
        cb_wait_front(cb_id_in0, pages_to_send);

        // bit of a hack to extract X/Y
        const auto dest_noc_address = get_noc_addr(p, dest_addr_gen, 0, NORMALIZED_NOC_INDEX);
        const size_t dest_addr = dest_noc_address & 0xFFFFFFFF;
        const size_t dest_noc_x = (dest_noc_address >> NOC_ADDR_LOCAL_BITS) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
        const size_t dest_noc_y =
            (dest_noc_address >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
        const size_t packet_size = page_size + sizeof(tt::fabric::PacketHeader);

        auto packet_addr = get_read_ptr(cb_id_in0);
        auto& packet_header = *reinterpret_cast<tt::fabric::PacketHeader*>(packet_addr);
        if constexpr (mcast_mode) {
            packet_header.to_write()
                .to_chip_multicast(tt::fabric::MulticastRoutingCommandHeader{config.mcast.distance, config.mcast.range})
                .to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
                    dest_addr,
                    (pages_to_send * page_size) + sizeof(tt::fabric::PacketHeader),
                    static_cast<uint8_t>(dest_noc_x),
                    static_cast<uint8_t>(dest_noc_y)});
            packet_header.reserved2 = 0x1111;  // debug only
        } else {
            packet_header.to_write()
                .to_chip_unicast(tt::fabric::UnicastRoutingCommandHeader{config.unicast.distance})
                .to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
                    dest_addr,
                    (pages_to_send * page_size) + sizeof(tt::fabric::PacketHeader),
                    static_cast<uint8_t>(dest_noc_x),
                    static_cast<uint8_t>(dest_noc_y)});
            packet_header.reserved2 = 0x1111;  // debug only
        }

        uint64_t buffer_address = sender.edm_buffer_addr +
                                  (*sender.buffer_index_ptr * (sender.buffer_size_bytes + sizeof(eth_channel_sync_t)));
        sender.send_payload_blocking_from_address(packet_addr, packet_size);
        noc_async_writes_flushed();
        cb_pop_front(cb_id_in0, pages_to_send);
    }

    if constexpr (!mcast_mode) {
        sender.wait_for_empty_write_slot();

        auto& packet_header = *reinterpret_cast<tt::fabric::PacketHeader*>(a_packet_header_addr);
        ASSERT(*last_message_semaphore_address == 0);
        packet_header.reserved = 0xE;
        packet_header.reserved2 = 0xFFFF;
        packet_header.to_atomic_inc();
        packet_header.to_chip_unicast(tt::fabric::UnicastRoutingCommandHeader{2});
        packet_header.to_noc_unicast_atomic_inc(tt::fabric::NocUnicastAtomicIncCommandHeader(
            reinterpret_cast<size_t>(last_message_semaphore_address), 1, 32, my_x[0], my_y[0]));

        sender.send_payload_blocking_from_address(
            a_packet_header_addr, packet_header.get_payload_size_including_header());

        noc_semaphore_wait(last_message_semaphore_address, 1);
    }

    bool closed = false;
    size_t num_endpoints_to_terminate = get_arg_val<uint32_t>(arg_idx++);
    for (size_t i = 0; i < num_endpoints_to_terminate; i++) {
        size_t edm_noc_x = get_arg_val<uint32_t>(arg_idx++);
        size_t edm_noc_y = get_arg_val<uint32_t>(arg_idx++);
        size_t distance = get_arg_val<uint32_t>(arg_idx++);
        size_t termination_addr = get_arg_val<uint32_t>(arg_idx++);

        if (!closed && distance == 0) {
            closed = true;
            sender.close();
        }
        if (distance == 0) {
            noc_inline_dw_write(
                get_noc_addr(edm_noc_x, edm_noc_y, termination_addr),
                tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE);
        } else {
            auto& packet_header = *reinterpret_cast<tt::fabric::PacketHeader*>(a_packet_header_addr);
            reinterpret_cast<volatile uint32_t*>(a_packet_header_addr)[sizeof(tt::fabric::PacketHeader) >> 2] =
                tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE;
            sender.wait_for_empty_write_slot();
            packet_header.to_write()
                .to_chip_unicast(tt::fabric::UnicastRoutingCommandHeader{static_cast<uint8_t>(distance/* - 1*/)})
                .to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
                    termination_addr,
                    sizeof(tt::fabric::PacketHeader) + sizeof(uint32_t),
                    static_cast<uint8_t>(edm_noc_x),
                    static_cast<uint8_t>(edm_noc_y)});
            sender.send_payload_blocking_from_address(
                a_packet_header_addr, packet_header.get_payload_size_including_header());
            noc_async_writes_flushed();
        }
    }
    if (!closed) {
        sender.close();
    }
}
