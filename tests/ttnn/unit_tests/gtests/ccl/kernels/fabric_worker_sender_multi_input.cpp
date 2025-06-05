// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_utils.hpp"

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

enum class ReadMode {
    // Fully drain one input CB before advancing to the next
    FULLY_ORDERED,
    // Drain both inputs in at some specified ratio (e.g. if 5:3 then 3 pages from CB0 then 5 from CB1)
    RATIOD_FORWARDING,
    // Read pages from either CB as soon as they are available
    ARBITRARILY_ORDERED
};

static constexpr size_t NORMALIZED_NOC_INDEX = 0;

template <typename AddrGen>
auto forward_to_fabric_from_cb(
    size_t total_pages_to_send,
    tt::tt_fabric::WorkerToFabricEdmSender& sender,
    uint32_t cb_id,
    const transmit_config& config,
    size_t page_size,
    const AddrGen& dest_addr_gen,
    size_t num_pages_per_send,
    size_t current_page) {
    // for (uint32_t p = 0; p < total_pages_to_send; p += num_pages_per_send) {
    uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - current_page);
    sender.wait_for_empty_write_slot();

    // bit of a hack to extract X/Y
    const auto noc0_dest_address = get_noc_addr(current_page, dest_addr_gen, 0, NORMALIZED_NOC_INDEX);
    const size_t packet_size = page_size + sizeof(PACKET_HEADER_TYPE);

    auto packet_addr = get_read_ptr(cb_id);
    auto& packet_header = *reinterpret_cast<PACKET_HEADER_TYPE*>(packet_addr);
    if constexpr (mcast_mode) {
        packet_header
            .to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{config.mcast.distance, config.mcast.range})
            .to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_address}, (pages_to_send * page_size));
    } else {
        packet_header.to_chip_unicast(config.unicast.distance)
            .to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_address}, (pages_to_send * page_size));
    }

    uint64_t buffer_address = sender.edm_buffer_addr + (*sender.buffer_index_ptr * (sender.buffer_size_bytes + sizeof(eth_channel_sync_t)));
    sender.send_payload_blocking_from_address(packet_addr, packet_size);
    noc_async_writes_flushed();
    // }
}

template <typename AddrGen>
void non_blocking_read_and_forward(
    size_t& current_page_in,
    uint32_t cb_id,
    const AddrGen& dest_addr_gen,
    tt::tt_fabric::WorkerToFabricEdmSender& sender,
    const transmit_config& config,
    uint32_t page_size,
    uint32_t total_pages_to_send,
    uint32_t num_pages_per_send) {
    uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - current_page_in);
    if (!cb_pages_available_at_front(cb_id, pages_to_send)) {
        return;
    }

    current_page_in += num_pages_per_send;
    cb_wait_front(cb_id, pages_to_send);
    forward_to_fabric_from_cb(
        total_pages_to_send,
        sender,
        cb_id,
        config,
        page_size,
        dest_addr_gen,
        num_pages_per_send,
        current_page_in
        );
    cb_pop_front(cb_id, pages_to_send);
}

// Worker core - Data Movement Writer -> Sends to Erisc Data Mover (sender side).
// -> takes input from local cb and pushes to erisc L1
void kernel_main() {

    // Test doesn't support multiple pages per send yet since we are writing
    // to interleaved which will never have subsequent pages on the same core
    // (and hence, able to share a packet header)
    constexpr uint32_t num_pages_per_send = 1;//get_compile_time_arg_val(0);
    constexpr uint32_t total_pages_to_send = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);
    constexpr bool dest0_is_dram = get_compile_time_arg_val(4) != 0;
    constexpr bool dest1_is_dram = get_compile_time_arg_val(5) != 0;
    constexpr ReadMode read_mode = static_cast<ReadMode>(get_compile_time_arg_val(6));

    transmit_config config;
    size_t arg_idx = 0;
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    volatile uint32_t* const last_message_semaphore_address = reinterpret_cast<volatile uint32_t* const >(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    size_t output_buffer0_addr = get_arg_val<uint32_t>(arg_idx++);
    size_t output_buffer1_addr = get_arg_val<uint32_t>(arg_idx++);
    config.unicast.distance = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));

    size_t read_ratio0 = (read_mode == ReadMode::ARBITRARILY_ORDERED) ? 0 :
                          (read_mode == ReadMode::FULLY_ORDERED) ? total_pages_to_send :
                          get_arg_val<uint32_t>(arg_idx++);
    size_t read_ratio1 = (read_mode == ReadMode::ARBITRARILY_ORDERED) ? 0 :
                          (read_mode == ReadMode::FULLY_ORDERED) ? total_pages_to_send :
                          get_arg_val<uint32_t>(arg_idx++);


    *last_message_semaphore_address = 0;
    const InterleavedAddrGen<dest0_is_dram> dest_addr_gen0 = {
        .bank_base_address = output_buffer0_addr, .page_size = page_size};
    const InterleavedAddrGen<dest1_is_dram> dest_addr_gen1 = {
        .bank_base_address = output_buffer1_addr, .page_size = page_size};

    ASSERT(num_buffers_per_channel > 0);

    sender.open();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in0;

    // We need to normalize all noc addresses to be for a consistent noc ID
    // so the remote sender core can correctly send the packet. In the future
    // we can decide if it's better for the noc index to be embedded in the packet
    // header (for now we don't do that)
    constexpr size_t NORMALIZED_NOC_INDEX = 0;

    cb_wait_front(cb_id_in0, 1);
    auto a_packet_header_addr = get_read_ptr(cb_id_in0);

    if constexpr (read_mode == ReadMode::FULLY_ORDERED || read_mode == ReadMode::RATIOD_FORWARDING) {

        size_t current_page_in0 = 0;
        size_t current_page_in1 = 0;
        while (current_page_in0 < total_pages_to_send || current_page_in1 < total_pages_to_send) {
            for (size_t read = 0; read < read_ratio0 && current_page_in0 < total_pages_to_send; current_page_in0 += num_pages_per_send, read++) {
                uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - current_page_in0);
                cb_wait_front(cb_id_in0, pages_to_send);
                non_blocking_read_and_forward(current_page_in0, cb_id_in0, dest_addr_gen0, sender, config, page_size, total_pages_to_send, num_pages_per_send);
                cb_pop_front(cb_id_in0, pages_to_send);
            }

            for (size_t read = 0; read < read_ratio1 && current_page_in1 < total_pages_to_send; current_page_in1 += num_pages_per_send, read++) {
                uint32_t pages_to_send = std::min<uint32_t>(num_pages_per_send, total_pages_to_send - current_page_in1);
                cb_wait_front(cb_id_in1, pages_to_send);
                non_blocking_read_and_forward(current_page_in1, cb_id_in1, dest_addr_gen1, sender, config, page_size, total_pages_to_send, num_pages_per_send);
                cb_pop_front(cb_id_in1, pages_to_send);
            }
        }

    } else if constexpr (read_mode == ReadMode::ARBITRARILY_ORDERED) {
        size_t current_page_in0 = 0;
        size_t current_page_in1 = 0;
        while (current_page_in0 < total_pages_to_send || current_page_in1 < total_pages_to_send) {
            if (current_page_in0 < total_pages_to_send) {
                non_blocking_read_and_forward(current_page_in0, cb_id_in0, dest_addr_gen0, sender, config, page_size, total_pages_to_send, num_pages_per_send);
            }
            if (current_page_in1 < total_pages_to_send) {
                non_blocking_read_and_forward(current_page_in1, cb_id_in1, dest_addr_gen1, sender, config, page_size, total_pages_to_send, num_pages_per_send);
            }
        }
    }

    sender.wait_for_empty_write_slot();

    constexpr size_t kLoopbackNumHopsToMyChip = 2;
    auto& packet_header = *reinterpret_cast<PACKET_HEADER_TYPE*>(a_packet_header_addr);
    ASSERT(*last_message_semaphore_address == 0);
    packet_header.reserved = 0xE;
    packet_header.reserved2 = 0xFFFF;
    uint64_t last_message_sem_noc_addr = get_noc_addr(my_x[0], my_y[0], last_message_semaphore_address);
    packet_header.to_chip_unicast(kLoopbackNumHopsToMyChip);
    packet_header.to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(last_message_sem_noc_addr, 1, 32));

    sender.send_payload_blocking_from_address(a_packet_header_addr, packet_header.get_payload_size_including_header());

    noc_semaphore_wait(last_message_semaphore_address, 1);

    sender.close();
}
